"""Molecular property computation pipeline.

ORCA jobs run in parallel; a single serial worker extracts HOMO/LUMO,
dipole, ESP/MPI, volume, density, and RDKit descriptors via Multiwfn
(which must not run concurrently with itself).

Input:
    smiles                     -- DataFrame with a 'SMILES' column.
    method, basis_set          -- ORCA level of theory.
    opt, freq, tightscf        -- ORCA job flags.
    ncore, multithreading,     -- parallelism budget; 'cpu_usage' is the
    cpu_usage                    fraction of physical cores to occupy.
    random_seed                -- RDKit 3D embedding seed.
    mol_calc_queue             -- optional; one message per event.
    output_path                -- root for per-molecule directories.

Output ('mol_calc' / 'prop_gen' return 'None' / 'DataFrame'):
    <output_path>/<mol_sha1>/   per-molecule .inp .out .png .molden.input
    <output_path>/mapping.csv   SMILES -> folder mapping (append-only).
    <output_path>/mol_prop.csv  one row per molecule: HOMO, LUMO, Dipole,
                                ESP_max, ESP_min, MPI, Vol, Rho, MW, TPSA,
                                LogP, HBA, HBD, FCsp3, Num_AR, Num_RB,
                                Num_sp, Num_sp2, Num_sp3, Num_HA, Num_SC.
    MolCalc.prop_df             in-memory 'mol_prop.csv'.
    queue messages              '[T<tid>] [ORCA|Prop] <status> <smiles>'
                                and '_progr<float>'.

'transform(experiment_condition, smiles, mol_prop)' builds the training
matrices from a mixture-level table; returns '(ml_dataset, dl_dataset)'.
"""

import hashlib
import os
import queue
import shutil
import subprocess
import threading
import time
from queue import Queue

import numpy as np
import pandas as pd
import psutil
import pygetwindow as gw
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    rdMolDescriptors,
    rdchem,
)
from skfp.fingerprints import KlekotaRothFingerprint
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Process-environment helper
# ---------------------------------------------------------------------------

def _single_thread_env():
    """Return a copy of 'os.environ' pinned to one BLAS/OpenMP thread.

    '%pal nprocs N' only caps ORCA MPI ranks; each rank still spawns
    'OMP_NUM_THREADS' workers, defaulting to the system logical-core count.
    Without this override 'nprocs 4' on a 16-core host can use 64 threads.
    """
    env = os.environ.copy()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = "1"
    return env


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def smiles_to_folder(smile, method, basis_set):
    """Deterministic per-molecule directory name (12-char SHA1 prefix)."""
    key = f"{smile}|{method}|{basis_set}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"mol_{h}"


def is_orca_completed(out_file, property_file=None, tail_size=8000):
    """Return True if ORCA has finished normally for this molecule."""
    if property_file and os.path.isfile(property_file):
        try:
            if os.path.getsize(property_file) > 0:
                with open(property_file, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(20000)
                if "NORMAL TERMINATION" in head:
                    return True
        except Exception:
            pass

    if os.path.isfile(out_file):
        try:
            size = os.path.getsize(out_file)
            if size > 0:
                with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
                    if size > tail_size:
                        f.seek(size - tail_size)
                    tail = f.read()
                if "ORCA TERMINATED NORMALLY" in tail:
                    return True
        except Exception:
            pass

    return False


def format_con(orca_2mkl_path, file_path, file_name):
    """Convert an ORCA ''.gbw'' to a Molden input via ''orca_2mkl''."""
    try:
        subprocess.run(
            [orca_2mkl_path, f"{file_name}", "-molden"],
            check=True,
            text=True,
            cwd=file_path,
            capture_output=True,
            timeout=30,
            env=_single_thread_env(),
        )
        return "Format conversion success"
    except subprocess.CalledProcessError:
        return "Format conversion error"
    except subprocess.TimeoutExpired:
        return "Format conversion timeout"
    except Exception:
        return "Unknown exception occurred"


# ---------------------------------------------------------------------------
# Multiwfn / RDKit extractors
# ---------------------------------------------------------------------------

def _parse_homo_lumo(path):
    """Parse HOMO/LUMO energies from 'path'; returns '(homo, lumo)'."""
    homo, lumo = [], []
    try:
        with open(path, "r") as f:
            for line in f:
                if "HOMO, energy" in line:
                    for tok in line.split():
                        try:
                            homo.append(float(tok))
                        except ValueError:
                            pass
                if "LUMO, energy" in line:
                    for tok in line.split():
                        try:
                            lumo.append(float(tok))
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    return homo, lumo


def get_homo_lumo(file_path, file_name):
    """Extract HOMO and LUMO energies (eV) from a Molden file via Multiwfn.

    Multiwfn opens a GUI window after printing the orbital table, so the
    process is polled at 0.5 s intervals and terminated as soon as both
    sections have appeared.  Hard ceiling is 150 s (matching the original
    timeout) and the window is closed unconditionally on exit.
    """
    with open(f"{file_path}/multiwfn_cmd.txt", "w") as f:
        f.write("0\n")

    proc = None
    homo, lumo = [], []
    try:
        with open(f"{file_path}/humo_lumo_value.txt", "w") as f_out:
            with open(f"{file_path}/multiwfn_cmd.txt", "r") as cmd_in:
                proc = subprocess.Popen(
                    ["multiwfn", f"{file_name}.molden.input"],
                    cwd=file_path,
                    stdin=cmd_in,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                    env=_single_thread_env(),
                )
            deadline = time.time() + 150
            while time.time() < deadline:
                time.sleep(0.5)
                homo, lumo = _parse_homo_lumo(f"{file_path}/humo_lumo_value.txt")
                if homo and lumo:
                    break
    except Exception:
        pass
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for w in gw.getWindowsWithTitle("Geometry structure"):
            try:
                w.close()
            except Exception:
                pass
    return homo, lumo


def get_dipole(file_path, file_name):
    """Return the dipole-moment magnitude (Debye) reported by Multiwfn."""
    with open(f"{file_path}/multiwfn_cmd.txt", "w") as f:
        f.write("300\n5\n0\nq\n")
    dipole = []
    try:
        with open(f"{file_path}/multiwfn_cmd.txt", "r") as cmd_in:
            result = subprocess.run(
                ["multiwfn", f"{file_name}.molden.input"],
                cwd=file_path,
                stdin=cmd_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                timeout=1800,
                env=_single_thread_env(),
            )
        for line in result.stdout.split("\n"):
            if "Magnitude of dipole moment" in line:
                for tok in line.split():
                    try:
                        dipole.append(float(tok))
                    except ValueError:
                        pass
        return dipole
    except Exception:
        return []


def get_esp_mpi(file_path, file_name):
    """Return ESP extrema (a.u.) and the molecular polarity index (kcal/mol)."""
    with open(f"{file_path}/multiwfn_cmd.txt", "w") as f:
        f.write("12\n0\n-1\n-1\nq\n")
    try:
        with open(f"{file_path}/multiwfn_cmd.txt", "r") as cmd_in:
            result = subprocess.run(
                ["multiwfn", f"{file_name}.molden.input"],
                stdin=cmd_in,
                check=True,
                cwd=file_path,
                capture_output=True,
                text=True,
                timeout=3600,
                env=_single_thread_env(),
            )
        esp, mpi = [], []
        for line in result.stdout.split("\n"):
            if "Minimal value:" in line and "Maximal value:" in line:
                for tok in line.split():
                    try:
                        esp.append(float(tok))
                    except ValueError:
                        pass
            if "Molecular polarity index" in line:
                for tok in line.split():
                    try:
                        mpi.append(float(tok))
                    except ValueError:
                        pass
        return esp, mpi
    except Exception:
        return [], []


def get_vol_rho(file_path, file_name):
    """Return the van der Waals volume (A^3) and estimated density (g/cm^3)."""
    with open(f"{file_path}/multiwfn_cmd.txt", "w") as f:
        f.write("12\n6\n-1\n-1\nq\n")
    try:
        with open(f"{file_path}/multiwfn_cmd.txt", "r") as cmd_in:
            result = subprocess.run(
                ["multiwfn", f"{file_name}.molden.input"],
                stdin=cmd_in,
                check=True,
                cwd=file_path,
                capture_output=True,
                text=True,
                timeout=1800,
                env=_single_thread_env(),
            )
        vol, rho = [], []
        for line in result.stdout.split("\n"):
            if "Volume" in line:
                for tok in line.split():
                    try:
                        vol.append(float(tok))
                    except ValueError:
                        pass
            if "Estimated density" in line:
                for tok in line.split():
                    try:
                        rho.append(float(tok))
                    except ValueError:
                        pass
        return vol, rho
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

def _safe_get(lst, idx, default=float("nan")):
    """Index into ''lst'' returning ''default'' when out of range."""
    return lst[idx] if lst is not None and len(lst) > idx else default


def _short_tid(thread_id):
    """Last four characters of a thread identifier, used in log prefixes."""
    if thread_id is None:
        return "----"
    s = str(thread_id)
    return s[-4:] if len(s) > 4 else s


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class MolCalc:
    """Two-stage ORCA + property-extraction pipeline."""

    prop_dict = []
    prop_df = None
    smiles = None

    def __init__(
        self,
        smiles,
        method="B3LYP",
        basis_set="def2-SVP",
        opt=True,
        freq=True,
        tightscf=True,
        ncore=4,
        multithreading=True,
        cpu_usage=0.8,
        random_seed=0xF00D,
        mol_calc_queue=None,
    ):
        smiles.dropna(axis=0, how="all", inplace=True)
        smiles.dropna(axis=1, how="all", inplace=True)
        self.smiles = MolCalc.smiles = smiles
        self.orca_path = []
        self.method = method
        self.basis_set = basis_set
        self.opt = opt
        self.freq = freq
        self.tightscf = tightscf
        self.ncore = ncore
        self.multithreading = multithreading
        self.cpu_usage = cpu_usage
        self.random_seed = random_seed
        self.smiles_list = smiles["SMILES"].to_list()
        self.prop_lock = threading.Lock()
        self.mol_calc_queue = mol_calc_queue
        self._progr_total = 1
        self._progr_done = 0

        self._prop_queue = None
        self._prop_thread = None
        self._progr_lock = threading.Lock()
        # Lazy cache of (smiles, method, basis) already written to mol_prop.csv.
        self._prop_existing_keys = None

        self._log(f"Get_list ----> {self.smiles_list}")

    # ------------------------------------------------------------------
    # Logging / progress
    # ------------------------------------------------------------------

    def _log(self, msg, thread_id=None):
        full = f"[T{_short_tid(thread_id)}] {msg}" if thread_id is not None else str(msg)
        if self.mol_calc_queue is not None:
            try:
                self.mol_calc_queue.put(full)
            except Exception:
                pass

    def _set_progress(self, value):
        try:
            value = max(0.0, min(1.0, float(value)))
        except Exception:
            return
        if self.mol_calc_queue is not None:
            try:
                self.mol_calc_queue.put(f"_progr{value:.4f}")
            except Exception:
                pass

    def _init_progress(self, total):
        self._progr_total = max(1, int(total))
        self._progr_done = 0
        self._set_progress(0.0)

    def _tick_progress(self):
        with self._progr_lock:
            self._progr_done += 1
            done, total = self._progr_done, self._progr_total
        self._set_progress(done / total)

    def _enqueue_prop(self, smile):
        if self._prop_queue is None:
            return
        try:
            self._prop_queue.put(smile)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Parallelism budget
    # ------------------------------------------------------------------

    def _plan_parallelism(self):
        """Return '(num_workers, effective_cores, budget)'.

        'budget' is 'cpu_usage' of the effective core count minus one
        core reserved for the serial Multiwfn/RDKit worker.  Physical core
        count is clamped by the logical count because virtualized hosts
        over-report 'logical=False'.  'MOLCALC_MAX_CORES' overrides the
        detected count when the container quota is stricter than what
        'psutil' sees.
        """
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical
        effective = min(physical, logical)

        override = os.environ.get("MOLCALC_MAX_CORES", "").strip()
        if override.isdigit() and int(override) > 0:
            effective = int(override)

        budget = max(1, int(effective * self.cpu_usage) - 1)
        num_workers = max(1, budget // max(1, self.ncore))
        return num_workers, effective, budget

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def mol_calc(self, output_path):
        """Run the full pipeline (ORCA || property extraction) into ''output_path''."""
        self.orca_path = shutil.which("orca")
        if self.orca_path is None:
            self._log("Error: Cannot find ORCA program in PATH")
            return

        os.makedirs(output_path, exist_ok=True)

        # Maintain the append-only SMILES -> folder mapping.
        mapping_path = os.path.join(output_path, "mapping.csv")
        existing = set()
        if os.path.exists(mapping_path):
            try:
                old = pd.read_csv(mapping_path)
                existing = set(zip(old["SMILES"], old["method"], old["basis_set"]))
            except Exception:
                existing = set()

        rows = [
            {
                "SMILES": s,
                "folder": smiles_to_folder(s, self.method, self.basis_set),
                "method": self.method,
                "basis_set": self.basis_set,
            }
            for s in self.smiles_list
            if (s, self.method, self.basis_set) not in existing
        ]
        if rows:
            new_df = pd.DataFrame(rows)
            write_header = (not os.path.exists(mapping_path)) or os.path.getsize(mapping_path) == 0
            new_df.to_csv(mapping_path, mode="a", header=write_header,
                          index=False, encoding="utf-8-sig")

        # Partition into already-computed vs. pending molecules.
        pending_list, skipped_list = [], []
        for s in self.smiles_list:
            name = smiles_to_folder(s, self.method, self.basis_set)
            out_file = os.path.join(output_path, name, f"{name}.out")
            prop_file = os.path.join(output_path, name, f"{name}.property.txt")
            molden_file = os.path.join(output_path, name, f"{name}.molden.input")
            if is_orca_completed(out_file, prop_file) and os.path.isfile(molden_file):
                skipped_list.append(s)
            else:
                pending_list.append(s)

        self._init_progress(len(pending_list) + len(self.smiles_list))
        total_time_start = time.time()

        # Serial property-extraction worker (Multiwfn uses a fixed scratch
        # file, so concurrency is impossible).
        self._prop_queue = Queue()
        tq_prop = tqdm(desc="Property", total=len(self.smiles_list),
                       mininterval=0.8, dynamic_ncols=True, colour="blue", unit="mol")

        def _prop_worker():
            while True:
                try:
                    item = self._prop_queue.get()
                except Exception:
                    break
                if item is None:
                    self._prop_queue.task_done()
                    break
                try:
                    self.prop_gen(output_path, smiles_list=[item],
                                  tq=None, disable_tqdm=True)
                except Exception as e:
                    self._log(f"[Prop] Error  {item}: {e}")
                finally:
                    self._tick_progress()
                    tq_prop.update(1)
                    self._prop_queue.task_done()

        self._prop_thread = threading.Thread(target=_prop_worker, daemon=True)
        self._prop_thread.start()

        try:
            for s in skipped_list:
                self._enqueue_prop(s)

            if pending_list:
                self._log(
                    f"ORCA pending: {len(pending_list)}, "
                    f"skipped (ORCA done): {len(skipped_list)}"
                )
                tq_orca = tqdm(desc="ORCA", total=len(pending_list),
                               mininterval=0.8, dynamic_ncols=True, unit="mol")

                if self.multithreading:
                    num_threads, effective, budget = self._plan_parallelism()
                    expected = num_threads * self.ncore + 1
                    self._log(
                        f"Multithreading enabled, worker threads = {num_threads} "
                        f"(effective_cores={effective}, ncore={self.ncore}, "
                        f"cpu_usage={self.cpu_usage}, expected_cores={expected}/{budget})"
                    )
                    if expected > budget:
                        self._log(
                            f"WARNING: requested {expected} cores exceeds the budget "
                            f"({budget}); reduce ncore or cpu_usage."
                        )
                    lock = threading.Lock()
                    q = Queue()
                    for smile in pending_list:
                        q.put([smile, 0])
                    threads = []
                    for _ in range(int(num_threads)):
                        thread = threading.Thread(
                            target=self.thread_worker, args=(q, lock, tq_orca, output_path)
                        )
                        threads.append(thread)
                        thread.start()
                    q.join()
                    for thread in threads:
                        thread.join()
                else:
                    thread_id = threading.get_ident()
                    self._log("Single-thread mode")
                    lock = threading.Lock()
                    for smile in pending_list:
                        try:
                            self.orca_calc(smile, 0, output_path, tq=tq_orca,
                                           lock=lock, thread_id=thread_id)
                        except Exception as e:
                            self._log(f"[ORCA] Exception {smile}: {e}", thread_id)
                        finally:
                            tq_orca.update(1)
                            self._tick_progress()
                            self._enqueue_prop(smile)

                tq_orca.close()
            else:
                self._log(
                    f"All {len(self.smiles_list)} molecules already calculated "
                    f"by ORCA. Skip ORCA stage."
                )

            self._set_progress(1.0)
            self._log(f"All done in {(time.time() - total_time_start) / 60:.2f} min")
        finally:
            # Guarantee the property worker terminates even on exceptions.
            self._prop_queue.put(None)
            self._prop_queue.join()
            self._prop_thread.join()
            tq_prop.close()

    # ------------------------------------------------------------------
    # ORCA stage
    # ------------------------------------------------------------------

    def thread_worker(self, q: Queue, lock: threading.Lock, tq: tqdm, output_path):
        """Thread body: consume tasks from ''q'' and run :meth:`orca_calc`."""
        thread_id = threading.get_ident()
        self._log("Worker thread started", thread_id)
        while True:
            try:
                task = q.get(timeout=0.5)
                with lock:
                    tq.set_postfix(Thread=_short_tid(thread_id), Mol=task[0])
                try:
                    self.orca_calc(task[0], 0, output_path, tq=tq,
                                   lock=lock, thread_id=thread_id)
                except Exception as e:
                    self._log(f"[ORCA] Exception {task[0]}: {e}", thread_id)
                finally:
                    # Enqueue unconditionally: prop_gen degrades to NaN.
                    self._enqueue_prop(task[0])
                    with lock:
                        tq.update(1)
                    self._tick_progress()
                    q.task_done()
                if q.empty():
                    raise queue.Empty
            except queue.Empty:
                break
        self._log("Worker thread finished", thread_id)

    def _postfix(self, tq, lock, thread_id, mol, info):
        """Set a tqdm postfix under the caller's lock (no sleeping)."""
        if tq is None:
            return
        if lock is not None:
            with lock:
                tq.set_postfix(Thread=_short_tid(thread_id), Mol=mol, Info=info)
        else:
            tq.set_postfix(Thread=_short_tid(thread_id), Mol=mol, Info=info)

    def orca_calc(self, smile, index, output_path, tq: tqdm,
                  lock: threading.Lock = None, thread_id=None):
        """Run ORCA (with 3D embedding + ''orca_2mkl'') for a single SMILES."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)
        AllChem.MMFFOptimizeMolecule(mol)
        num_atoms = mol.GetNumAtoms()

        file_name = smiles_to_folder(smile, self.method, self.basis_set)
        file_path = os.path.join(output_path, file_name)
        os.makedirs(file_path, exist_ok=True)

        smiles_file = os.path.join(file_path, "smiles.txt")
        if not os.path.exists(smiles_file):
            with open(smiles_file, "w", encoding="utf-8") as f:
                f.write(f"SMILES: {smile}\nmethod: {self.method}\n"
                        f"basis_set: {self.basis_set}\n")

        png_path = os.path.join(file_path, f"{file_name}.png")
        if not os.path.exists(png_path):
            try:
                Draw.MolToFile(mol, png_path, size=(300, 300), kekulize=True)
            except Exception:
                pass

        orca_2mkl_path = shutil.which("orca_2mkl")
        out_file = os.path.join(os.path.abspath(file_path), f"{file_name}.out")
        prop_file = os.path.join(os.path.abspath(file_path), f"{file_name}.property.txt")

        if is_orca_completed(out_file, prop_file):
            molden_file = os.path.join(os.path.abspath(file_path), f"{file_name}.molden.input")
            if not os.path.isfile(molden_file):
                if orca_2mkl_path is not None:
                    info = format_con(orca_2mkl_path, file_path, file_name)
                    self._postfix(tq, lock, thread_id, smile, info)
                    self._log(f"[Skip] {smile} — {info}", thread_id)
                    return None
                self._postfix(tq, lock, thread_id, smile, "Cannot find orca_2mkl program")
                self._log(f"[Skip] {smile} — orca_2mkl not found", thread_id)
                return None
            self._postfix(tq, lock, thread_id, smile, "Calculation completed")
            self._log(f"[Skip] {smile} — already completed", thread_id)
            return None

        self._log(f"[ORCA] Start  {smile}", thread_id)
        with open(os.path.join(file_path, f"{file_name}.inp"), "w") as f:
            conf = mol.GetConformer()
            OPT = "OPT" if self.opt else ""
            FREQ = "FREQ" if self.freq else ""
            TightSCF = "TightSCF" if self.tightscf else ""
            f.write(
                f"! {self.method} {self.basis_set} {OPT} {FREQ} {TightSCF}\n"
                f"%pal\n nprocs {self.ncore} \nend \n"
                f"%output\n Print[ P_MOs ] 1 \n PrintLevel Normal \nend\n"
            )
            f.write("* xyz 0 1 \n")
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                symbol = mol.GetAtomWithIdx(i).GetSymbol()
                f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
            f.write("*")

        self._postfix(tq, lock, thread_id, smile, f"Generated {file_name}.inp")

        try:
            inp_file = os.path.join(os.path.abspath(file_path), f"{file_name}.inp")
            self._postfix(tq, lock, thread_id, smile, "ORCA is calculating...")
            with open(out_file, "w") as f_out:
                subprocess.run(
                    [self.orca_path, inp_file],
                    cwd=file_path,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    env=_single_thread_env(),
                )
            self._log(f"[ORCA] Done   {smile}", thread_id)
        except subprocess.TimeoutExpired:
            self._postfix(tq, lock, thread_id, smile, "ORCA timeout")
            self._log(f"[ORCA] Timeout {smile}", thread_id)
        except subprocess.CalledProcessError as e:
            self._postfix(tq, lock, thread_id, smile, f"ORCA failed rc={e.returncode}")
            self._log(f"[ORCA] Failed {smile}: returncode={e.returncode}", thread_id)

        if orca_2mkl_path is not None:
            info = format_con(orca_2mkl_path, file_path, file_name)
            self._log(f"[mkl ] {smile} — {info}", thread_id)
        else:
            self._postfix(tq, lock, thread_id, smile, "Cannot find orca_2mkl program")
            self._log(f"[mkl ] orca_2mkl not found for {smile}", thread_id)
        return None

    # ------------------------------------------------------------------
    # Property stage
    # ------------------------------------------------------------------

    def prop_gen(self, input_path, output_file="mol_prop", smiles_list=None,
                 tq=None, disable_tqdm=False, thread_id=None):
        property_config = {
            "homo_lumo": ["HOMO", "LUMO"],
            "dipole": ["Dipole"],
            "esp_mpi": ["ESP_max", "ESP_min", "MPI"],
            "vol_rho": ["Vol", "Rho"],
            "mw": ["MW"],
            "tpsa": ["TPSA"],
            "logp": ["LogP"],
            "hba_hbd": ["HBA", "HBD"],
            "csp3": ["FCsp3"],
            "num_ar": ["Num_AR"],
            "num_rb": ["Num_RB"],
            "num_sp_sp2_sp3": ["Num_sp", "Num_sp2", "Num_sp3"],
            "num_ha": ["Num_HA"],
            "num_sc": ["Num_SC"],
        }
        ordered_cols = ["SMILES", "METHOD", "BASIS"]
        for props in property_config.values():
            ordered_cols.extend(props)

        process_list = self.smiles_list if smiles_list is None else smiles_list
        total_time_start = time.time()

        own_tq = tq is None
        if own_tq:
            tq = tqdm(process_list, desc="Property", total=len(process_list),
                      mininterval=0.8, dynamic_ncols=True, colour="blue",
                      disable=disable_tqdm)

        for item in process_list:
            key = (item, self.method, self.basis_set)

            with self.prop_lock:
                if self._prop_existing_keys is None:
                    keys = set()
                    csv_path = f"{output_file}.csv"
                    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
                        try:
                            data_csv = pd.read_csv(
                                csv_path,
                                usecols=lambda c: c in {"SMILES", "METHOD", "BASIS"},
                            )
                            if {"SMILES", "METHOD", "BASIS"}.issubset(data_csv.columns):
                                keys = set(zip(data_csv["SMILES"],
                                               data_csv["METHOD"],
                                               data_csv["BASIS"]))
                            else:
                                keys = {(s, self.method, self.basis_set)
                                        for s in data_csv.iloc[:, 0].tolist()}
                        except Exception:
                            keys = set()
                    self._prop_existing_keys = keys
                existing_keys = self._prop_existing_keys

            if key in existing_keys:
                self._log(f"[Prop] Skip   {item} — already in mol_prop.csv", thread_id)
                if tq is not None:
                    tq.update(1)
                continue

            self._log(f"[Prop] Start  {item}", thread_id)

            mol = Chem.MolFromSmiles(item)
            if mol is None:
                self._log(f"[Prop] Invalid SMILES {item}", thread_id)
                if tq is not None:
                    tq.update(1)
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)
            AllChem.MMFFOptimizeMolecule(mol)
            row = {"SMILES": item, "METHOD": self.method, "BASIS": self.basis_set}
            file_name = smiles_to_folder(item, self.method, self.basis_set)
            file_path = os.path.join(input_path, file_name)

            if tq is not None:
                tq.set_postfix(Mol=item, Info="HOMO & LUMO")
            homo_value, lumo_value = get_homo_lumo(file_path, file_name)
            row["HOMO"] = _safe_get(homo_value, 2)
            row["LUMO"] = _safe_get(lumo_value, 2)

            if tq is not None:
                tq.set_postfix(Mol=item, Info="Dipole")
            row["Dipole"] = _safe_get(get_dipole(file_path, file_name), 1)

            if tq is not None:
                tq.set_postfix(Mol=item, Info="ESP & MPI")
            esp_value, mpi_value = get_esp_mpi(file_path, file_name)
            row["ESP_max"] = _safe_get(esp_value, 1)
            row["ESP_min"] = _safe_get(esp_value, 0)
            row["MPI"] = _safe_get(mpi_value, 1)

            if tq is not None:
                tq.set_postfix(Mol=item, Info="Vol & Rho")
            vol_value, rho_value = get_vol_rho(file_path, file_name)
            row["Vol"] = _safe_get(vol_value, 0)
            row["Rho"] = _safe_get(rho_value, 0)

            if tq is not None:
                tq.set_postfix(Mol=item, Info="2D descriptors")

            row["MW"] = Descriptors.MolWt(mol)
            row["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
            row["LogP"] = Crippen.MolLogP(mol)
            row["HBA"] = Lipinski.NumHAcceptors(mol)
            row["HBD"] = Lipinski.NumHDonors(mol)
            row["FCsp3"] = rdMolDescriptors.CalcFractionCSP3(mol)
            row["Num_AR"] = rdMolDescriptors.CalcNumAromaticRings(mol)
            row["Num_RB"] = Lipinski.NumRotatableBonds(mol)

            sp = sp2 = sp3 = 0
            ha = 0
            for atom in mol.GetAtoms():
                hyb = atom.GetHybridization()
                if hyb == rdchem.HybridizationType.SP:
                    sp += 1
                elif hyb == rdchem.HybridizationType.SP2:
                    sp2 += 1
                elif hyb == rdchem.HybridizationType.SP3:
                    sp3 += 1
                if atom.GetAtomicNum() not in (1, 6):
                    ha += 1
            row["Num_sp"], row["Num_sp2"], row["Num_sp3"] = sp, sp2, sp3
            row["Num_HA"] = ha

            Chem.AssignStereochemistry(mol)
            row["Num_SC"] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

            row_df = pd.DataFrame([row]).reindex(columns=ordered_cols)

            with self.prop_lock:
                write_header = (
                    (not os.path.isfile(f"{output_file}.csv"))
                    or os.path.getsize(f"{output_file}.csv") == 0
                )
                row_df.to_csv(f"{output_file}.csv", mode="a",
                              header=write_header, index=False)
                if self._prop_existing_keys is not None:
                    self._prop_existing_keys.add(key)

            self._log(f"[Prop] Done   {item}", thread_id)
            if tq is not None:
                tq.update(1)

        if tq is not None:
            try:
                tq.set_postfix(Total_Time=f"{(time.time() - total_time_start) / 3600:.3f} h")
            except Exception:
                pass
        if own_tq:
            tq.close()
        try:
            MolCalc.prop_df = pd.read_csv(f"{output_file}.csv")
        except Exception:
            pass
        return MolCalc.prop_df

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    @classmethod
    def transform(cls, experiment_condition, smiles=None, mol_prop=None):
        """Assemble the ML/DL design matrices for the given experiment rows."""
        exp_cond = experiment_condition
        prop_df = cls.prop_df if mol_prop is None else mol_prop
        prop_df["smiles_index"] = prop_df["SMILES"]
        prop_df = prop_df.set_index("smiles_index")

        prop_sam1 = pd.DataFrame(columns=[c + "_sam1" for c in prop_df.columns])
        prop_sam2 = pd.DataFrame(columns=[c + "_sam2" for c in prop_df.columns])
        for idx in range(exp_cond.shape[0]):
            s1 = str(exp_cond.iloc[idx, 0]).strip()
            s2 = str(exp_cond.iloc[idx, 1]).strip()
            a = prop_df.loc[[s1]].copy()
            b = prop_df.loc[[s2]].copy()
            a.columns = prop_sam1.columns
            b.columns = prop_sam2.columns
            prop_sam1 = pd.concat([prop_sam1, a], axis=0, ignore_index=True)
            prop_sam2 = pd.concat([prop_sam2, b], axis=0, ignore_index=True)
        prop_sam1 = prop_sam1.drop(columns=["SMILES_sam1"])
        prop_sam2 = prop_sam2.drop(columns=["SMILES_sam2"])
        prop_sam1_sam2 = pd.concat([prop_sam1, prop_sam2], axis=1)

        smiles_str = cls.smiles if smiles is None else smiles
        smiles_str.dropna(axis=0, how="all", inplace=True)
        smiles_str.dropna(axis=1, how="all", inplace=True)
        smiles_str = smiles_str.copy()
        smiles_str["SMILES"] = smiles_str["SMILES"].astype(str).str.strip()

        sym = set(smiles_str.loc[smiles_str["Symmetric"] == 1, "SMILES"])
        asym = set(smiles_str.loc[smiles_str["Symmetric"] == 0, "SMILES"])
        het = set(smiles_str.loc[smiles_str["HeteroEndFrag"] == 1, "SMILES"])
        chains = {
            n: set(smiles_str.loc[smiles_str["Chain_length"] == n, "SMILES"])
            for n in range(1, 6)
        }

        dir_str = {"HeteroEndFrag": [], "HeteroSym": [],
                   "BthAsymStr": [], "UnifChainLen": []}
        for idx in range(exp_cond.shape[0]):
            s1 = str(exp_cond.iloc[idx, 0]).strip()
            s2 = str(exp_cond.iloc[idx, 1]).strip()
            dir_str["HeteroEndFrag"].append(1 if (s1 in het or s2 in het) else 0)
            dir_str["HeteroSym"].append(
                1 if ((s1 in sym and s2 in asym) or (s1 in asym and s2 in sym)) else 0
            )
            dir_str["BthAsymStr"].append(1 if (s1 in asym and s2 in asym) else 0)
            dir_str["UnifChainLen"].append(
                1 if any(s1 in c and s2 in c for c in chains.values()) else 0
            )

        divided = pd.concat([exp_cond, pd.DataFrame(dir_str), prop_sam1_sam2], axis=1)
        divided = divided.dropna(axis=0, how="any")
        divided = divided.reindex(
            columns=[c for c in divided.columns if c not in {"Voc", "FF", "Jsc"}]
        ).dropna(axis=0, how="any")

        r = divided["SAMs_ratio"]

        def blend(src):
            return r * divided[f"{src}_sam1"] + (1 - r) * divided[f"{src}_sam2"]

        ml = pd.DataFrame(index=divided.index)
        for col in ("Bandgap", "Mixture", "HeteroEndFrag", "HeteroSym",
                    "BthAsymStr", "UnifChainLen", "MACl_additive"):
            ml[col] = divided[col]

        for src in ("HOMO", "LUMO", "Dipole", "ESP_max", "ESP_min",
                    "MPI", "Vol", "Rho", "MW", "LogP", "TPSA", "FCsp3"):
            ml[f"{src}_avg"] = blend(src)

        for src in ("HOMO", "LUMO", "LogP", "TPSA"):
            ml[f"{src}_diff"] = (divided[f"{src}_sam1"]
                                 - divided[f"{src}_sam2"]).abs()

        ml["HOMO_LUMO_gap"] = (ml["HOMO_avg"] - ml["LUMO_avg"]).abs()
        ml["ESP_range"] = (ml["ESP_max_avg"] - ml["ESP_min_avg"]).abs()
        ml["Vol_ratio"] = divided["Vol_sam1"] / divided["Vol_sam2"]

        for src, name in (
            ("HBA", "HBA_total"), ("HBD", "HBD_total"),
            ("Num_AR", "AR_total"), ("Num_RB", "RB_total"),
            ("Num_sp", "sp_total"), ("Num_sp2", "sp2_total"),
            ("Num_sp3", "sp3_total"), ("Num_HA", "HA_total"),
            ("Num_SC", "SC_total"),
        ):
            ml[name] = divided[f"{src}_sam1"] + divided[f"{src}_sam2"]

        if "PCE" in divided.columns:
            ml["PCE"] = divided["PCE"]

        # Canonical column order consumed by the training pipeline.
        ordered = [
            "Bandgap", "Mixture", "HeteroEndFrag", "HeteroSym", "BthAsymStr",
            "UnifChainLen", "MACl_additive", "HOMO_avg", "LUMO_avg",
            "HOMO_LUMO_gap", "HOMO_diff", "LUMO_diff", "Dipole_avg",
            "ESP_max_avg", "ESP_min_avg", "ESP_range", "MPI_avg", "Vol_avg",
            "Vol_ratio", "Rho_avg", "MW_avg", "LogP_avg", "LogP_diff",
            "TPSA_avg", "TPSA_diff", "HBA_total", "HBD_total", "FCsp3_avg",
            "AR_total", "RB_total", "sp_total", "sp2_total", "sp3_total",
            "HA_total", "SC_total", "PCE",
        ]
        ml = ml[[c for c in ordered if c in ml.columns]]

        # Drop all-NaN descriptors; always keep the target column.
        keep = [c for c in ml.columns if c == "PCE" or ml[c].notna().any()]
        ml = ml[keep]

        # Klekota-Roth fingerprint features (order-preserving union of both
        # SAMs).  'n_jobs=1' prevents background parallelism.
        smiles_list = (
            cls.smiles["SMILES"].to_list() if smiles is None
            else smiles["SMILES"].to_list()
        )
        krfps = KlekotaRothFingerprint(n_jobs=1).transform(smiles_list)
        idx_of = {s: i for i, s in enumerate(smiles_list)}
        dl_rows = []
        for idx in range(exp_cond.shape[0]):
            s1 = str(exp_cond.iloc[idx, 0]).strip()
            s2 = str(exp_cond.iloc[idx, 1]).strip()
            fused = krfps[idx_of[s1]] + krfps[idx_of[s2]]
            dl_rows.append([1 if v > 1 else v for v in fused])
        dl = pd.DataFrame(dl_rows, columns=[f"krfp_{i}" for i in range(4860)])
        dl = pd.concat([ml.reset_index(drop=True), dl], axis=1)
        cols = [c for c in dl.columns if c != "PCE"]
        if "PCE" in dl.columns:
            cols = cols + ["PCE"]
        return ml, dl.reindex(columns=cols)