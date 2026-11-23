"""MSEC: VLM perception, hierarchy check, and CoT reasoning.

Drives the multi-modal explanation pipeline behind the GUI's *Agent* page:

    1. 'vlm_perception'    - describe SHAP dependence curves + substructure diagrams.
    2. 'hierarchy_checker' - verify ids and chemical classes against the SMARTS graph.
    3. 'cot_reasoner'      - chain-of-thought synthesis into a Markdown report plus
                             a structured JSON block.

Input:
    shap_dependency_img_path, substructure_img_path  -- images fed to the VLM.
    df_shap_descriptor_vec                           -- per-descriptor SHAP values.
    acin_substructure_weight                         -- ACIN weights.
    q, stop_event                                    -- GUI queue + cancellation flag.
    api_key, base_url, model, temperature            -- OpenAI-compatible endpoint.
    max_tokens (<=0 = no cap), descriptor_top_n,
    thinking                                         -- endpoint + tuning knobs.

Output:
    run_pipeline() -> {"status", "verification", "vlm_raw_observation",
                       "valid_evidence_after_hierarchy_checker",
                       "analysis_result"}.

Weights rank and label; images describe; the model groups and selects on evidence;
the rule engine verifies only decidable facts."""


from __future__ import annotations

import base64
import threading
from collections import deque
import json
import os
import re
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import pandas as pd
from openai import OpenAI
from rdkit import Chem

# ---------------------------------------------------------------------------
# Protocol constants shared with the GUI.
# ---------------------------------------------------------------------------
MSG_THINK = "Think"
MSG_REPLY = "Reply"
MSG_PROGRESS = "_PROGR"
MSG_INIT = "_Init"
MSG_END = "_End"
MSG_ERROR = "_Error_"

SEP_JSON = "---JSON---"
CONTRIB_POSITIVE = "positive"      # sign of the ACIN gating weight
CONTRIB_NEGATIVE = "negative"
DIVIDER = "-" * 28

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "acin_weights.csv"
DEFAULT_SHAP_CSV = ROOT / "plots" / "shap_values" / "CatBoostRegressor_shap_values.csv"
DEFAULT_DEPS = ROOT / "plots" / "deps" / "gam" / "CatBoostRegressor"
DEFAULT_PERCEPTION_TIMEOUT = 240      # seconds for one perception call
DEFAULT_REASONING_TIMEOUT = 600       # seconds for one reasoning pass
DEFAULT_TIMEOUT = 600                 # client default (max of the two)
DEFAULT_MAX_RETRIES = 0               # no SDK retries; we split and retry ourselves

# ---------------------------------------------------------------------------
# JSON and image helpers
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> str:
    """Return the first balanced {...} block, tolerating prose around it."""
    start = text.find("{")
    if start < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def clean_llm_json(text: str) -> str:
    """Strip code fences and trailing commas before parsing."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    text = extract_json_object(text) or text
    return re.sub(r",\s*([}\]])", r"\1", text)


def encode_image_to_base64(path: str, max_side: int = 0, quality: int = 80) -> str:
    """Base64 image, optionally downscaled and JPEG-compressed."""
    if not max_side:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            if max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize((max(1, int(image.width * scale)),
                                      max(1, int(image.height * scale))),
                                     Image.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")


def collect_images(path: str) -> List[str]:
    """Enumerate image files under a file or directory path."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if not path:
        return []
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.lower().endswith(exts)]
    if os.path.isfile(path) and path.lower().endswith(exts):
        return [path]
    return []


def plan_batches(structures: Sequence[str], curves: Sequence[str],
                 max_bytes: int, max_items: int, structure_batch: int,
                 size_of=None) -> List[Dict[str, List[str]]]:
    """Group structures into requests and interleave the SHAP curves.

    Every request carries a share of the curves (round-robin over the ranked
    list), so no batch is asked to report trends it cannot see and a failed
    batch only loses a spread of curves rather than the whole top of the list.
    """
    size_of = size_of or os.path.getsize
    chunks: List[List[str]] = []
    current: List[str] = []
    size = 0
    for path in structures:
        payload = size_of(path)
        full = (len(current) >= structure_batch
                or len(current) + 1 > max_items
                or (current and size + payload > max_bytes))
        if full:
            chunks.append(current)
            current, size = [], 0
        current.append(path)
        size += payload
    if current or not chunks:
        chunks.append(current)

    batches = [{"curves": [], "structures": chunk} for chunk in chunks]
    for position, curve in enumerate(curves):
        batches[position % len(batches)]["curves"].append(curve)
    return batches


# ---------------------------------------------------------------------------
# Broad chemical-class vocabulary (not tailored to any single molecule pair)
# ---------------------------------------------------------------------------

CLASS_DEFINITIONS = """- alkane_chain: three or more aliphatic carbons, no ring
- cycloalkyl: a non-aromatic carbon ring
- alkene: non-aromatic C=C
- alkyne: C#C
- arene: a benzene-type aromatic carbocycle
- fused_arene: two or more fused aromatic carbocycles (naphthalene)
- n_heteroarene: aromatic ring containing nitrogen
- o_heteroarene: aromatic ring containing oxygen
- s_heteroarene: aromatic ring containing sulfur (thiophene)
- azole: five-membered aromatic ring with two or more heteroatoms
- azine: six-membered aromatic ring containing nitrogen
- fused_n_heteroarene: fused aromatic system containing nitrogen (indole, carbazole, quinoline)
- amine: nitrogen with one to three single bonds, not aromatic, not an amide
- aniline: nitrogen single-bonded to an aromatic carbon
- amide: carbonyl carbon bonded to nitrogen
- imine: non-aromatic C=N
- nitrile: C#N
- nitro: N(=O)2 or [N+](=O)[O-]
- azo: N=N
- azide: N=N=N
- hydrazine: N-N single bond
- oxime: C=N-O
- carbamate: O-C(=O)-N
- urea: N-C(=O)-N
- alcohol: O-H not part of a carboxylic or phosphonic acid
- ether: oxygen bonded to two carbons, not part of a carbonyl
- alkoxyarene: ether oxygen bonded to an aromatic and an aliphatic carbon
- ester: C(=O)-O-C
- carboxylic_acid: C(=O)-OH with a free hydroxyl
- ketone: carbonyl carbon bonded to two carbons
- aldehyde: carbonyl carbon bonded to hydrogen
- anhydride: C(=O)-O-C(=O)
- peroxide: O-O
- thiol: S-H or a terminal sulfur
- thioether: sulfur bonded to carbon (C-S-C, including aryl-S)
- disulfide: S-S
- sulfoxide: S=O with two carbon neighbours
- sulfone: S(=O)2
- sulfonic_acid: S(=O)2-OH
- phosphine: phosphorus bonded to three carbons, no oxygen
- phosphine_oxide: P=O with three carbon substituents
- phosphonic_acid: P with two or more oxygens and no P-O-C bond
- phosphonate_ester: P with a P-O-C bond and no P-OH
- phosphate: P with four oxygens
- halide: fluorine, chlorine, bromine or iodine
- trifluoromethyl: carbon bonded to three fluorines
- silane: silicon present
- boronic: boron present
- quaternary_carbon: carbon with four heavy-atom neighbours
- generic_bond: a one- or two-atom pattern with no chemical content
- other: none of the above"""

NON_DESIGN_CLASSES = {"generic_bond", "other"}
# Umbrella claims are easy to satisfy; they pass but are counted separately so a
# high verification rate cannot be obtained by always answering "arene".
UMBRELLA_CLASSES = {"arene", "halide", "amine"}

# One declarative rule per class. Keys:
#   smarts: str | [str]   any pattern that must match
#   not:    [str]         patterns that must NOT match
#   only:   {symbols}     elements allowed (wildcards ignored)
#   min:    {symbol: n}   minimum element counts
#   rings:  (lo, hi)      ring-count range
#   aromatic: (lo, hi)    aromatic-atom range
#   hetero_min: n         aromatic N/O/S atoms at least n
#   atoms_max: n          pattern size at most n
CLASS_RULES: Dict[str, Dict[str, Any]] = {
    # hydrocarbons
    "alkane_chain": {"only": {"C"}, "rings": (0, 0), "aromatic": (0, 0), "min": {"C": 2}},
    "cycloalkyl": {"only": {"C"}, "rings": (1, None), "aromatic": (0, 0)},
    "alkene": {"smarts": "[CX3]=[CX3]"},
    "alkyne": {"smarts": "C#C"},
    "arene": {"smarts": "[c]"},
    "fused_arene": {"only": {"C"}, "rings": (2, None), "aromatic": (10, None)},
    # heteroaromatics
    "n_heteroarene": {"smarts": "[n]"},
    "o_heteroarene": {"smarts": "[o]"},
    "s_heteroarene": {"smarts": "[s]"},
    "azole": {"smarts": "[a;r5]", "hetero_min": 2},
    "azine": {"smarts": "[n;r6]"},
    "fused_n_heteroarene": {"smarts": "[n]", "rings": (2, None), "aromatic": (8, None)},
    # nitrogen functions
    "amine": {"smarts": "[N;X1,X2,X3;!a]", "not": ["[N]-[C]=O", "[N]-S"]},
    "aniline": {"smarts": "[N;!a]-[c]"},
    "amide": {"smarts": "[C](=O)-[N]"},
    "imine": {"smarts": "[C]=[N]"},
    "nitrile": {"smarts": "C#N"},
    "nitro": {"smarts": ["[N+](=O)[O-]", "[N](=O)=O"]},
    "azo": {"smarts": "[N]=[N]"},
    "azide": {"smarts": ["[N]=[N]=[N]", "[N-]=[N+]=[N]"]},
    "hydrazine": {"smarts": "[N]-[N]"},
    "oxime": {"smarts": "[C]=[N]-[O]"},
    "carbamate": {"smarts": "[O]-[C](=O)-[N]"},
    "urea": {"smarts": "[N]-[C](=O)-[N]"},
    # oxygen functions
    "alcohol": {"smarts": "[O;X1]", "not": ["[O]-[C]=O", "[O]-[P]"]},
    "ether": {"smarts": "[O;X2](-[#6])-[#6]"},
    "alkoxyarene": {"smarts": "[c]-[O;X2]-[#6]"},
    "ester": {"smarts": "[C](=O)-[O;X2]-[#6]"},
    "carboxylic_acid": {"smarts": "[C](=O)-[O;X1]"},
    "ketone": {"smarts": "[#6]-[C;X3](=O)-[#6]"},
    "aldehyde": {"smarts": "[C;H1]=[O]"},
    "anhydride": {"smarts": "[C](=O)-[O]-[C](=O)"},
    "peroxide": {"smarts": "[O]-[O]"},
    # sulfur functions
    "thiol": {"smarts": "[S;H1,X1]", "not": ["[S]-[S]"]},
    "thioether": {"smarts": ["[S;X2]", "[S;X1]-[#6]"], "aromatic": (0, None)},
    "disulfide": {"smarts": "[S]-[S]"},
    "sulfoxide": {"smarts": "[S;X3](=O)(-[#6])-[#6]"},
    "sulfone": {"smarts": "[S](=O)(=O)"},
    "sulfonic_acid": {"smarts": "[S](=O)(=O)-[O;X1]"},
    # phosphorus functions
    "phosphine": {"smarts": "[P;X3](-[#6])-[#6]", "not": ["[P]=[O]"]},
    "phosphine_oxide": {"smarts": "[P](=O)(-[#6])-[#6]"},
    "phosphonic_acid": {"smarts": "[P]", "min": {"O": 2}, "not": ["[P]-[O]-[#6]"]},
    "phosphonate_ester": {"smarts": "[P]-[O]-[#6]"},
    "phosphate": {"smarts": "[P]", "min": {"O": 4}},
    # halogens and other heteroatoms
    "halide": {"smarts": "[F,Cl,Br,I]"},
    "trifluoromethyl": {"smarts": "[C](-[F])(-[F])-[F]"},
    "silane": {"smarts": "[Si]"},
    "boronic": {"smarts": "[B]"},
    # generic
    "quaternary_carbon": {"smarts": "[C;X4]", "rings": (0, 0), "aromatic": (0, 0)},
    "generic_bond": {"atoms_max": 2},
}

_COMPILED: Dict[str, Chem.Mol] = {}


def _smarts(pattern: str) -> Chem.Mol:
    """Compile and cache one SMARTS string."""
    if pattern not in _COMPILED:
        mol = Chem.MolFromSmarts(pattern)
        if mol is None:
            raise ValueError(f"invalid SMARTS: {pattern}")
        _COMPILED[pattern] = mol
    return _COMPILED[pattern]


def _features(query: Chem.Mol) -> Tuple[Dict[str, int], int, int]:
    """Element counts, aromatic atom count and ring count of a SMARTS pattern."""
    Chem.FastFindRings(query)
    elements: Dict[str, int] = {}
    for atom in query.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != "*":                      # '[!#1]' wildcards are not elements
            elements[symbol] = elements.get(symbol, 0) + 1
    aromatic = sum(1 for atom in query.GetAtoms() if atom.GetIsAromatic())
    return elements, aromatic, query.GetRingInfo().NumRings()


def _rule_ok(rule: Dict[str, Any], query: Chem.Mol, elements: Dict[str, int],
             aromatic: int, rings: int, atoms: int, hetero: int) -> bool:
    """Evaluate one declarative class rule against a pattern."""
    if "only" in rule and not set(elements) <= rule["only"]:
        return False
    for symbol, count in rule.get("min", {}).items():
        if elements.get(symbol, 0) < count:
            return False
    for key, value in (("rings", rings), ("aromatic", aromatic)):
        if key in rule:
            low, high = rule[key]
            if value < low or (high is not None and value > high):
                return False
    if "hetero_min" in rule and hetero < rule["hetero_min"]:
        return False
    if "atoms_max" in rule and atoms > rule["atoms_max"]:
        return False
    for pattern in rule.get("not", []):
        if query.HasSubstructMatch(_smarts(pattern)):
            return False
    smarts = rule.get("smarts")
    if smarts is None:
        return True
    if isinstance(smarts, str):
        smarts = [smarts]
    return any(query.HasSubstructMatch(_smarts(p)) for p in smarts)


def classify(query: Chem.Mol) -> set:
    """Return every class consistent with one KRFP SMARTS pattern."""
    query.UpdatePropertyCache(strict=False)
    elements, aromatic, rings = _features(query)
    atoms = query.GetNumAtoms()
    hetero = sum(1 for atom in query.GetAtoms()
                 if atom.GetIsAromatic() and atom.GetSymbol() in ("N", "O", "S"))
    return {name for name, rule in CLASS_RULES.items()
            if _rule_ok(rule, query, elements, aromatic, rings, atoms, hetero)}


# Declaring a narrower class is acceptable when the pattern also satisfies it;
# umbrella entries let a broader label cover a consistent narrower one.
CLAIM_COMPATIBLE = {name: (name,) for name in CLASS_RULES}
CLAIM_COMPATIBLE.update({
    "arene": ("arene", "fused_arene", "n_heteroarene", "o_heteroarene",
              "s_heteroarene", "azole", "azine", "fused_n_heteroarene"),
    "n_heteroarene": ("n_heteroarene", "azole", "azine", "fused_n_heteroarene"),
    "amine": ("amine", "aniline"),
    "ether": ("ether", "alkoxyarene"),
    "thioether": ("thioether", "s_heteroarene"),
    "halide": ("halide", "trifluoromethyl"),
})


def class_supported(pattern: Chem.Mol, claim: str) -> bool:
    """True when the declared class is consistent with the SMARTS graph.

    'other' is accepted without verification and counted separately, so a model
    cannot inflate its verified rate by abstaining.
    """
    if claim == "other":
        return True
    return bool(classify(pattern) & set(CLAIM_COMPATIBLE.get(claim, ())))


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class MSECAgent:
    """Three-stage MSEC with a verifiable evidence chain."""

    vlm_prompt = """Task: pure visual observation and chemical description. DO NOT infer mechanisms.

You are looking at two types of images:
1. SHAP dependence plots for physicochemical descriptors.
2. Molecular substructure diagrams labelled "KRFP <n>".

The ACIN weight table is given below. It already decides which substructures matter;
you must NOT judge or restate their importance. Your job is to describe each
substructure and, separately, to report the shape of each SHAP curve.

For every substructure image:
- Convert the label "KRFP 123" to "krfp_123".
- Assign one class from the vocabulary, using the drawn structure:
{vocabulary}
- Add "desc": at most 12 words describing the structural features (rings, heteroatoms,
  functional groups). Do not use causal or importance language.
- If no class fits, use "other" and say what you see in "desc".

For the SHAP dependence plots:
- Report the visible trend: monotonic increasing | monotonic decreasing | peak |
  valley | threshold | complex.
- Report an optimal range only if it is visible on the axis.
- Report where the SHAP value is positive.
- Do NOT explain why.

Output JSON only:
{{"substructure_observations":[{{"substructure_idx":"krfp_123",
   "claim":"arene","desc":"six-membered aromatic ring"}}],
 "shap_dependence_observations":[{{"feature":"MW_avg","trend":"monotonic",
   "optimal_range":"0.4 to 0.6","sign_region":"positive above 0.4"}}]}}
""".strip()

    cot_prompt = """You are a data-driven materials scientist working on Co-SAM buried interfaces.

You are given:
1. valid_evidence (substructure-level, verified; each item has claim, desc and acin_weight):
{valid_evidence}

2. descriptor statistics (CatBoost SHAP global attribution):
{shap_descriptor_text}

3. descriptor_dictionary (physicochemical definitions, units, calculation logic):
{descriptor_dictionary}

4. shap_dependence_observations (curve shapes and axis values read from the plots):
{shap_dependence_observations}

5. verification_summary (what was checked and what failed):
{verification}

Follow this reasoning procedure strictly:

Step 1: Descriptor-level mechanism extraction
- For each top descriptor in shap_descriptor_text, interpret the sign of mean, the
  pos_ratio and the magnitude.
- Every optimal range must be taken from shap_dependence_observations: quote the axis
  numbers you read on the curve. If a descriptor has no observation, write "not read"
  rather than supplying a range.
- Summarize 3-5 descriptor-level mechanisms, each citing a descriptor and its statistics.
- Bandgap is a device parameter, not a molecular handle: mention it at most as a
  screening condition, never as a design rule.

Step 2: Substructure-level mechanism extraction
- Group valid_evidence by the chemical class and the SMARTS pattern. Report for each
  group: the krfp indices, their acin_weight range, and the sign of that weight.
- Patterns from the same chemical family may overlap; keep both only if each
  contributes a distinct chemical role, and say which.
- Do NOT invent fragments that are not in valid_evidence.

Step 3: Synergy hypothesis
- Propose 4 synergistic mechanisms for the Co-SAM buried interface.
- Each mechanism connects a descriptor window, one or more fragment groups, and a
  plausible physical/chemical role (polarity, energy alignment, packing, passivation).
- Use cautious language: "evidence suggests", "may indicate", "consistent with".
- Do NOT mention any specific molecule name unless it is given in the input.

Step 4: Evidence-based design recommendations
- CRITICAL: do NOT simply rank and select substructures by their acin_weight. A high
  weight alone does not make a good recommendation. Select on the evidence gathered
  in Steps 1-3.
- Select up to {max_recommendations} fragments whose ACIN weight is positive and that best align with
  the descriptor windows and the synergistic mechanisms. Fewer is acceptable, and
  "insufficient evidence" is an acceptable answer. Redundant patterns may be included
  when the evidence supports different roles, but say why.
- Do NOT invent or use any krfp outside valid_evidence.
- Each recommended item must include "substructure_idx", "acin_weight", "contrib" and
  a "reason" grounded in evidence. Do NOT use "highest ACIN weight" as a reason.
- Group the selected substructures into 2-4 favorable combinations; the union of their
  krfp indices must equal the recommended set. Add avoided combinations where the
  evidence is antagonistic or missing.

First, output a Markdown report with exactly the following sections:

Never put a vertical bar inside a table cell: the reader splits every row on "|", so a
bar inside a cell shifts the whole table. Use ";" or "/" instead. Keep each table row on
a single physical line, with exactly as many cells as the header.

# MSEC Analysis Report

## 1. Descriptor-Level Mechanisms
| Feature | Evidence | Direction | Optimal Range |
|---------|----------|-----------|---------------|
| ...     | ...      | ...       | ...           |

## 2. Substructure-Level Mechanisms
| Fragment Group | KRFP Indices | ACIN Weight Range | Contribution | Role Hypothesis |
|----------------|--------------|-------------------|--------------|-----------------|
| ...            | ...          | ...               | ...          | ...             |

## 3. Synergistic Mechanisms
Do not use Bandgap as a descriptor in any mechanism below.

1. **Mechanism**: ...
   - Descriptor evidence: ...
   - Substructure evidence: ...
   - Confidence: high/medium/low

## 4. Design Recommendations
### Recommended Substructures (Evidence-Based Selection)
| Substructure | ACIN Weight | Reason |
|--------------|-------------|--------|
| ...          | ...         | ...    |

### Favorable Combinations
- **Combination**: ...
  - KRFP indices: ...
  - Reason: ...
  - Supporting evidence: ...

### Avoided Combinations
- **Combination**: ...
  - Reason: ...
  - Supporting evidence: ...

Then output a line with exactly:
---JSON---

Then output the structured JSON object exactly as specified below:
{{
  "descriptor_mechanisms": [...],
  "substructure_mechanisms": [...],
  "synergistic_mechanisms": [...],
  "recommended_substructures": [...],
  "favorable_substructure_combinations": [...],
  "avoided_substructure_combinations": [...]
}}

Do not output any other text before the Markdown report, between the Markdown report
and ---JSON---, or after the JSON object.
CRITICAL: You MUST output the line "---JSON---" exactly, on its own line.
""".strip()

    DESCRIPTOR_DICTIONARY = {
        "Bandgap": "Experimentally measured bandgap value (eV). Note: device-level parameter; exclude from molecular design mechanisms.",
        "Mixture": "Binary indicator for a binary mixed system (0 = single component, 1 = two-component mixture).",
        "HeteroEndFrag": "Binary indicator for asymmetric terminal fragments (0 = no, 1 = yes).",
        "HeteroSym": "Binary indicator for hetero-symmetry (0 = no, 1 = one symmetric and one asymmetric molecule).",
        "BthAsymStr": "Binary indicator for whether both molecules in the pair are asymmetric (0 = no, 1 = yes).",
        "UnifChainLen": "Binary indicator for uniform chain lengths between the two molecules (0 = no, 1 = yes).",
        "MACl_additive": "Binary indicator for the presence of the MACl additive (0 = no, 1 = yes).",
        "HOMO_avg": "Molar-ratio weighted average of the HOMO energies (eV).",
        "LUMO_avg": "Molar-ratio weighted average of the LUMO energies (eV).",
        "HOMO_LUMO_gap": "Absolute energy difference between HOMO_avg and LUMO_avg (eV).",
        "HOMO_diff": "Absolute HOMO difference between the two molecules (eV), reflecting the energy-level gradient.",
        "LUMO_diff": "Absolute LUMO difference between the two molecules (eV).",
        "Dipole_avg": "Molar-ratio weighted average of the molecular dipole moments (Debye).",
        "ESP_max_avg": "Molar-ratio weighted average of the maximum electrostatic potential (a.u.).",
        "ESP_min_avg": "Molar-ratio weighted average of the minimum electrostatic potential (a.u.).",
        "ESP_range": "Range between maximum and minimum electrostatic potentials (a.u.), reflecting polarizability.",
        "MPI_avg": "Molar-ratio weighted average of the Molecular Polarity Index (kcal/mol).",
        "Vol_avg": "Molar-ratio weighted average of molecular volumes (A^3).",
        "Vol_ratio": "Volume ratio of the two molecules (Vol_sam1 / Vol_sam2), reflecting spatial complementarity.",
        "Rho_avg": "Molar-ratio weighted average of molecular densities (g/cm^3).",
        "MW_avg": "Molar-ratio weighted average of molecular weights (g/mol).",
        "LogP_avg": "Molar-ratio weighted average of the partition coefficients, reflecting hydrophobicity.",
        "LogP_diff": "Absolute LogP difference between the two molecules.",
        "TPSA_avg": "Molar-ratio weighted average of the Topological Polar Surface Area (A^2), reflecting polarity.",
        "TPSA_diff": "Absolute TPSA difference between the two molecules.",
        "HBA_total": "Total hydrogen-bond acceptors across both molecules (SAM1 + SAM2), reflecting passivation capability.",
        "HBD_total": "Total hydrogen-bond donors across both molecules (SAM1 + SAM2).",
        "FCsp3_avg": "Molar-ratio weighted average of the sp3 carbon fraction, reflecting flexibility.",
        "AR_total": "Total aromatic rings across both molecules (SAM1 + SAM2).",
        "RB_total": "Total rotatable bonds across both molecules (SAM1 + SAM2), reflecting conformational flexibility.",
        "sp_total": "Total sp-hybridized atoms across both molecules (SAM1 + SAM2).",
        "sp2_total": "Total sp2-hybridized atoms across both molecules (SAM1 + SAM2), affecting pi-pi stacking.",
        "sp3_total": "Total sp3-hybridized atoms across both molecules (SAM1 + SAM2).",
        "HA_total": "Total heavy (non-hydrogen) atoms across both molecules (SAM1 + SAM2).",
        "SC_total": "Total stereocentres across both molecules (SAM1 + SAM2).",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int = 0,
        stop_event=None,
        q=None,
        descriptor_top_n: int = 20,
        thinking: bool = True,
    ) -> None:
        self.descriptor_top_n = descriptor_top_n
        self.q = q
        self.stop_event = stop_event
        self.thinking = thinking

        # Runtime knobs with fixed defaults; expose as attributes, not arguments.
        self.max_recommendations = 10
        self.pool_size = 20
        self.batch_items = 60
        self.structure_batch = 30
        self.concurrency = 0            # 0: adaptive, 1: serial, ≥2: Parallel
        self.max_concurrency = 3
        self.batch_bytes = 4_000_000
        self.perception_timeout = DEFAULT_PERCEPTION_TIMEOUT   # one perception call
        self.reasoning_timeout = DEFAULT_REASONING_TIMEOUT     # one report pass
        self.timeout = max(self.perception_timeout, self.reasoning_timeout)
        self.max_retries = DEFAULT_MAX_RETRIES

        self.shap_image_cap: Optional[int] = None   # None -> descriptor_top_n curves
        self.image_max_side = 600
        self.image_quality = 80
        self.text_only = False

        self._api_key = api_key
        self._base_url = base_url
        self.model_name = model
        self.temperature = temperature
        self.max_tokens: Optional[int] = (
            int(max_tokens) if max_tokens and int(max_tokens) > 0 else None)
        self.cot_history_msg: List[Dict[str, str]] = []
        self.make_client()

        # Populated by data_prep; kept on the instance so the public signatures
        # stay unchanged.
        self._patterns = None
        self._encoded: Dict[str, str] = {}
        self._payload_lock = threading.Lock()
        self._cohort: set = set()

    def make_client(self) -> None:
        """(Re)build the OpenAI client from the stored endpoint and timeout."""
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            http_client=httpx.Client(trust_env=False, timeout=self.timeout),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    # -- GUI plumbing ---------------------------------------------------
    def _emit(self, kind: str, text: str = "", *, only_if_thinking: bool = False) -> None:
        """Push one protocol message onto the GUI queue, tolerating 'None'."""
        if self.q is None:
            return
        if only_if_thinking and not self.thinking:
            return
        try:
            self.q.put(f"{kind}{text}")
        except Exception:
            pass

    def _stopped(self) -> bool:
        """True when the GUI asked the agent to stop."""
        return self.stop_event is not None and self.stop_event.is_set()

    @staticmethod
    def _wrap(text: str, limit: int = 96) -> str:
        """Break long lines at spaces so the GUI think box stays inside its frame."""
        lines: List[str] = []
        for line in text.splitlines() or [""]:
            while len(line) > limit:
                cut = line.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                lines.append(line[:cut].rstrip())
                line = line[cut:].lstrip()
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Human-readable duration."""
        if seconds < 1.0:
            return f"{seconds:.2f}s"
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        return f"{int(seconds // 60)}m {seconds % 60:.0f}s"

    def _stage(self, number: str, name: str) -> None:
        """Open one pipeline stage: banner plus a reset of the stage clock."""
        self._t_last = time.perf_counter()
        bar = "-" * max(6, 46 - len(number) - len(name))
        self._think(f"--- {number} {name} {bar}")

    def _done(self, extra: str = "") -> None:
        """Close the current stage and report how long it took."""
        elapsed = time.perf_counter() - (self._t_last or time.perf_counter())
        suffix = f"  ({extra})" if extra else ""
        self._think(f"  done {self._fmt_time(elapsed)}{suffix}")

    def _verify_table(self, rows: List[Dict[str, Any]], stats: Dict[str, Any]) -> None:
        """Print the audit table: what was accepted, what was rejected and why."""
        total = len(rows)
        accepted = [r for r in rows if r["ok"]]
        rejected = [r for r in rows if not r["ok"]]
        rate = len(accepted) / total if total else 0.0
        shown = 3
        self._think(f"  accepted  {len(accepted)}/{total}  ({rate:.0%})")
        for row in accepted[:shown]:
            self._think(f"    {row['id']:<10} {row['weight']:+.3f}  {row['claim']:<20} ok")
        if len(accepted) > shown:
            self._think(f"    ... {len(accepted) - shown} more accepted")
        self._think(f"  rejected  {len(rejected)}")
        for row in rejected[:5]:
            weight = f"{row['weight']:+.3f}" if row["weight"] is not None else "  --  "
            self._think(f"    {row['id']:<10} {weight}  {row['claim']:<20} {row['reason']}")
        if len(rejected) > 5:
            self._think(f"    ... {len(rejected) - 5} more rejected")
        self._think(f"  integrity  unknown id {stats['no_index']} · "
                    f"not shown {stats['not_staged']} · duplicate {stats['duplicate']}")
        self._done(f"{len(accepted)} kept, {len(rejected)} rejected")

    def _think(self, text: str) -> None:
        """Send one progress line to the GUI think box and to stdout."""
        payload = self._wrap(text).rstrip("\n") + "\n"
        try:
            print(text, flush=True)
        except UnicodeEncodeError:                 # non-UTF8 console
            print(text.encode("ascii", "ignore").decode(), flush=True)
        self._emit(MSG_THINK, payload, only_if_thinking=True)

    def _stream_reasoning(self, messages: List[Dict[str, str]]) -> str:
        """Stream one completion; text before '---JSON---' goes to the reply bubble."""
        stream = self.client.chat.completions.create(
            **self._request_kwargs(messages, stream=True,
                                   timeout=self.reasoning_timeout))

        parts: List[str] = []
        pending = ""
        json_started = False
        self._reply_emitted = False
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and not self._reply_emitted:
                self._emit(MSG_THINK, reasoning, only_if_thinking=True)
            text = getattr(delta, "content", None) or ""
            if not text:
                continue
            parts.append(text)
            if json_started:
                continue
            pending += text
            if SEP_JSON in pending:
                head, _, _ = pending.partition(SEP_JSON)
                if head:
                    self._emit(MSG_REPLY, head)
                    self._reply_emitted = True
                json_started = True
                pending = ""
            elif len(pending) > len(SEP_JSON):
                head, pending = pending[:-len(SEP_JSON)], pending[-len(SEP_JSON):]
                self._emit(MSG_REPLY, head)
                self._reply_emitted = True
        if not json_started and pending:
            self._emit(MSG_REPLY, pending)
            self._reply_emitted = True
        return "".join(parts)

    @staticmethod
    def send_error(q, exc) -> None:
        """Push an error message to the GUI queue."""
        if q is not None:
            q.put(f"{MSG_ERROR}Error: {exc}")

    # -- model calls ----------------------------------------------------
    def _request_kwargs(self, messages: List[Dict[str, Any]], *,
                        json_mode: bool = False, stream: bool = False,
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """Shared request kwargs for chat and vision calls."""
        kwargs: Dict[str, Any] = {"model": self.model_name,
                                  "temperature": self.temperature,
                                  "messages": messages,
                                  "timeout": timeout if timeout is not None
                                  else self.perception_timeout}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream"] = True
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def _chat(self, content: Any = None, json_mode: bool = True,
              messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Non-streaming completion, optionally constrained to a JSON object."""
        if messages is None:
            messages = [{"role": "user", "content": content}]
        response = self.client.chat.completions.create(
            **self._request_kwargs(messages, json_mode=json_mode))
        return response.choices[0].message.content or ""

    def _payload(self, path: str) -> str:
        """Base64 payload of one image, encoded once per run."""
        with self._payload_lock:
            payload = self._encoded.get(path)
            if payload is None:
                payload = encode_image_to_base64(path, self.image_max_side,
                                                 self.image_quality)
                self._encoded[path] = payload
            return payload

    def _vision(self, text: str, image_paths: Sequence[str]) -> str:
        """Multi-image completion for the perception stage."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"
                                                 + self._payload(path)}})
        return self._chat(content, json_mode=False)

    # ------------------------------------------------------------------
    # Stage 1: inputs
    # ------------------------------------------------------------------
    @staticmethod
    def compact_shap_agg(df: pd.DataFrame, top_n: int = 20) -> str:
        """Compact per-descriptor SHAP statistics for the reasoning prompt."""
        if df is None or df.empty:
            return ""
        top_n = min(top_n, df.shape[1])
        importance = df.abs().mean().nlargest(top_n)
        lines = []
        for rank, column in enumerate(importance.index, 1):
            series = df[column]
            lines.append(
                f"#{rank} {column}: |mean|={series.abs().mean():.3f}, "
                f"mean={series.mean():+.3f}, std={series.std():.3f}, "
                f"pos_ratio={(series > 0).mean():.2f}")
        return "\n".join(lines)

    def data_prep(
        self,
        shap_dependency_img_path: str,
        substructure_img_path: str,
        df_shap_descriptor_vec: pd.DataFrame,
        acin_substructure_weight: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Collect images, weights, SMARTS patterns, cohort and descriptor statistics."""
        from skfp.fingerprints import KlekotaRothFingerprint

        self._stage("1/4", "INPUTS")
        deps = collect_images(shap_dependency_img_path)
        # Order the dependency plots by |mean SHAP| so the cap keeps the curves
        # that the report actually discusses, not the alphabetically first ones.
        if deps and df_shap_descriptor_vec is not None:
            importance = df_shap_descriptor_vec.abs().mean().sort_values(ascending=False)
            by_name = {Path(p).stem: p for p in deps}
            ranked = [by_name[name] for name in importance.index if name in by_name]
            deps = ranked or deps
        shap_files = deps
        sub_files = collect_images(substructure_img_path)
        if df_shap_descriptor_vec is None:
            raise ValueError("df_shap_descriptor_vec is None -- SHAP analysis incomplete")

        weights = {str(k): float(v) for k, v in zip(
            acin_substructure_weight.columns, acin_substructure_weight.iloc[0])}
        self._patterns = KlekotaRothFingerprint(n_jobs=1).patterns
        self._cohort = {f"krfp_{m.group(1)}" for path in sub_files
                        if (m := re.search(r"krfp_(\d+)", os.path.basename(path)))}

        # The weight table must describe exactly the fragments whose images are
        # staged (the GUI's substructure_top_k), not a fixed global slice.
        ranked = sorted(((k, weights[k]) for k in self._cohort if k in weights),
                        key=lambda kv: -kv[1])
        if not ranked:
            ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:120]
        package = {
            "shap_images": shap_files[: (self.shap_image_cap or self.descriptor_top_n)],
            "substructure_images": sub_files,
            "shap_descriptor_text": self.compact_shap_agg(
                df_shap_descriptor_vec, self.descriptor_top_n),
            "acin_substructure_weights": acin_substructure_weight,
            "weights": weights,
            "cohort": self._cohort,
            "weight_table": "\n".join(f"{k}: {v:+.3f}" for k, v in ranked),
        }
        curves = [Path(p).stem for p in package["shap_images"]]
        self._think(f"  substructures: {len(sub_files)}, descriptors: {len(DESCRIPTORS)}")
        self._think(f"  SHAP: {len(curves)} dependence curves")
        for start in range(0, len(curves), 6):
            self._think("    " + ", ".join(curves[start:start + 6]))
        self._done(f"{len(sub_files)} structures, {len(curves)} curves")
        self._emit(MSG_PROGRESS, f"{1 / 5}")
        return package

    # ------------------------------------------------------------------
    # Stage 2: perception (weights visible; description only, batched)
    # ------------------------------------------------------------------
    def _absorb(self, parsed: Dict[str, Any], fragments: List[Dict[str, Any]],
                shap_obs: List[Dict[str, Any]], seen: set,
                seen_features: Optional[set] = None) -> None:
        """Normalise one perception reply into the running lists."""
        seen_features = seen_features if seen_features is not None else set()
        for record in parsed.get("substructure_observations", []):
            idx = self.normalize_sub_idx(str(record.get("substructure_idx", "")))
            if idx and idx not in seen:
                seen.add(idx)
                fragments.append({
                    "substructure_idx": idx,
                    "claim": str(record.get("claim", "other")).strip().lower(),
                    "desc": str(record.get("desc", "")),
                })
        for item in parsed.get("shap_dependence_observations", []):
            feature = str(item.get("feature", "")).strip()
            if not feature or feature.lower() in seen_features:
                continue
            seen_features.add(feature.lower())
            shap_obs.append(item)

    def _text_perception(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Classify each SMARTS pattern with text only, no images."""
        self._stage("2/4", "PERCEPTION (text only)")
        patterns = self._patterns
        ids = sorted(data_package.get("cohort") or set(),
                     key=lambda k: -data_package["weights"].get(k, -1.0))
        prompt_head = (
            "Classify each substructure pattern below using only its SMARTS string.\n"
            "Vocabulary:\n" + CLASS_DEFINITIONS +
            "\nReturn JSON only: {\"substructure_observations\":[{\"substructure_idx\":\"krfp_1\","
            "\"claim\":\"arene\",\"desc\":\"<=12 words\"}]}\n\nPatterns:\n")
        fragments: List[Dict[str, Any]] = []
        seen: set = set()
        for start in range(0, max(1, len(ids)), 40):
            chunk = ids[start:start + 40]
            lines = [f"{idx}: {Chem.MolToSmarts(patterns[int(idx.split('_')[-1])])}"
                     for idx in chunk]
            raw = self._chat(prompt_head + "\n".join(lines) + "\n/no_think", json_mode=False)
            parsed = json.loads(clean_llm_json(raw))
            self._absorb(parsed, fragments, [], seen)
            self._think(f"  {len(fragments)} records so far")
        self._done(f"{len(fragments)} records")
        return {"substructure_observations": fragments,
                "shap_dependence_observations": []}

    def vlm_perception(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the VLM to describe SHAP curves and substructure diagrams."""
        if self.text_only:
            return self._text_perception(data_package)
        self._stage("2/4", "PERCEPTION")
        prompt = (self.vlm_prompt.format(vocabulary=CLASS_DEFINITIONS)
                  + "\n\nACIN weight table (signed gating weights):\n"
                  + data_package.get("weight_table", ""))
        sub_images = data_package.get("substructure_images", [])
        fragments: List[Dict[str, Any]] = []
        shap_obs: List[Dict[str, Any]] = []
        seen: set = set()

        batches = plan_batches(sub_images, data_package.get("shap_images", []),
                               self.batch_bytes, self.batch_items, self.structure_batch,
                               size_of=lambda p: len(self._payload(p)))
        total = len(batches)
        print(f"[perception] {total} batch(es) x {self.concurrency} concurrent "
              f"for {len(sub_images)} structures", flush=True)

        def run_one(index: int, batch: Dict[str, List[str]]) -> List[Dict[str, Any]]:
            """Send one batch; split it in halves when the whole request fails."""
            images = batch["curves"] + batch["structures"]
            self._think(f"  batch {index}/{total}: {len(batch['curves'])} curves + "
                        f"{len(batch['structures'])} structures, "
                        f"{sum(len(self._payload(p)) for p in images) / 1e6:.1f} MB")
            try:
                return [json.loads(clean_llm_json(self._vision(prompt, images)))]
            except Exception as exc:
                self._think(f"  batch {index} failed ({type(exc).__name__}); "
                            f"retrying in halves")
                parts: List[Dict[str, Any]] = []
                half = max(1, len(images) // 2)
                for offset in range(0, len(images), half):
                    chunk = images[offset:offset + half]
                    try:
                        parts.append(json.loads(clean_llm_json(self._vision(prompt, chunk))))
                    except Exception as inner:
                        self._think(f"  batch {index} half failed ({type(inner).__name__})")
                return parts

        seen_features: set = set()
        jobs = list(enumerate(batches, 1))
        results: Dict[int, List[Dict[str, Any]]] = {}

        adaptive = self.concurrency <= 0
        planned = (2 if len(jobs) >= 3 else 1) if adaptive else self.concurrency
        limit = max(1, min(planned, self.max_concurrency))
        if limit > 1:
            self._think(f"  concurrency {limit} (cap {self.max_concurrency})")
        streak = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {}
            pending = deque(jobs)
            while pending or futures:
                while pending and len(futures) < limit:
                    index, batch = pending.popleft()
                    futures[pool.submit(run_one, index, batch)] = index
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    try:
                        parts = future.result()
                    except Exception as exc:
                        self._think(f"  batch {index} raised {type(exc).__name__}")
                        results[index] = []
                        streak = 0
                        continue
                    results[index] = parts
                    if parts:
                        streak += 1
                        if adaptive and streak >= 3 and limit < self.max_concurrency:
                            limit += 1
                            streak = 0
                            self._think(f"  three batches in a row -> concurrency {limit}")
                    else:
                        streak = 0
                        if limit > 1:
                            limit -= 1
                            self._think(f"  batch {index} lost -> concurrency {limit}")
                            time.sleep(3)

        # Merge in batch order so the outcome does not depend on completion order.
        for index in range(1, total + 1):
            for parsed in results.get(index, []):
                self._absorb(parsed, fragments, shap_obs, seen, seen_features)
                self._think(f"  -> {len(parsed.get('substructure_observations', []))} records, "
                            f"{len(parsed.get('shap_dependence_observations', []))} curves")

        self._done(f"{len(fragments)} records from {len(batches)} batch(es)")
        self._emit(MSG_PROGRESS, f"{2 / 5}")
        return {"substructure_observations": fragments,
                "shap_dependence_observations": shap_obs}

    # ------------------------------------------------------------------
    # Stage 3: verification (id exists, id was shown, class matches the SMARTS graph)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_sub_idx(name: str) -> str:
        """Normalize a free-form substructure label into 'krfp_<n>'."""
        if not isinstance(name, str):
            return ""
        s = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        s = re.sub(r"^krfp(?=\d)", "krfp_", s)
        s = re.sub(r"^krfp_+", "krfp_", s)
        match = re.search(r"krfp_\d+", s)
        return match.group(0) if match else s

    @staticmethod
    def _evidence_repr(sub_idx: str, contrib_label: str, weight: float) -> str:
        """Compact single-line rendering of one evidence row for the trace."""
        return (f'{{"substructure_idx": {sub_idx}, "contrib_label": {contrib_label}, '
                f'"acin_weight": {weight}}}')

    def hierarchy_checker(
        self,
        vlm_observations: List[Dict[str, Any]],
        acin_weights: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Verify every observation and return evidence in msec.py's format."""
        self._stage("3/4", "VERIFICATION")
        self._t_verification = time.perf_counter()
        weights = {str(k): float(v) for k, v in zip(
            acin_weights.columns, acin_weights.iloc[0])}
        patterns = self._patterns or []

        seen: set = set()
        evidence: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        stats = {"seen": 0, "kept": 0, "no_index": 0, "not_staged": 0,
                 "bad_class": 0, "duplicate": 0, "other": 0, "coarse": 0,
                 "examples": []}

        for obs in vlm_observations:
            idx = self.normalize_sub_idx(str(obs.get("substructure_idx", "")))
            if not idx:
                continue
            stats["seen"] += 1
            claim = str(obs.get("claim", "other")).strip().lower()
            if idx in seen:
                stats["duplicate"] += 1
                rows.append({"id": idx, "weight": None, "claim": claim,
                             "ok": False, "reason": "duplicate observation"})
                continue
            seen.add(idx)
            if idx not in weights:
                stats["no_index"] += 1
                stats["examples"].append({"id": idx, "reason": "no_index"})
                rows.append({"id": idx, "weight": None, "claim": claim,
                             "ok": False, "reason": "not in the 4,860-bit space"})
                continue
            if self._cohort and idx not in self._cohort:
                stats["not_staged"] += 1
                stats["examples"].append({"id": idx, "reason": "not_staged"})
                rows.append({"id": idx, "weight": None, "claim": claim,
                             "ok": False, "reason": "not shown in this batch"})
                continue

            bit = int(idx.split("_")[-1])
            if not class_supported(patterns[bit], claim):
                stats["bad_class"] += 1
                stats["examples"].append({"id": idx, "reason": f"class_not_supported:{claim}"})
                rows.append({"id": idx, "weight": float(weights[idx]), "claim": claim,
                             "ok": False, "reason": f"class not supported by SMARTS"})
                continue

            weight = float(weights[idx])
            if claim == "other":
                stats["other"] += 1
            matched = classify(patterns[bit])
            if claim in UMBRELLA_CLASSES and any(
                    c not in UMBRELLA_CLASSES for c in matched):
                stats["coarse"] += 1
            rows.append({"id": idx, "weight": weight, "claim": claim,
                         "ok": True, "reason": ""})
            evidence.append({
                "substructure_idx": idx,
                "contrib_label": CONTRIB_POSITIVE if weight > 0 else CONTRIB_NEGATIVE,
                "acin_weight": weight,
                "smarts_pattern": Chem.MolToSmarts(patterns[bit]),
                "claim": claim,
                "desc": str(obs.get("desc", "")),
            })

        stats["kept"] = len(evidence)
        stats["verified_rate"] = ((len(evidence) - stats["other"]) / len(evidence)
                                  if evidence else 0.0)
        evidence.sort(key=lambda e: -e["acin_weight"])
        self.verification = {**stats, "examples": stats["examples"][:10]}
        self._verify_table(rows, stats)
        self._emit(MSG_PROGRESS, f"{3 / 5}")
        return evidence

    # ------------------------------------------------------------------
    # Stage 4: prompt assembly (usable fragments ordered by weight, capped)
    # ------------------------------------------------------------------
    def _build_pool(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Candidate pool: usable fragments ordered by weight, capped by pool_size."""
        usable = [e for e in evidence if e["claim"] not in NON_DESIGN_CLASSES]
        pool = sorted(usable, key=lambda e: -e["acin_weight"])[: self.pool_size]

        annotated: List[Dict[str, Any]] = []
        for item in pool:
            annotated.append({
                "substructure_idx": item["substructure_idx"],
                "acin_weight": round(item["acin_weight"], 3),
                "contrib": item["contrib_label"],
                "claim": item["claim"],
                "desc": item["desc"],
                "smarts_pattern": item["smarts_pattern"],
            })
        return annotated

    # ------------------------------------------------------------------
    # Stage 5: reasoning in the original report framework
    # ------------------------------------------------------------------
    def _render_cot(self, pool: List[Dict[str, Any]], shap_descriptor_text: str,
                    shap_dependence_observations: List[Dict[str, Any]],
                    verification: Dict[str, Any]) -> str:
        """Fill the report-framework prompt for one candidate pool."""
        return self.cot_prompt.format(
            valid_evidence=json.dumps(pool, ensure_ascii=False),
            shap_descriptor_text=shap_descriptor_text,
            descriptor_dictionary=json.dumps(self._dictionary_for(shap_descriptor_text),
                                             ensure_ascii=False),
            shap_dependence_observations=json.dumps(
                shap_dependence_observations, ensure_ascii=False),
            verification=json.dumps({k: v for k, v in verification.items()
                                     if k != "examples"}, ensure_ascii=False),
            max_recommendations=self.max_recommendations,
        )

    def _dictionary_for(self, descriptor_text: str) -> Dict[str, str]:
        """Definitions of the descriptors that actually appear in the statistics."""
        return {k: v for k, v in self.DESCRIPTOR_DICTIONARY.items() if k in descriptor_text}

    def cot_reasoner(
        self,
        valid_evidence: List[Dict[str, Any]],
        shap_descriptor_text: str,
        shap_dependence_observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the reasoning pass; streams the Markdown report before the JSON block."""
        pool = self._build_pool(valid_evidence)
        verification = getattr(self, "verification", {})

        system_prompt = self._render_cot(pool, shap_descriptor_text,
                                         shap_dependence_observations, verification)
        self._emit(MSG_PROGRESS, f"{4 / 5}")
        self._stage("4/4", "REASONING")
        usable = sum(1 for e in valid_evidence if e["claim"] not in NON_DESIGN_CLASSES)
        self._think(f"  pool {len(pool)} of {usable} usable fragments "
                    f"(cap {self.pool_size}), prompt {len(system_prompt):,} chars")
        raw = ""
        ladder = [pool] + [pool[:n] for n in (24, 16, 10, 6) if n < len(pool)]
        for attempt_pool in ladder:
            prompt_attempt = system_prompt
            if attempt_pool is not pool:
                prompt_attempt = self._render_cot(
                    attempt_pool, shap_descriptor_text,
                    shap_dependence_observations, verification)
                print(f"[reason] retry with pool={len(attempt_pool)}", flush=True)
            self.cot_history_msg = [{"role": "user", "content": prompt_attempt}]
            try:
                raw = self._stream_reasoning(self.cot_history_msg)
                break
            except Exception as exc:
                self._think(f"  attempt failed: {type(exc).__name__}: {str(exc)[:120]}")
                if getattr(self, "_reply_emitted", False) or attempt_pool is ladder[-1]:
                    raise
        self.cot_history_msg.append({"role": "assistant", "content": raw})
        result = self._parse_cot_output(raw)

        analysis = self._postcheck(result, pool)
        self._emit(MSG_PROGRESS, f"{5 / 5}")
        self._emit(MSG_INIT)
        return analysis

    @staticmethod
    def _parse_cot_output(raw: str) -> Dict[str, Any]:
        """Split a reply into the Markdown report and the trailing JSON block."""
        markdown, payload = (raw.split(SEP_JSON, 1) if SEP_JSON in raw
                             else (raw, extract_json_object(raw)))
        result: Dict[str, Any] = {"raw_output": raw}
        try:
            result.update(json.loads(clean_llm_json(payload)))
        except json.JSONDecodeError:
            pass
        if markdown.strip():
            result["human_readable_summary"] = markdown.strip()
        return result

    def _postcheck(self, analysis: Dict[str, Any],
                   pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-check: recommendations inside the pool, real descriptors, no Bandgap handle."""
        pool_ids = {e["substructure_idx"] for e in pool}
        recs = analysis.get("recommended_substructures") or []
        rec_ids = []
        for item in recs:
            idx = self.normalize_sub_idx(str(item.get("substructure_idx", "")
                                             if isinstance(item, dict) else item))
            if idx:
                rec_ids.append(idx)

        # Fallback: the JSON block may be truncated even when the table survived.
        if not rec_ids:
            table = analysis.get("human_readable_summary", "")
            table = table.split("Recommended Substructures", 1)[-1]
            table = table.split("Favorable Combinations", 1)[0]
            rec_ids = re.findall(r"krfp_\d+", table)

        text = analysis.get("human_readable_summary", "")
        mentioned = set(re.findall(r"\b[A-Za-z]+(?:_[A-Za-z]+)+\b", text))
        stats_words = {"pos_ratio", "abs_mean", "mean", "std", "n_obs", "n_scored",
                       "pass_rate", "verified_rate", "n_kept", "n_seen"}
        unknown_descriptors = sorted(d for d in mentioned
                                     if d.endswith(("_avg", "_total", "_diff", "_gap",
                                                    "_ratio", "_range"))
                                     and d not in DESCRIPTORS and d not in stats_words)
        bandgap_in_rules = "Bandgap" in text.split("## 4", 1)[-1]

        analysis["checks"] = {
            "recommended_outside_pool": sorted(set(rec_ids) - pool_ids),
            "n_recommended": len(rec_ids),
            "unknown_descriptors": unknown_descriptors,
            "bandgap_in_design_rules": bandgap_in_rules,
            "pool_size": len(pool),
        }
        return analysis

    # ------------------------------------------------------------------
    # Multi-turn continuation
    # ------------------------------------------------------------------
    def multi_turn_conv(self, text: str) -> None:
        """Continue the conversation after the initial analysis."""
        if not self.cot_history_msg or not text.strip():
            self._emit(MSG_END)
            return

        self.cot_history_msg.append({
            "role": "user",
            "content": text + "\nPlease answer in natural language and do not output JSON.",
        })
        self._think(f"{DIVIDER} Follow-up question... {DIVIDER}")
        try:
            raw = self._stream_reasoning(self.cot_history_msg)
        except Exception as exc:
            self.cot_history_msg.pop()          # drop the dangling user turn
            self._think(f"  follow-up failed: {type(exc).__name__}")
            self.send_error(self.q, exc)
            self._emit(MSG_END)
            return

        self.cot_history_msg.append({"role": "assistant", "content": raw})
        self._emit(MSG_END)


    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        shap_dependency_img_path: str,
        substructure_img_path: str,
        df_shap_descriptor_vec: pd.DataFrame,
        acin_substructure_weight: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run perception, verification and reasoning; return a structured result."""
        try:
            package = self.data_prep(
                shap_dependency_img_path=shap_dependency_img_path,
                substructure_img_path=substructure_img_path,
                df_shap_descriptor_vec=df_shap_descriptor_vec,
                acin_substructure_weight=acin_substructure_weight,
            )
            vlm_obs = self.vlm_perception(package)
            valid_evidence = self.hierarchy_checker(
                vlm_obs["substructure_observations"],
                package["acin_substructure_weights"],
            )
            final_reply = self.cot_reasoner(
                valid_evidence,
                package["shap_descriptor_text"],
                vlm_obs["shap_dependence_observations"],
            )
            return {
                "status": "ok",
                "verification": getattr(self, "verification", {}),
                "vlm_raw_observation": vlm_obs,
                "valid_evidence_after_hierarchy_checker": valid_evidence,
                "analysis_result": final_reply,
            }
        except Exception as exc:
            traceback.print_exc()
            self.send_error(self.q, exc)
            raise


# The feature list is the key set of the dictionary above: one source of truth.
DESCRIPTORS = tuple(MSECAgent.DESCRIPTOR_DICTIONARY)


# ---------------------------------------------------------------------------
# Shared helpers (used by the GUI and the analysis scripts)
# ---------------------------------------------------------------------------
def load_weights(weights_path: Path) -> Dict[str, float]:
    """Read 'acin_weights.csv' into a {krfp_id: weight} mapping."""
    df = pd.read_csv(weights_path)
    return {str(k): float(v) for k, v in zip(df["krfp_idx"], df["weights"])}


def read_endpoint_config() -> Dict[str, str]:
    """Read the endpoint from EHC_* or AGENT_* environment variables."""
    cfg = {"api_key": os.environ.get("EHC_API_KEY") or os.environ.get("AGENT_API_KEY", ""),
           "base_url": os.environ.get("EHC_BASE_URL") or os.environ.get("AGENT_BASE_URL", ""),
           "model": os.environ.get("EHC_MODEL") or os.environ.get("AGENT_MODEL_NAME", "")}
    return cfg
