"""MSEC: VLM perception, hierarchy check, and CoT reasoning.

Drives the multi-modal explanation pipeline behind the GUI's *Agent* page:

    1. 'vlm_perception'    — describe SHAP plots + substructure diagrams.
    2. 'hierarchy_checker' — reconcile VLM contributions with ACIN weights.
    3. 'cot_reasoner'      — chain-of-thought synthesis into a Markdown
                               report plus a structured JSON block.

Input:
    shap_*_img_path, substructure_img_path   -- images fed to the VLM.
    df_shap_descriptor_vec                   -- per-descriptor SHAP values.
    acin_substructure_weight                 -- ACIN attention weights.
    q, stop_event                            -- GUI queue + cancellation flag.
    api_key, base_url, model, temperature    -- OpenAI-compatible endpoint.
    max_tokens (<=0 = no cap), thresh,
    descriptor_top_n, thinking               -- endpoint + tuning knobs.

Output:
    run_pipeline() -> {"status", "vlm_raw_observation",
                       "valid_evidence_after_hierarchy_checker",
                       "analysis_result"}.
    Messages on ``q``: Think / Reply / _PROGR / _Init / _End / _Error_
        (prefix is the channel; suffix is the payload).
"""

import base64
import json
import os
import re
import traceback
from queue import Queue  # kept for API compatibility
from typing import Any, Dict, Generator, List, Optional, Tuple
from skfp.fingerprints import KlekotaRothFingerprint
import httpx
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

MSG_THINK = "Think"
MSG_REPLY = "Reply"
MSG_PROGRESS = "_PROGR"
MSG_INIT = "_Init"
MSG_END = "_End"
MSG_ERROR = "_Error_"

CONTRIB_HIGH = "high_contrib"
CONTRIB_LOW = "low_contrib"

SEP_JSON = "---JSON---"
TEXT_WRAP_LIMIT = 160
DIVIDER = "-" * 35


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def clean_llm_json(raw_text: str) -> str:
    """Strip Markdown code fences from an LLM JSON payload."""
    s = raw_text.strip()
    if s.startswith("'`json"):
        s = s.removeprefix("'`json").removesuffix("'`").strip()
    elif s.startswith("'`"):
        s = s.removeprefix("'`").removesuffix("'`").strip()
    return s


def extract_json_object(raw_text: str) -> str:
    """Return the first valid JSON object/array embedded in 'raw_text'."""
    if not raw_text:
        return ""

    m = re.search(r"'`(?:json)?\s*(.+?)\s*'`", raw_text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end > start:
        candidate = raw_text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    start, end = raw_text.find("["), raw_text.rfind("]")
    if start != -1 and end > start:
        candidate = raw_text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return raw_text.strip()


def encode_image_to_base64(img_path: str) -> str:
    """Return the base64 payload of an image file (no data-URI prefix)."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MSECAgent:
    """Multi-modal explainability agent for Co-SAM buried interfaces."""

    DESCRIPTOR_DICTIONARY = {
        "Bandgap": "Experimentally measured bandgap value (eV). Note: Device-level parameter; exclude from molecular design synergistic mechanisms.",
        "Mixture": "Binary indicator for binary mixed system (0 = single component, 1 = two-component mixture).",
        "HeteroEndFrag": "Binary indicator for the presence of asymmetric terminal fragments (0 = no, 1 = yes).",
        "HeteroSym": "Binary indicator for hetero-symmetry in the system (0 = no, 1 = one symmetric and one asymmetric molecule).",
        "BthAsymStr": "Binary indicator for whether both molecules in the pair are asymmetric (0 = no, 1 = yes).",
        "UnifChainLen": "Binary indicator for whether the two molecules have uniform chain lengths (0 = no, 1 = yes).",
        "MACl_additive": "Binary indicator for the presence of MACl additive (0 = no, 1 = yes).",
        "HOMO_avg": "Molar-ratio weighted average of the Highest Occupied Molecular Orbital energy levels (eV).",
        "LUMO_avg": "Molar-ratio weighted average of the Lowest Unoccupied Molecular Orbital energy levels (eV).",
        "HOMO_LUMO_gap": "Absolute energy difference between HOMO_avg and LUMO_avg (eV).",
        "HOMO_diff": "Absolute difference between the HOMO levels of the two molecules (eV), reflecting energy level gradient.",
        "LUMO_diff": "Absolute difference between the LUMO levels of the two molecules (eV).",
        "Dipole_avg": "Molar-ratio weighted average of the molecular dipole moments (Debye).",
        "ESP_max_avg": "Molar-ratio weighted average of the maximum electrostatic potential values (a.u.).",
        "ESP_min_avg": "Molar-ratio weighted average of the minimum electrostatic potential values (a.u.).",
        "ESP_range": "Range between maximum and minimum electrostatic potentials (a.u.), reflecting polarizability.",
        "MPI_avg": "Molar-ratio weighted average of the Molecular Polarity Index (kcal/mol).",
        "Vol_avg": "Molar-ratio weighted average of molecular volumes (Å³).",
        "Vol_ratio": "Ratio of the volumes of the two molecules (Vol_sam1 / Vol_sam2), reflecting spatial complementarity.",
        "Rho_avg": "Molar-ratio weighted average of molecular densities (g/cm³).",
        "MW_avg": "Molar-ratio weighted average of molecular weights (g/mol).",
        "LogP_avg": "Molar-ratio weighted average of the partition coefficients (LogP), reflecting hydrophobicity.",
        "LogP_diff": "Absolute difference between the LogP values of the two molecules.",
        "TPSA_avg": "Molar-ratio weighted average of the Topological Polar Surface Area (Å²), reflecting polarity.",
        "TPSA_diff": "Absolute difference between the TPSA values of the two molecules.",
        "HBA_total": "Total number of Hydrogen Bond Acceptors across both molecules (SAM1 + SAM2), reflecting passivation capability.",
        "HBD_total": "Total number of Hydrogen Bond Donors across both molecules (SAM1 + SAM2).",
        "FCsp3_avg": "Molar-ratio weighted average of the fraction of sp3-hybridized carbon atoms, reflecting molecular flexibility.",
        "AR_total": "Total number of Aromatic Rings across both molecules (SAM1 + SAM2).",
        "RB_total": "Total number of Rotatable Bonds across both molecules (SAM1 + SAM2), reflecting conformational flexibility.",
        "sp_total": "Total number of sp-hybridized atoms across both molecules (SAM1 + SAM2).",
        "sp2_total": "Total number of sp2-hybridized atoms across both molecules (SAM1 + SAM2), affecting pi-pi stacking.",
        "sp3_total": "Total number of sp3-hybridized atoms across both molecules (SAM1 + SAM2).",
        "HA_total": "Total number of heavy (non-hydrogen) atoms across both molecules (SAM1 + SAM2).",
        "SC_total": "Total number of stereocenters across both molecules (SAM1 + SAM2)."
    }

    vlm_prompt = """
Task: Pure visual observation and chemical intuition. DO NOT infer physical mechanisms.

You are looking at two types of images:
1. SHAP dependence plots for physicochemical descriptors.
2. Molecular substructure diagrams labeled "KRFP <n>".

For substructure images:
- Each has a label at the top like "KRFP 123".
- Convert it to "krfp_123" (lowercase, underscore, number unchanged).
- CRITICAL: You CANNOT see the numerical weights. You MUST judge whether it is "high_contrib" or "low_contrib" purely based on your chemical intuition using the following rules:
  - "high_contrib" (High Contribution): Look for complex conjugated systems, fused aromatic rings, anchoring groups (like phosphonic acid, silane, carboxylic acid), sulfur/heteroatom-containing fragments, or extended rigid backbones. These are typically crucial for Co-SAM buried interface interactions.
  - "low_contrib" (Low Contribution): Look for simple isolated alkyl chains, small unbranched substituents, simple non-conjugated rings, or structures lacking functional anchoring groups.
- You MUST output either "high_contrib" or "low_contrib" for every substructure.

For SHAP dependence plots:
- Describe the visible trend of SHAP value vs. feature value.
- Identify if there is an optimal range, a peak, a threshold, or a monotonic trend.
- Do NOT explain why. Just report what the curve/points show.

Output ONLY pure JSON:
{
  "substructure_observations": [
    {"substructure_idx": "krfp_<n>", "contrib": "high_contrib|low_contrib"}
  ],
  "shap_dependence_observations": [
    {
      "feature": "<descriptor name visible on plot>",
      "trend": "<monotonic increasing|monotonic decreasing|peak|valley|threshold|complex>",
      "optimal_range": "<if visible, e.g. '1.3-1.5', '>8', 'around 64'>",
      "sign_region": "<where SHAP > 0, where SHAP < 0>"
    }
  ]
}
No extra text, no markdown.
""".strip()

    cot_prompt = """
You are a data-driven materials scientist working on Co-SAM buried interfaces.

You are given:
1. valid_evidence (substructure-level, filtered by ACIN weights):
{valid_evidence}

2. shap_descriptor_text (CatBoost SHAP global attribution):
{shap_descriptor_text}

3. descriptor_dictionary (Physicochemical definitions, units, and calculation logic):
{descriptor_dictionary}

4. shap_dependence_observations (visual trends from SHAP dependence plots):
{shap_dependence_observations}

Follow this reasoning procedure strictly:

Step 1: Descriptor-level mechanism extraction
- For each top descriptor in shap_descriptor_text, interpret:
  - sign of mean: does higher feature value increase or decrease predicted PCE?
  - pos_ratio: how consistent is the positive contribution?
  - |mean|: how strong is the average impact?
  - std: how variable is the contribution across samples?
- If shap_dependence_observations are available, use them to infer optimal ranges or thresholds.
- Summarize 3-5 descriptor-level mechanisms. Each must cite at least one descriptor and its statistics.

Step 2: Substructure-level mechanism extraction
- Group valid_evidence by chemical intuition. You MUST use the provided "smarts_pattern" field to deduce the chemical fragment type for each substructure.
  You may describe fragment types based on the SMARTS, such as:
  "fused aromatic rings", "sulfur-containing aromatic fragments", "thioether linkers",
  "phosphonic acid anchoring groups", "flexible alkyl chains", "rigid conjugated backbones",
  "symmetric vs asymmetric substitution patterns".
- For each group, report:
  - which krfp indices belong to it,
  - their acin_weight range,
  - whether they are high_contrib or low_contrib.
- Do NOT invent fragments that are not supported by the krfp indices and weights.

Step 3: Synergy hypothesis
- Combine Step 1 and Step 2.
- Propose 4 synergistic mechanisms for the Co-SAM buried interface.
  Each mechanism must connect:
  - one or more physicochemical descriptors with optimal ranges/trends,
  - one or more substructure groups with high ACIN weights,
  - a plausible physical/chemical role (polarity, energy alignment, packing, passivation, etc.).
- Do NOT use Bandgap in the synergistic mechanisms. Bandgap is a device-level band structure parameter rather than a molecular design handle for Co-SAM, so exclude it from all Step 3 mechanisms even if it appears in Step 1.
- Use cautious language: "evidence suggests", "may indicate", "consistent with".
- Do NOT mention any specific molecule name unless it is directly given in the input.

Step 4: Evidence-based Design recommendations
- CRITICAL: Do NOT simply rank and select substructures by their acin_weight. High weight alone does not guarantee a good design recommendation. You must select based on the chemical and physical EVIDENCE you gathered in Steps 1-3.
- Select exactly 10 high_contrib substructures from valid_evidence that best align with the descriptor trends and synergistic mechanisms identified earlier.
  If fewer than 10 high_contrib substructures exist, select all available and explicitly state "insufficient evidence for 10".
  Do NOT invent or use any krfp outside valid_evidence.
- recommended_substructures: output exactly 10 selected substructures, or all available if fewer than 10.
  Each item must include:
  - "substructure_idx": "krfp_<n>"
  - "acin_weight": numeric value
  - "contrib": "high_contrib"
  - "reason": concise reason explaining WHY this substructure is recommended based on your evidence (e.g., "Linked to polarity passivation via HBA_total", "Contributes to energy alignment based on HOMO trends"). Do NOT use "highest ACIN weight" as the reason.
- favorable_substructure_combinations: group the selected substructures into 2-4 synergistic combinations.
  Each combination must explicitly list its krfp indices in "krfp_indices".
  The union of unique krfp indices across all favorable_substructure_combinations MUST equal the 10 recommended_substructures.
- avoided_substructure_combinations: list combinations that may be antagonistic or unsupported.
- If evidence is insufficient, say "insufficient evidence".

First, output a Markdown report with exactly the following sections:

When outputting Markdown tables, escape any vertical bar in cell content as \| (or use &#124;), and do not insert line breaks inside a table row. Keep each table row on a single line.
CRITICAL: Each table row must be exactly one physical line. Do not wrap cell content across lines. If the content is long, shorten it instead of wrapping. Any newline inside a table will break rendering.

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
### Recommended Substructures (Evidence-Based Top 10)
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

Do not output any other text before the Markdown report, between the Markdown report and ---JSON---, or after the JSON object.
CRITICAL: You MUST output the line "---JSON---" exactly, on its own line, after the Markdown report and before the JSON object. Do not omit it.
""".strip()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
        stop_event=None,
        q: "Queue" = None,
        thresh: float = 0.15,
        descriptor_top_n: int = 20,
        thinking: bool = True,
    ) -> None:
        self.thresh = thresh
        self.descriptor_top_n = descriptor_top_n
        self.q = q
        self.stop_event = stop_event

        # 'trust_env=False' bypasses any corporate proxy settings the
        # OpenAI SDK would otherwise honor; required for the direct MaaS
        # endpoint used in production.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        self.model_name = model
        self.temperature = temperature
        # 'max_tokens <= 0' means "no limit": the 'max_output_tokens'
        # kwarg is omitted entirely (not passed as 'None') so the
        # provider applies its own default.
        self.max_tokens: Optional[int] = (
            int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
        )
        self.thinking = thinking
        self.cot_history_msg: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # GUI message plumbing
    # ------------------------------------------------------------------

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
        return self.stop_event is not None and self.stop_event.is_set()

    @staticmethod
    def _safe_close(resp) -> None:
        try:
            resp.close()
        except Exception:
            pass

    @staticmethod
    def send_error(q, e) -> None:
        if q is not None:
            q.put(f"{MSG_ERROR}Error: {e}")

    # ------------------------------------------------------------------
    # Responses-object helpers
    # ------------------------------------------------------------------

    def _base_kwargs(self) -> Dict[str, Any]:
        """Common kwargs for 'responses.create'.

        'max_output_tokens' is only present when 'self.max_tokens' is a
        positive integer.  Omitting the key lets the provider use its own
        default; passing 'None' would be sent verbatim and rejected by
        most back-ends.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        return kwargs

    @staticmethod
    def _extract_text(resp) -> str:
        """Robustly pull text from a Responses object.

        'resp.output_text' is the SDK convenience aggregator; some
        third-party providers populate 'resp.output[*].content[*].text'
        but leave the aggregator empty.
        """
        text = getattr(resp, "output_text", None)
        if text:
            return text
        chunks: List[str] = []
        try:
            for item in getattr(resp, "output", None) or []:
                for block in getattr(item, "content", None) or []:
                    t = getattr(block, "text", None)
                    if t:
                        chunks.append(t)
        except Exception:
            pass
        return "".join(chunks)

    @staticmethod
    def _diagnose_empty(resp) -> str:
        """Best-effort diagnosis of an empty Responses object."""
        parts: List[str] = []
        try:
            status = getattr(resp, "status", None)
            if status:
                parts.append(f"status={status}")
        except Exception:
            pass
        try:
            incomplete = getattr(resp, "incomplete_details", None)
            if incomplete is not None:
                reason = getattr(incomplete, "reason", None) or str(incomplete)
                parts.append(f"incomplete_reason={reason}")
        except Exception:
            pass
        try:
            err = getattr(resp, "error", None)
            if err is not None:
                code = getattr(err, "code", None) or "?"
                message = getattr(err, "message", None) or str(err)
                parts.append(f"error.code={code} error.message={message}")
        except Exception:
            pass
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                out_tokens = getattr(usage, "output_tokens", None)
                if out_tokens is not None:
                    parts.append(f"output_tokens={out_tokens}")
        except Exception:
            pass
        try:
            refusal = getattr(resp, "refusal", None)
            if refusal:
                parts.append(f"refusal={refusal}")
        except Exception:
            pass
        return " | ".join(parts) if parts else "no diagnostic fields available"

    @staticmethod
    def _incomplete_reason(resp) -> Optional[str]:
        """Return 'incomplete_details.reason' when the response was cut short."""
        try:
            if getattr(resp, "status", None) != "incomplete":
                return None
            incomplete = getattr(resp, "incomplete_details", None)
            if incomplete is None:
                return "unknown"
            reason = getattr(incomplete, "reason", None)
            return reason or str(incomplete) or "unknown"
        except Exception:
            return None

    def _emit_truncation_warning(self, reason: str) -> None:
        """Warn the user that the reply was cut short by 'max_output_tokens'."""
        if reason == "max_output_tokens":
            if self.max_tokens is not None:
                hint = (
                    f"Output truncated: this process is using max_tokens="
                    f"{self.max_tokens} (captured at startup). "
                    f"If you changed max_tokens or set it to 0 for no limit, "
                    f"click New Chat to apply the new value."
                )
            else:
                hint = "Output truncated by the provider's token limit."
        else:
            hint = f"Output truncated ({reason})."

            # Always surface the warning on the chat transcript, so it survives
            # even when the reasoning trace is hidden and lands in the saved report.
        self._emit(MSG_REPLY, f"\n\n> ⚠️ **{hint}**\n")

        if self.thinking:
            self._emit(MSG_THINK, f"\n⚠️ {hint}\n")

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_text(delta: str, count: int, limit: int = TEXT_WRAP_LIMIT) -> Tuple[str, int]:
        """Insert a newline every ~'limit' characters to keep the GUI readable."""
        count += len(delta)
        if "\n" in delta:
            count = 0
        elif count >= limit and " " in delta:
            count = 0
            delta = delta.replace(" ", "\n", 1)
        return delta, count

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_response(
        self,
        resp,
        *,
        split_json: bool = False,
        progress_thinking: Optional[float] = None,
        progress_answer: Optional[float] = None,
    ) -> Tuple[str, str, bool]:
        """Consume a streaming Responses object.

        Sends 'Think' / 'Reply' / '_PROGR' messages as events arrive.
        Returns '(raw_reasoning, raw_output, stopped)'.

        Detects 'response.incomplete' events and emits a truncation
        warning (via '_emit_truncation_warning') once the stream is done;
        partial output is preserved so downstream parsing still works.
        """
        raw_reasoning = ""
        raw_output = ""
        text_wrap_count = 0
        thinking_start = False
        answer_start = False
        json_started = False
        progress_started = False
        display_buffer = ""
        truncated_reason: Optional[str] = None

        for event in resp:
            if not progress_started:
                if progress_thinking is not None:
                    self._emit(MSG_PROGRESS, f"{progress_thinking}")
                progress_started = True

            # Truncation is signaled by a dedicated terminal event; catch
            # it before the content handlers and skip to the next event.
            if event.type == "response.incomplete":
                try:
                    incomplete = getattr(getattr(event, "response", None),
                                          "incomplete_details", None)
                    truncated_reason = getattr(incomplete, "reason", None) or "unknown"
                except Exception:
                    truncated_reason = "unknown"
                continue

            if event.type == "response.reasoning_text.delta" and self.thinking:
                if not thinking_start:
                    print(f"{DIVIDER} 🤔Thinking... {DIVIDER}", end="\n", flush=True)
                    self._emit(MSG_THINK, f"{DIVIDER} 🤔Thinking... {DIVIDER}\n")
                    thinking_start = True

                delta = event.delta
                raw_reasoning += delta
                if self._stopped():
                    self._safe_close(resp)
                    return raw_reasoning, raw_output, True

                delta, text_wrap_count = self._wrap_text(delta, text_wrap_count)
                print(delta, end="", flush=True)
                self._emit(MSG_THINK, delta)

            elif event.type == "response.output_text.delta":
                if not answer_start:
                    print(f"\n{DIVIDER} 📝Answering... {DIVIDER}\n", end="", flush=True)
                    if progress_answer is not None:
                        self._emit(MSG_PROGRESS, f"{progress_answer}")
                    answer_start = True
                    text_wrap_count = 0

                delta = event.delta
                raw_output += delta

                to_send = ""
                if split_json and not json_started:
                    display_buffer += delta
                    if SEP_JSON in display_buffer:
                        to_send, _ = display_buffer.split(SEP_JSON, 1)
                        json_started = True
                        display_buffer = ""
                    else:
                        keep = len(SEP_JSON) - 1
                        if len(display_buffer) > keep:
                            to_send = display_buffer[:-keep]
                            display_buffer = display_buffer[-keep:]
                elif not split_json:
                    to_send = delta

                if not to_send:
                    continue

                if self._stopped():
                    self._safe_close(resp)
                    return raw_reasoning, raw_output, True

                to_send, text_wrap_count = self._wrap_text(to_send, text_wrap_count)
                print(to_send, end="", flush=True)
                self._emit(MSG_REPLY, to_send)

        if truncated_reason is not None:
            self._emit_truncation_warning(truncated_reason)

        return raw_reasoning, raw_output, False

    def _finalize_history(self, raw_output: str, stopped: bool) -> None:
        """Append the assistant reply, or roll back the dangling user turn."""
        if not stopped:
            self.cot_history_msg.append({"role": "assistant", "content": raw_output})
        elif self.cot_history_msg and self.cot_history_msg[-1].get("role") == "user":
            self.cot_history_msg.pop()

    # ------------------------------------------------------------------
    # Stage 1: data preparation
    # ------------------------------------------------------------------

    def data_prep(
        self,
        shap_beeswarm_img_path: str,
        shap_dependency_img_path: str,
        substructure_img_path: str,
        df_shap_descriptor_vec: pd.DataFrame,
        acin_substructure_weight: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Collect images, encode them to base64, and compact SHAP stats."""
        try:
            shap_files = (
                self.collect_images(shap_beeswarm_img_path)
                + self.collect_images(shap_dependency_img_path)
            )
            sub_files = self.collect_images(substructure_img_path)

            if df_shap_descriptor_vec is None:
                raise ValueError(
                    "df_shap_descriptor_vec is None — SHAP analysis incomplete"
                )

            package = {
                "shap_images_b64": [encode_image_to_base64(p) for p in shap_files],
                "substructure_image_b64": [encode_image_to_base64(p) for p in sub_files],
                "shap_descriptor_text": self.compact_shap_agg(
                    df_shap_descriptor_vec, top_n=self.descriptor_top_n
                ),
                "acin_substructure_weights": acin_substructure_weight,
            }
            self._emit(MSG_PROGRESS, f"{1 / 5}")
            return package
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            raise

    @staticmethod
    def compact_shap_agg(df: pd.DataFrame, top_n: int = 20) -> str:
        """Compact per-descriptor SHAP statistics for the CoT prompt."""
        if df is None or df.empty:
            return ""
        top_n = min(top_n, df.shape[1])
        imp = df.abs().mean().nlargest(top_n)
        lines = []
        for rank, col in enumerate(imp.index, 1):
            series = df[col]
            lines.append(
                f"#{rank} {col}: |mean|={series.abs().mean():.3f}, "
                f"mean={series.mean():+.3f}, std={series.std():.3f}, "
                f"pos_ratio={(series > 0).mean():.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def collect_images(path: str) -> List[str]:
        """Enumerate image files under 'path' (file or directory)."""
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if not path:
            return []
        if os.path.isdir(path):
            return [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.lower().endswith(exts)
            ]
        if os.path.isfile(path) and path.lower().endswith(exts):
            return [path]
        return []

    # ------------------------------------------------------------------
    # Stage 2: VLM perception
    # ------------------------------------------------------------------

    def vlm_perception(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the VLM to describe SHAP plots and substructure diagrams."""
        print(f"{DIVIDER} 👀Observing images... {DIVIDER}")
        self._emit(
            MSG_THINK, f"{DIVIDER} 👀Observing images... {DIVIDER}",
            only_if_thinking=True,
        )
        try:
            content = [{"type": "input_text", "text": self.vlm_prompt}]
            for b64 in data_package["shap_images_b64"]:
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                })
            for b64 in data_package["substructure_image_b64"]:
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                })

            resp = self.client.responses.create(
                **self._base_kwargs(),
                input=[{"role": "user", "content": content}],
            )

            # Truncation is unrecoverable here: the JSON block is almost
            # certainly cut mid-structure, so fail with a clear hint.
            incomplete = self._incomplete_reason(resp)
            if incomplete == "max_output_tokens":
                if self.max_tokens is not None:
                    raise RuntimeError(
                        f"VLM response was truncated: this process is using "
                        f"max_tokens={self.max_tokens} (captured at startup). "
                        f"If you changed max_tokens or set it to 0 for no limit in "
                        f"the Agent settings, click New Chat to apply the new value. "
                        f"Otherwise, reduce the number of input images or raise the limit."
                    )
                raise RuntimeError(
                    "VLM response was truncated by the provider's token limit "
                    "(max_tokens is already disabled locally). "
                    "Reduce the number of input images."
                )
            if incomplete is not None:
                raise RuntimeError(
                    f"VLM response was incomplete ({incomplete})."
                )

            raw_output = self._extract_text(resp)
            if not raw_output.strip():
                diag = self._diagnose_empty(resp)
                message = f"VLM returned empty output ({diag})."
                self._emit(MSG_THINK, f"\n⚠️ {message}\n", only_if_thinking=True)
                raise RuntimeError(message)

            candidate = extract_json_object(raw_output)
            try:
                vlm_obs = json.loads(candidate)
            except json.JSONDecodeError as exc:
                snippet = raw_output[:1500]
                self._emit(
                    MSG_THINK,
                    f"\n⚠️ VLM output was not valid JSON. Raw output:\n{snippet}\n",
                    only_if_thinking=True,
                )
                raise RuntimeError(
                    f"VLM output is not valid JSON: {exc}. "
                    f"Raw output (first 500 chars): {raw_output[:500]}"
                ) from exc

            if not isinstance(vlm_obs, dict):
                raise RuntimeError(
                    f"VLM returned a JSON {type(vlm_obs).__name__}, "
                    f"expected an object."
                )
            # Guarantee downstream callers can index these keys even when
            # the model omitted one of them.
            vlm_obs.setdefault("substructure_observations", [])
            vlm_obs.setdefault("shap_dependence_observations", [])

            print(f"{DIVIDER} 🤓Observation completed. {DIVIDER}\n\n")
            self._emit(MSG_THINK, f"\n{raw_output}\n", only_if_thinking=True)
            self._emit(
                MSG_THINK, f"{DIVIDER} 🤓Observation completed. {DIVIDER}\n\n",
                only_if_thinking=True,
            )
            self._emit(MSG_PROGRESS, f"{2 / 5}")
            return vlm_obs
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            raise

    # ------------------------------------------------------------------
    # Stage 3: hierarchy check
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_sub_idx(name: str) -> str:
        """Normalize a free-form substructure label into 'krfp_<n>'."""
        if not isinstance(name, str):
            return ""
        s = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        s = re.sub(r"^krfp(?=\d)", "krfp_", s)
        s = re.sub(r"^krfp_+", "krfp_", s)
        m = re.search(r"krfp_\d+", s)
        return m.group(0) if m else s

    @staticmethod
    def _evidence_repr(sub_idx: str, contrib_label: str, weight: float) -> str:
        """Compact single-line rendering of one evidence row for the trace."""
        return (
            f'{{"substructure_idx": {sub_idx}, '
            f'"contrib_label": {contrib_label}, '
            f'"acin_weight": {weight}}}'
        )

    def hierarchy_checker(
        self,
        vlm_observations: List[Dict[str, str]],
        acin_weights: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Reconcile VLM contribution labels with the ACIN weight threshold.

        An observation is accepted when the label agrees with the ACIN
        weight relative to 'self.thresh':
        'high_contrib' requires 'weight >= thresh';
        'low_contrib' requires 'weight <= thresh'.
        """
        print(f"{DIVIDER} 🔎Checking hierarchy... {DIVIDER}")
        self._emit(
            MSG_THINK, f"{DIVIDER} 🔎Checking hierarchy... {DIVIDER}\n",
            only_if_thinking=True,
        )
        try:
            valid_evidence: List[Dict[str, Any]] = []
            columns = set(acin_weights.columns)

            smarts_list = KlekotaRothFingerprint().get_feature_names_out()

            for obs in vlm_observations:
                sub_idx = self.normalize_sub_idx(obs.get("substructure_idx", ""))
                contrib_label = obs.get("contrib", "")
                if sub_idx not in columns:
                    continue

                weight = acin_weights.iloc[0][sub_idx]
                passed = (
                    (contrib_label == CONTRIB_HIGH and weight >= self.thresh)
                    or (contrib_label == CONTRIB_LOW and weight <= self.thresh)
                )

                self._emit(
                    MSG_THINK,
                    self._evidence_repr(sub_idx, contrib_label, weight)
                    + (" Passed✅;\n" if passed else " Not Passed❌;\n"),
                    only_if_thinking=True,
                )

                if passed:
                    try:
                        krfp_idx = int(sub_idx.split("_")[-1])
                        smarts_pattern = smarts_list[krfp_idx]
                    except Exception:
                        smarts_pattern = "Unknown"

                    valid_evidence.append({
                        "substructure_idx": sub_idx,
                        "contrib_label": contrib_label,
                        "acin_weight": weight,
                        "smarts_pattern": smarts_pattern,
                    })

            print(f"{DIVIDER} 👌Hierarchy check completed. {DIVIDER}\n")
            self._emit(
                MSG_THINK,
                f"{DIVIDER} 👌Hierarchy check completed. {DIVIDER}\n\n",
                only_if_thinking=True,
            )
            self._emit(MSG_PROGRESS, f"{3 / 5}")
            return valid_evidence
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            raise

    # ------------------------------------------------------------------
    # Stage 4: chain-of-thought
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cot_output(raw_output: str) -> Dict[str, Any]:
        """Split a CoT reply into 'human_readable_summary' + JSON dict."""
        if SEP_JSON in raw_output:
            markdown_part, json_part = raw_output.split(SEP_JSON, 1)
        else:
            markdown_part = raw_output
            json_part = extract_json_object(raw_output)

        cleaned = clean_llm_json(json_part)
        result: Dict[str, Any] = {"raw_output": raw_output}
        if cleaned and cleaned != raw_output:
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        if markdown_part.strip():
            result["human_readable_summary"] = markdown_part.strip()
        return result

    def cot_reasoner(
        self,
        valid_evidence: List[Dict[str, Any]],
        shap_descriptor_text: str,
        shap_dependence_observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the CoT pass; streams the Markdown report before the JSON.

        If the provider truncates the reply ('incomplete_details.reason ==
        "max_output_tokens"'), a warning is emitted through the GUI queue
        by '_stream_response'; the partial Markdown still lands in
        'human_readable_summary' and the JSON block may be 'None'.
        """
        resp = None
        try:
            dictionary_str = json.dumps(self.DESCRIPTOR_DICTIONARY, indent=2, ensure_ascii=False)

            system_prompt = self.cot_prompt.format(
                valid_evidence=json.dumps(valid_evidence, ensure_ascii=False),
                shap_descriptor_text=shap_descriptor_text,
                descriptor_dictionary=dictionary_str,
                shap_dependence_observations=json.dumps(
                    shap_dependence_observations, ensure_ascii=False
                ),
            )
            self.cot_history_msg.append({"role": "user", "content": system_prompt})

            resp = self.client.responses.create(
                **self._base_kwargs(),
                input=self.cot_history_msg,
                stream=True,
            )
            _, raw_output, stopped = self._stream_response(
                resp, split_json=True, progress_answer=4 / 5,
            )
            self._finalize_history(raw_output, stopped)
            return self._parse_cot_output(raw_output)
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            if resp is not None:
                self._safe_close(resp)
            raise
        finally:
            # '_Init' enables the multi-turn chat input; skip it on stop
            # because the transcript may be truncated.
            if not self._stopped():
                self._emit(MSG_INIT)
            self._emit(MSG_PROGRESS, f"{5 / 5}")

    # ------------------------------------------------------------------
    # Stage 5: multi-turn conversation
    # ------------------------------------------------------------------

    def multi_turn_conv(self, text: str) -> None:
        """Continue the conversation after the initial analysis."""
        if not self.cot_history_msg:
            print("Please complete the initial analysis first...")
            self._emit(MSG_END)
            return
        if not text.strip():
            self._emit(MSG_END)
            return

        self.cot_history_msg.append({
            "role": "user",
            "content": text + "\nPlease answer in natural language and do not output JSON.",
        })

        resp = None
        try:
            resp = self.client.responses.create(
                **self._base_kwargs(),
                instructions=(
                    "You are a data-driven materials scientist working on Co-SAM "
                    "buried interfaces. Answer in natural language, maintain your "
                    "expert persona, and base your answers on the previous analysis "
                    "context."
                ),
                input=self.cot_history_msg,
                stream=True,
            )
            _, raw_output, stopped = self._stream_response(
                resp,
                split_json=False,
                progress_thinking=1 / 3,
                progress_answer=2 / 3,
            )
            self._finalize_history(raw_output, stopped)
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            if resp is not None:
                self._safe_close(resp)
            raise
        finally:
            self._emit(MSG_PROGRESS, f"{3 / 3}")

    # ------------------------------------------------------------------
    # Top-level pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        shap_beeswarm_img_path: str,
        shap_dependency_img_path: str,
        substructure_img_path: str,
        df_shap_descriptor_vec: pd.DataFrame,
        acin_substructure_weight: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run the full four-stage analysis and return a structured result."""
        try:
            package = self.data_prep(
                shap_beeswarm_img_path=shap_beeswarm_img_path,
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
                "vlm_raw_observation": vlm_obs,
                "valid_evidence_after_hierarchy_checker": valid_evidence,
                "analysis_result": final_reply,
            }
        except Exception as e:
            traceback.print_exc()
            self.send_error(self.q, e)
            raise