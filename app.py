# app.py
import os
import json
import re
import time
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient
import joblib
import pandas as pd
import logging
import numpy as np
from sklearn.inspection import permutation_importance

# optional SHAP
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# === Config & thresholds ===
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
MAX_ITEMS_PER_YEAR = int(os.getenv("MAX_ITEMS_PER_YEAR", "12"))

# Make Major/Moderate thresholds configurable via .env
MAJOR_PCT = float(os.getenv("MAJOR_PCT", "35.0"))    # >= this pct -> Major
MODERATE_PCT = float(os.getenv("MODERATE_PCT", "12.0"))  # >= this pct -> Moderate

# --- Logging ---
logger = logging.getLogger("child_mortality")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# --- HF Client (best-effort) ---
client = None
if HF_API_KEY:
    try:
        client = InferenceClient(api_key=HF_API_KEY, timeout=120)
        logger.info("Initialized Hugging Face InferenceClient.")
    except Exception as e:
        logger.exception("Failed HF init: %s", e)
else:
    logger.warning("HF API key missing — fallback mode only")

# --- Flask app + model loader ---
app = Flask(__name__, static_folder="static", static_url_path="/static")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            return m["pipeline"] if isinstance(m, dict) and "pipeline" in m else m
        except Exception as e:
            logger.exception("Model load failed: %s", e)
    else:
        logger.warning("Model file not found at %s", MODEL_PATH)
    return None

model = load_model()

# === population stats for heuristics ===
_POP_STATS = {}
try:
    sample_df = pd.read_csv(os.path.join("models", "sample_input.csv"))
    _POP_STATS["median"] = sample_df.median(numeric_only=True).to_dict()
    _POP_STATS["std"] = sample_df.std(numeric_only=True).replace(0, 1).to_dict()
except Exception:
    _POP_STATS["median"] = {}
    _POP_STATS["std"] = {}

# === Input validation ===
def validate_and_build_df(data: dict):
    required = ["birth_weight", "maternal_age", "immunized", "nutrition", "socioeconomic", "prenatal_visits"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    try:
        return pd.DataFrame([{
            "birth_weight": float(data["birth_weight"]),
            "maternal_age": float(data["maternal_age"]),
            "immunized": int(data["immunized"]),
            "nutrition": float(data["nutrition"]),
            "socioeconomic": int(data["socioeconomic"]),
            "prenatal_visits": float(data["prenatal_visits"])
        }])
    except Exception as e:
        raise ValueError(f"Invalid data format: {e}")

# === Fallback plan (unchanged) ===
def generate_fallback_plan():
    return {
        "risk_level": "high",
        "years": {
            "Year 0-1": ["Doctor visits", "Vaccinations", "Monitor growth and nutrition"],
            "Year 1-2": ["Regular checkups", "Balanced diet", "Monitor development milestones"],
            "Year 2-3": ["Speech and motor skill support", "Vaccinations update", "Nutritional supplements"],
            "Year 3-4": ["School readiness assessment", "Preventive health checks", "Encourage physical activity"],
            "Year 4-5": ["Annual pediatric screening", "Vaccinations", "Healthy lifestyle counseling"]
        },
        "warning": "Used fallback plan due to HF API failure or parsing error."
    }

EXPECTED_KEYS = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"]

# === Parser helpers (kept robust from earlier) ===
def _normalize_year_key(key_text):
    nums = re.findall(r"\d+", key_text)
    if len(nums) >= 2:
        return f"Year {nums[0]}-{nums[1]}"
    return None

def _map_json_to_expected(plan_json):
    mapped = {}
    for ek in EXPECTED_KEYS:
        if ek in plan_json:
            mapped[ek] = plan_json[ek]
    for k, v in plan_json.items():
        nk = _normalize_year_key(k)
        if nk and nk in EXPECTED_KEYS and nk not in mapped:
            mapped[nk] = v
    return mapped

def _parse_plan_from_text(raw_text):
    default_steps = list(generate_fallback_plan()["years"].values())
    if not raw_text or not isinstance(raw_text, str):
        return {k: default_steps[i] for i, k in enumerate(EXPECTED_KEYS)}
    text = raw_text.strip()
    start_idx = text.find("{")
    json_candidate = None
    if start_idx != -1:
        depth = 0
        for i in range(start_idx, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_candidate = text[start_idx:i+1]
                    break

    def try_load_and_map(s):
        try:
            parsed = json.loads(s)
            mapped = _map_json_to_expected(parsed)
            cleaned = {}
            for i, ek in enumerate(EXPECTED_KEYS):
                v = mapped.get(ek)
                if isinstance(v, list) and len(v) > 0:
                    cleaned[ek] = [str(x).strip() for x in v][:MAX_ITEMS_PER_YEAR]
                else:
                    cleaned[ek] = default_steps[i]
            return cleaned
        except Exception:
            return None

    if json_candidate:
        out = try_load_and_map(json_candidate)
        if out:
            logger.debug("Parser: used balanced JSON candidate.")
            return out
        repaired = re.sub(r",(\s*[\]\}])", r"\1", json_candidate)
        out = try_load_and_map(repaired)
        if out:
            logger.debug("Parser: repaired trailing commas and parsed JSON.")
            return out
        cleaned_candidate = "".join(ch for ch in repaired if (31 < ord(ch) < 127) or ch in "\n\r\t")
        out = try_load_and_map(cleaned_candidate)
        if out:
            logger.debug("Parser: removed non-ASCII and parsed JSON.")
            return out
        if "'" in json_candidate and '"' not in json_candidate:
            alt = json_candidate.replace("'", '"')
            alt = re.sub(r",(\s*[\]\}])", r"\1", alt)
            out = try_load_and_map(alt)
            if out:
                logger.debug("Parser: replaced single quotes and parsed JSON.")
                return out
        logger.debug("Parser: found JSON-like block but could not parse/repair it.")

    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        joined = "\n".join(lines)
        heading_re = re.compile(r"(Year\s*\d+\s*(?:-|to|–)\s*\d+)", flags=re.I)
        headings = [(m.group(1).strip(), m.start()) for m in heading_re.finditer(joined)]
        if headings:
            sections = {}
            positions = [pos for (_, pos) in headings] + [len(joined)]
            keys = [h for (h, _) in headings]
            for idx, key in enumerate(keys):
                start = positions[idx]
                end = positions[idx + 1]
                seg = joined[start:end].strip()
                seg_lines = seg.splitlines()
                body = "\n".join(seg_lines[1:]) if len(seg_lines) > 1 else ""
                items = re.split(r"[\n\r]+|[•\-\*\u2022]+|[;,\|]\s*", body)
                items = [it.strip() for it in items if it and len(it.strip()) > 2]
                safe_items = []
                for it in items:
                    if len(safe_items) >= MAX_ITEMS_PER_YEAR:
                        break
                    if len(it) > 160:
                        continue
                    ascii_ratio = sum(1 for ch in it if ord(ch) < 128) / max(1, len(it))
                    if ascii_ratio < 0.55 and len(it.split()) > 6:
                        continue
                    safe_items.append(it)
                sections[key] = safe_items
            cleaned = {}
            for i, ek in enumerate(EXPECTED_KEYS):
                ek_nums = re.findall(r"\d+", ek)
                found = None
                for pk, val in sections.items():
                    if re.findall(r"\d+", pk) == ek_nums and val:
                        found = val
                        break
                cleaned[ek] = found[:MAX_ITEMS_PER_YEAR] if found else default_steps[i]
            logger.debug("Parser: used headings heuristic with trimming.")
            return cleaned
    except Exception as e:
        logger.debug("Parser heuristic failed: %s", e)

    logger.debug("Parser: returning default fallback plan.")
    return {k: default_steps[i] for i, k in enumerate(EXPECTED_KEYS)}

# =============================================================
# HF caller — uses exact system prompt pattern (kept simple here)
# =============================================================
USER_SYSTEM_PROMPT = (
    "You are a pediatric health assistant and clinical decision-support assistant. \n"
    "Your role is to generate HIGH-QUALITY, EVIDENCE-INFORMED guidance for reducing infant/child mortality risk.\n\n"
    "IMPORTANT RULES — FOLLOW STRICTLY:\n"
    "1. OUTPUT **ONLY ONE JSON OBJECT** and NOTHING ELSE.\n"
    "2. The JSON MUST contain exactly these keys:\n"
    "   \"Year 0-1\", \"Year 1-2\", \"Year 2-3\", \"Year 3-4\", \"Year 4-5\".\n"
    "3. Each key MUST map to an array (list) of SHORT, ACTIONABLE, CLINICALLY SAFE steps.\n"
    "4. Provide **5 to 12 steps per year**. Always prefer medically relevant, risk-reducing, practical actions.\n"
    "5. PERSONALIZE the guidance based on the child’s features (birth weight, maternal age, immunization status, nutrition score, socioeconomic status, prenatal visits, etc.).\n"
    "6. GUIDANCE MUST BE RISK-FOCUSED:\n"
    "      - Emphasize infection prevention,\n"
    "      - Growth monitoring,\n"
    "      - Nutrition optimization,\n"
    "      - Early detection of developmental delays,\n"
    "      - Vaccination details,\n"
    "      - Hospital referral red flags,\n"
    "      - Safety and hygiene,\n"
    "      - Maternal health and family counseling.\n"
    "7. DO NOT hallucinate medicines, diagnoses, or treatments. \n"
    "   Provide **general personalized preventive and supportive recommendations based on the child’s features (birth weight, maternal age, immunization status, nutrition score, socioeconomic status, prenatal visits, etc.) only**.\n"
    "8. DO NOT include disclaimers, explanations, reasoning, or text outside the JSON.\n"
    "9. DO NOT wrap JSON in code blocks. Output MUST start with \"{\" and end with \"}\" with no additional content.\n\n"
    "Your task:\n"
    "Given the child's risk factors, generate a **comprehensive, personalized survival & care plan** designed to REDUCE MORTALITY for HIGH-RISK infants/children through age 5.\n\n"
    "Example JSON TEMPLATE (structure only; fill with your own medically-safe steps):\n"
    "{\n"
    "  \"Year 0-1\": [\"step1\", \"step2\", ...],\n"
    "  \"Year 1-2\": [\"step1\", \"step2\", ...],\n"
    "  \"Year 2-3\": [\"step1\", \"step2\", ...],\n"
    "  \"Year 3-4\": [\"step1\", \"step2\", ...],\n"
    "  \"Year 4-5\": [\"step1\", \"step2\", ...]\n"
    "}\n\n"
    "If you cannot produce the required JSON, output an empty JSON object: {}\n"
)

def _generate_survival_plan_hf_uncached(features, model_name=HF_MODEL, max_retries=1):
    if client is None:
        logger.warning("HF client not initialized; returning fallback.")
        return generate_fallback_plan(), {"hf_raw": None, "error": "no_client"}

    baby_info = ", ".join([f"{k}: {v}" for k, v in features.items()])
    messages = [
        {"role": "system", "content": USER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Child features: {baby_info}. Provide JSON as described."}
    ]
    last_raw = None
    for attempt in range(1, max_retries + 2):
        try:
            logger.debug("Calling HF model=%s attempt=%d", model_name, attempt)
            start = time.time()
            resp = client.chat_completion(model=model_name, messages=messages, max_tokens=1200, temperature=0.0)
            elapsed = time.time() - start
            try:
                raw_text = resp.choices[0].message["content"]
            except Exception:
                raw_text = str(resp)
            last_raw = (raw_text or "")[:32000]
            logger.debug("HF responded in %.2fs; raw-trunc=%s", elapsed, last_raw[:400].replace("\n", " "))
            parsed = _parse_plan_from_text(raw_text)
            if parsed:
                for k in EXPECTED_KEYS:
                    lst = parsed.get(k, [])
                    if len(lst) < 5:
                        fallback = generate_fallback_plan()["years"][k]
                        to_add = []
                        for item in fallback:
                            if item not in lst:
                                to_add.append(item)
                            if len(lst) + len(to_add) >= 5:
                                break
                        parsed[k] = lst + to_add
                    parsed[k] = parsed[k][:MAX_ITEMS_PER_YEAR]
                return {"risk_level": "high", "years": parsed}, {"hf_raw": last_raw}
            else:
                return generate_fallback_plan(), {"hf_raw": last_raw, "error": "parsed_empty"}
        except Exception as exc:
            logger.exception("HF call attempt %d failed: %s", attempt, exc)
            last_raw = (str(exc) + "\n" + (last_raw or ""))[:32000]
            if attempt <= max_retries:
                time.sleep(2 ** attempt)
                continue
            return generate_fallback_plan(), {"hf_raw": last_raw, "error": str(exc)}
    return generate_fallback_plan(), {"hf_raw": last_raw, "error": "max_retries_exceeded"}

# --- Caching wrapper + public generator for HF survival plan ---
@lru_cache(maxsize=256)
def _cached_generate_plan_tuple(features_tuple):
    """
    Cache key is a tuple of the six numeric features in a deterministic order.
    We store the JSON string of (plan, debug) to keep it JSON-serializable.
    """
    features = {
        "birth_weight": features_tuple[0],
        "maternal_age": features_tuple[1],
        "immunized": features_tuple[2],
        "nutrition": features_tuple[3],
        "socioeconomic": features_tuple[4],
        "prenatal_visits": features_tuple[5]
    }
    plan, debug = _generate_survival_plan_hf_uncached(features)
    # return a JSON-serializable string
    return json.dumps((plan, debug))

def generate_survival_plan_hf(features):
    """
    Public wrapper: deterministic tuple -> cached call -> returns (plan, debug)
    Falls back to uncached HF call if cache decode fails.
    """
    tup = (
        float(features.get("birth_weight", 0.0)),
        float(features.get("maternal_age", 0.0)),
        int(features.get("immunized", 0)),
        float(features.get("nutrition", 0.0)),
        int(features.get("socioeconomic", 0)),
        float(features.get("prenatal_visits", 0.0)),
    )
    try:
        cached = _cached_generate_plan_tuple(tup)
        plan, debug = json.loads(cached)
        return plan, debug
    except Exception:
        # If anything goes wrong with cache decoding, call the uncached function
        try:
            return _generate_survival_plan_hf_uncached({
                "birth_weight": tup[0],
                "maternal_age": tup[1],
                "immunized": tup[2],
                "nutrition": tup[3],
                "socioeconomic": tup[4],
                "prenatal_visits": tup[5]
            })
        except Exception as exc:
            logger.exception("generate_survival_plan_hf fallback failed: %s", exc)
            return generate_fallback_plan(), {"hf_raw": None, "error": "uncached_call_failed"}

# === Explanation function (REPLACE / improved) ===
def explain_local_prediction(input_df):
    """
    Improved explanation generator:
    - Tries SHAP -> linear coef -> permutation -> heuristic
    - Returns raw 'feature_contributions' and human-friendly 'feature_importance_human'
    - Builds ui_html for quick display and keeps technical JSON for clinicians
    """
    if model is None:
        return {"error": "no_local_model"}

    X = input_df.copy()
    feature_names = list(X.columns)
    standardized_feats = []
    method = "unknown"

    # 1) SHAP (best-effort)
    try:
        if _HAS_SHAP:
            core_model = model
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                core_model = model.named_steps["model"]
            expl = None
            try:
                if hasattr(core_model, "feature_importances_") or core_model.__class__.__name__.lower().startswith(("lgbm","xgboost","catboost","randomforest")):
                    expl = shap.TreeExplainer(core_model)
                else:
                    # KernelExplainer may be slow; use input X as background for KernelExplainer
                    expl = shap.KernelExplainer(core_model.predict_proba, X)
            except Exception:
                expl = None
            if expl is not None:
                shap_vals = expl.shap_values(X)
                if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                    contrib = np.array(shap_vals[1])[0]
                else:
                    contrib = np.array(shap_vals)[0]
                for fname, val, c in zip(feature_names, X.iloc[0].tolist(), contrib.tolist()):
                    standardized_feats.append({"feature": fname, "value": float(val), "contribution": float(c)})
                method = "shap"
    except Exception as e:
        logger.debug("SHAP failed: %s", e)

    # 2) Linear coefficients (for linear models)
    if not standardized_feats:
        try:
            core = model
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                core = model.named_steps["model"]
            if hasattr(core, "coef_"):
                try:
                    X_trans = X.copy()
                    if hasattr(model, "named_steps"):
                        if "preproc" in model.named_steps:
                            X_trans = pd.DataFrame(model.named_steps["preproc"].transform(X_trans), columns=X_trans.columns)
                except Exception:
                    X_trans = X.copy()
                coefs = core.coef_.ravel()
                for fname, val, coef in zip(feature_names, X.iloc[0].tolist(), coefs.tolist()):
                    contribution = float(coef * float(val))
                    standardized_feats.append({"feature": fname, "value": float(val), "contribution": contribution})
                method = "linear_coef"
        except Exception as e:
            logger.debug("Linear coef explanation failed: %s", e)

    # 3) Permutation importance
    if not standardized_feats:
        try:
            try:
                ref = pd.read_csv(os.path.join("models", "sample_input.csv"))
                ref = ref[list(X.columns)]
                ref_y = np.zeros(len(ref))
            except Exception:
                ref = pd.concat([X] * 12, ignore_index=True)
                ref_y = np.zeros(len(ref))
            r = permutation_importance(model, ref, ref_y, n_repeats=6, random_state=42, n_jobs=-1)
            importances = r.importances_mean
            for fname, val, imp in zip(X.columns, X.iloc[0].tolist(), importances.tolist()):
                standardized_feats.append({"feature": fname, "value": float(val), "contribution": float(imp)})
            method = "permutation"
        except Exception as e:
            logger.debug("Permutation importance failed: %s", e)

    # 4) Heuristic fallback
    if not standardized_feats:
        try:
            core = model
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                core = model.named_steps["model"]
            imp_scores = None
            if hasattr(core, "feature_importances_"):
                imp_scores = np.array(core.feature_importances_)
                if imp_scores.sum() != 0:
                    imp_scores = imp_scores / (imp_scores.sum())
            for i, (fname, val) in enumerate(zip(X.columns, X.iloc[0].tolist())):
                median = _POP_STATS.get("median", {}).get(fname, 0.0)
                std = _POP_STATS.get("std", {}).get(fname, 1.0)
                z = (float(val) - float(median)) / float(std) if std != 0 else 0.0
                weight = 1.0
                if imp_scores is not None:
                    try:
                        weight = float(imp_scores[i])
                    except Exception:
                        weight = 1.0
                contrib = float(z * weight)
                standardized_feats.append({"feature": fname, "value": float(val), "contribution": contrib})
            method = "heuristic"
        except Exception as e:
            logger.debug("Heuristic failed: %s", e)

    if not standardized_feats:
        return {"error": "explanation_failed"}

    # Normalize & sort by absolute effect
    standardized_feats = sorted(standardized_feats, key=lambda x: abs(x.get("contribution", 0.0)), reverse=True)

    # Friendly labels & default recs
    FRIENDLY = {
        "birth_weight": ("Birth weight", "Provide feeding support and frequent weight checks; monitor growth."),
        "maternal_age": ("Maternal age", "Arrange maternal follow-up and counseling as needed."),
        "immunized": ("Immunization status", "Keep vaccinations on schedule and follow local immunization program."),
        "nutrition": ("Nutrition score", "Offer breastfeeding support and nutrient-rich complementary foods."),
        "socioeconomic": ("Socioeconomic status", "Link family to community resources and social support."),
        "prenatal_visits": ("Prenatal visits", "Arrange antenatal follow-up and skilled birth attendance.")
    }

    # Compute relative importance (%) and magnitude label using configurable thresholds
    total_abs = sum(abs(float(x.get("contribution", 0.0))) for x in standardized_feats) or 1.0
    humanified = []
    top_n = min(6, len(standardized_feats))
    top_feats = standardized_feats[:top_n]
    bullets = []
    recommendations = []
    positive_sum = 0.0

    for item in top_feats:
        feat = item["feature"]
        contrib = float(item.get("contribution", 0.0))
        val = item.get("value")
        rel = (abs(contrib) / total_abs) * 100.0
        if rel >= MAJOR_PCT:
            magnitude = "Major"
        elif rel >= MODERATE_PCT:
            magnitude = "Moderate"
        else:
            magnitude = "Minor"
        direction = "Increase" if contrib > 0.03 else ("Decrease" if contrib < -0.03 else "Neutral")
        label, rec_text = FRIENDLY.get(feat, (feat.replace("_", " ").capitalize(), "Follow routine clinical care and monitoring."))
        if direction == "Increase":
            bullets.append(f"{label}: below/absent or suboptimal — this likely raises risk.")
            recommendations.append({"feature": label, "recommendation": rec_text})
            positive_sum += abs(contrib)
        elif direction == "Decrease":
            bullets.append(f"{label}: appears protective and lowers risk.")
        else:
            bullets.append(f"{label}: limited impact on this prediction.")
        humanified.append({
            "feature": feat,
            "label": label,
            "value": float(val),
            "contribution": contrib,
            "relative_importance_pct": round(rel, 1),
            "magnitude_label": magnitude,
            "direction": direction,
            "recommendation": rec_text
        })

    top_labels = [h["label"] for h in humanified[:3]]
    summary = "Main drivers: " + ", ".join(top_labels) if top_labels else "No clear drivers identified."

    # Build ui_html
    def _arrow_word(c):
        if c > 0.03:
            return ("↑", "#ef4444", "Increases")
        elif c < -0.03:
            return ("↓", "#10b981", "Decreases")
        else:
            return ("→", "#6b7280", "Neutral")

    list_items_html = ""
    for it in humanified:
        arrow, color, verb = _arrow_word(it["contribution"])
        reason = ""
        if arrow == "↑":
            reason = f"{it['label']} is below ideal or missing — this likely raises risk."
        elif arrow == "↓":
            reason = f"{it['label']} appears protective and lowers risk."
        else:
            reason = f"{it['label']} has limited effect on the prediction."
        list_items_html += (
            "<li style='display:flex; justify-content:space-between; align-items:flex-start; padding:6px 0; border-bottom:1px solid rgba(0,0,0,0.04)'>"
            f"<div style='max-width:78%'><div style='font-weight:600'>{it['label']} <span aria-hidden='true' style='color:{color}; margin-left:6px'>{arrow}</span> "
            f"<small style='font-weight:600;margin-left:8px'>{it['magnitude_label']} ({it['relative_importance_pct']}%)</small></div>"
            f"<div style='font-size:13px; color:#374151; margin-top:4px'>{reason}</div>"
            f"<div style='margin-top:6px; font-size:13px'><strong>Action:</strong> {it['recommendation']}</div></div>"
            f"<div style='text-align:right; min-width:40px'><div style='font-size:12px; color:{color}; font-weight:700'>{verb}</div></div>"
            "</li>"
        )

    recommendations_html = "".join([f"<li>• <strong>{r['feature']}:</strong> {r['recommendation']}</li>" for r in recommendations[:4]])
    tech_json = json.dumps({"feature_contributions": standardized_feats}, indent=2)

    ui_html = (
        f"<div>"
        f"<div style='font-weight:600;margin-bottom:8px'>{summary}</div>"
        f"<ul style='list-style:none;padding-left:0;margin:0 0 8px 0'>{list_items_html}</ul>"
        f"<div style='margin-top:8px;font-size:13px'><div style='font-weight:600;margin-bottom:6px'>Suggested next steps</div>"
        f"<ul style='padding-left:1rem;margin:0'>{recommendations_html or '<li>No specific recommendations available.</li>'}</ul></div>"
        f"<div style='margin-top:10px;font-size:12px;color:#374151'><strong>Red flags:</strong> fever, difficulty breathing, poor feeding, lethargy — seek urgent care.</div>"
        f"<div aria-hidden='true' style='margin-top:8px;font-size:12px;color:#6b7280'><span style='margin-right:8px'><strong>Legend:</strong></span>"
        f"<span style='color:#ef4444;font-weight:700;margin-right:6px'>↑ increases risk</span>"
        f"<span style='color:#10b981;font-weight:700;margin-right:6px'>↓ lowers risk</span>"
        f"<span style='color:#6b7280;font-weight:700'>→ neutral</span></div>"
        f"<details style='margin-top:8px'><summary style='cursor:pointer;font-weight:600'>Show technical details</summary>"
        f"<pre style='white-space:pre-wrap;padding:8px;background:#f9fafb;border-radius:6px;margin-top:6px'>{tech_json}</pre></details>"
        f"</div>"
    )

    severity_score = int(min(100, max(0, positive_sum * 10)))

    explanation_payload = {
        "method": method,
        "feature_contributions": standardized_feats,
        "feature_importance_human": humanified,
        "summary": summary,
        "bullets": bullets,
        "recommendations": recommendations,
        "ui_html": ui_html,
        "severity_score": severity_score
    }

    # --- Log the final chosen method (single-line) ---
    logger.info("Explanation method used: %s", method)

    return explanation_payload

# === Survival plan selector ===
def survival_plan(prediction, features=None, debug=False):
    if int(prediction) == 0:
        return {"risk_level": "low", "message": "Low risk. Continue preventive care: vaccines, nutrition, safe environment."}, {"branch": "low"}
    else:
        if features:
            plan, debug_info = generate_survival_plan_hf(features)
            return plan, {"branch": "high", **(debug_info or {})}
        else:
            return generate_fallback_plan(), {"branch": "high", "hf_raw": None, "error": "no_features"}

# === Routes ===
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/admin/reload_model", methods=["POST"])
def reload_model():
    global model
    model = load_model()
    return jsonify({"reloaded": model is not None})

@app.route("/api/predict", methods=["POST"])
def predict_api():
    global model
    if model is None:
        return jsonify({"error": "Local model not found. Please train it or place models/model.pkl"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    q_debug = request.args.get("debug", "").lower() == "true"
    body_debug = bool(data.get("debug")) if isinstance(data, dict) else False
    debug_flag = q_debug or body_debug

    try:
        df = validate_and_build_df(data)
    except ValueError as e:
        return jsonify({"error": "Invalid input format", "details": str(e)}), 400

    try:
        prob = None
        try:
            prob = float(model.predict_proba(df)[:, 1][0])
        except Exception:
            prob = None
        pred = int(model.predict(df)[0])
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        return jsonify({"error": "Model inference failed", "details": str(e)}), 500

    features = df.iloc[0].to_dict()
    plan_obj, debug_info = survival_plan(pred, features, debug=debug_flag)

    response = {
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        "survival_plan": plan_obj,
        "debug": {"branch": debug_info.get("branch", "high" if pred == 1 else "low")}
    }

    if debug_flag:
        hf_raw = debug_info.get("hf_raw")
        if hf_raw:
            response["debug"]["hf_raw_truncated"] = hf_raw[:8000]
        if "error" in debug_info:
            response["debug"]["error"] = debug_info["error"]

        # attach explanation (attempt)
        try:
            explanation = explain_local_prediction(df)
            response["debug"]["explanation"] = explanation
            response["debug"]["explanation_summary"] = explanation.get("summary") if isinstance(explanation, dict) else None
        except Exception as ex:
            logger.exception("Explanation generation failed: %s", ex)
            response["debug"]["explanation_error"] = str(ex)

    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict_alias():
    return predict_api()

# === Run server ===
if __name__ == "__main__":
    print("🚀 Starting Child Mortality API server on http://127.0.0.1:8000 ")
    app.run(host="0.0.0.0", port=8000, debug=False)
