# app.py
import os
import json
import re
import time
import uuid
import sqlite3
import traceback
from functools import lru_cache
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, session as flask_session
from huggingface_hub import InferenceClient
import joblib
import pandas as pd
import logging
import numpy as np
from sklearn.inspection import permutation_importance
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

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
MAX_ITEMS_PER_YEAR = int(os.getenv("MAX_ITEMS_PER_YEAR", "8"))

MAJOR_PCT = float(os.getenv("MAJOR_PCT", "35.0"))
MODERATE_PCT = float(os.getenv("MODERATE_PCT", "12.0"))

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-please-change")
# important session cookie settings for dev:
# - SameSite Lax helps cookies be sent on top-level navigations (safe default).
# - For cross-origin credentialed fetch you will still need CORS(app, supports_credentials=True)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
# In production set to True if you serve under HTTPS
app.config["SESSION_COOKIE_SECURE"] = False

# Allow cross-origin requests that use cookies (session-based auth)
CORS(app, supports_credentials=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            # If it's a pipeline/dict, extract logic (though new model is just CatBoost object)
            return m["pipeline"] if isinstance(m, dict) and "pipeline" in m else m
        except Exception as e:
            logger.exception("Model load failed: %s", e)
    else:
        logger.warning("Model file not found at %s", MODEL_PATH)
    return None

# Load model
model = load_model()

# Load specific feature order if available
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "models", "features.pkl")
MODEL_FEATURES = None
if os.path.exists(FEATURES_PATH):
    try:
        MODEL_FEATURES = joblib.load(FEATURES_PATH)
        logger.info("Loaded feature list: %s", MODEL_FEATURES[:5] + ["..."])
    except Exception as e:
        logger.warning("Failed to load features.pkl: %s", e)

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
        df = pd.DataFrame([{
            "birth_weight": float(data["birth_weight"]),
            "maternal_age": float(data["maternal_age"]),
            "immunized": int(data["immunized"]),
            "nutrition": float(data["nutrition"]),
            "socioeconomic": int(data["socioeconomic"]),
            "prenatal_visits": float(data["prenatal_visits"])
        }])
        
        # Feature Engineering for Model V2
        # (It's safe to add these columns even for V1 if V1 ignores them or uses a pipeline that selects cols)
        # However, to be ultra-safe, we should check which model is loaded. 
        # But commonly, extra columns are just ignored by sklearn models if not in the signature, 
        # UNLESS it's a pipeline with strict column checks.
        # CatBoost (V2) definitely needs them.
        
        df["risk_low_bw"] = (df["birth_weight"] < 2.5).astype(int)
        df["risk_very_low_bw"] = (df["birth_weight"] < 1.5).astype(int)
        df["risk_teen_mom"] = (df["maternal_age"] < 18).astype(int)
        df["risk_advanced_mom"] = (df["maternal_age"] > 35).astype(int)
        df["risk_no_visits"] = (df["prenatal_visits"] < 3).astype(int)
        df["risk_poor_nutrition"] = (df["nutrition"] < 40).astype(int)
        
        # Interaction
        df["socio_nut_risk"] = ((df["socioeconomic"] == 0) & (df["nutrition"] < 50)).astype(int)
        
        # Cumulative
        df["cumulative_risk_flags"] = (
            df["risk_low_bw"] + 
            df["risk_teen_mom"] + 
            df["risk_no_visits"] + 
            (1 - df["immunized"]) * 2
        )
        
        # Enforce column order if known
        if MODEL_FEATURES:
            # Add missing columns if any (as 0) - safety net
            for col in MODEL_FEATURES:
                if col not in df.columns:
                    df[col] = 0
            # Select and reorder
            df = df[MODEL_FEATURES]
        
        return df
    except Exception as e:
        raise ValueError(f"Invalid data format: {e}")

# === Fallback plan ===
def generate_fallback_plan():
    return {
        "risk_level": "high",
        "years": {
            "Year 0-1": [
                "Monitor vital signs daily (temperature, respiratory rate, heart rate)",
                "Ensure exclusive breastfeeding or formula feeding with proper hygiene",
                "Complete vaccination schedule on time (BCG, Polio, Hepatitis B, DPT)",
                "Watch for danger signs: fever, difficulty breathing, poor feeding, lethargy",
                "Maintain clean umbilical cord care and safe sleeping environment",
                "Regular weight and length monitoring at health clinic",
                "Prevent infections through handwashing and sanitization",
                "Seek immediate care for signs of jaundice, seizures, or severe illness"
            ],
            "Year 1-2": [
                "Continue immunization schedule (DPT booster, Polio, measles)",
                "Introduce appropriate complementary foods at 6 months",
                "Monitor developmental milestones (sitting, babbling, grasping)",
                "Watch for signs of malnutrition or growth faltering",
                "Ensure safe play environment free from hazards and choking risks",
                "Regular health checkups every 3 months",
                "Treat diarrhea and infections promptly with oral rehydration",
                "Provide iron-rich foods and vitamin supplements as recommended"
            ],
            "Year 2-3": [
                "Complete primary immunization series with all boosters",
                "Monitor language development and encourage verbal communication",
                "Provide balanced diet with proteins, vegetables, fruits, and grains",
                "Watch for signs of developmental delay or behavioral concerns",
                "Ensure safe play with age-appropriate toys and supervision",
                "Regular dental check-ups and oral hygiene practices",
                "Prevent accidents through childproofing and constant supervision",
                "Screen for vision and hearing problems through clinical assessment"
            ],
            "Year 3-4": [
                "Ensure all routine vaccinations are up-to-date before school entry",
                "Assess cognitive and motor skill development regularly",
                "Provide nutritious meals and limit sugary snacks and drinks",
                "Encourage physical activity and outdoor play daily",
                "Monitor emotional and social development with peers",
                "Screen for common childhood illnesses (asthma, allergies, anemia)",
                "Teach basic hygiene, handwashing, and disease prevention practices",
                "Arrange vision and hearing screening before school enrollment"
            ],
            "Year 4-5": [
                "Conduct comprehensive pre-school medical examination",
                "Ensure all vaccinations including DPT booster and Polio are completed",
                "Assess growth parameters and nutritional status",
                "Evaluate motor skills, coordination, and physical fitness",
                "Screen for behavioral and emotional problems or learning difficulties",
                "Provide counseling on healthy lifestyle, nutrition, and hygiene",
                "Update dental and vision care; address any identified issues",
                "Discuss school readiness and prepare child for educational transition"
            ]
        },
        "warning": "Used fallback plan due to HF API failure or parsing error. Please consult with healthcare provider for personalized medical advice."
    }

EXPECTED_KEYS = ["Year 0-1", "Year 1-2", "Year 2-3", "Year 3-4", "Year 4-5"]

# === Parser helpers ===
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
                    # Clean each item: strip whitespace and quotes, filter short items and duplicates
                    cleaned_items = []
                    seen = set()
                    for item in v:
                        it = ""
                        if isinstance(item, dict):
                            # Fallback: extract first string-like value if HF returns objects
                            vals = [str(x) for x in item.values() if isinstance(x, str)]
                            if vals:
                                it = vals[0]
                            else:
                                it = str(item)
                        else:
                            it = str(item)
                        
                        it = it.strip().strip('"\'')  # Remove surrounding quotes
                        # Skip incomplete sentences ending with space + dot
                        if it.rstrip().endswith((' .', ' ..', '...')):
                            it = it.rstrip().rstrip('. ')
                        # Skip if too short (less than 10 chars or 3 words = not valuable)
                        if not it or len(it) < 10 or len(it.split()) < 3:
                            continue
                        # Skip if too long (likely malformed)
                        if len(it) > 160:
                            continue
                        # Skip duplicates
                        it_lower = it.lower()
                        if it_lower in seen:
                            continue
                        seen.add(it_lower)
                        cleaned_items.append(it)
                    # Use cleaned items if available, otherwise fallback
                    if cleaned_items:
                        cleaned[ek] = cleaned_items[:MAX_ITEMS_PER_YEAR]
                    else:
                        cleaned[ek] = default_steps[i]
                else:
                    cleaned[ek] = default_steps[i]
            
            # Cross-year deduplication: remove items that appear in multiple years
            all_seen_lower = {}
            for ek in EXPECTED_KEYS:
                items_list = cleaned[ek]
                unique_items = []
                for it in items_list:
                    it_lower = it.lower()
                    if it_lower not in all_seen_lower:
                        unique_items.append(it)
                        all_seen_lower[it_lower] = ek
                cleaned[ek] = unique_items
            
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
                # Filter and clean items
                safe_items = []
                seen = set()
                for it in items:
                    it = it.strip().strip('"\'')  # Remove surrounding quotes
                    # Skip empty or items with only whitespace/quotes
                    if not it or it in ['""', "''", '"-"', "'-'"]:
                        continue
                    # Skip incomplete sentences (ending with space + dot or only dots)
                    if it.rstrip().endswith((' .', ' ..', '...')):
                        it = it.rstrip().rstrip('. ')
                    if not it or len(it) < 10:  # Ensure meaningful content (10+ chars)
                        continue
                    # Skip if too short (less than 3 words = less than valuable)
                    if len(it.split()) < 3:
                        continue
                    # Skip if too long (likely malformed)
                    if len(it) > 160:
                        continue
                    # Skip non-ASCII content
                    ascii_ratio = sum(1 for ch in it if ord(ch) < 128) / max(1, len(it))
                    if ascii_ratio < 0.55 and len(it.split()) > 6:
                        continue
                    # Skip duplicates (case-insensitive)
                    it_lower = it.lower()
                    if it_lower in seen:
                        continue
                    # Enforce max items per year
                    if len(safe_items) >= MAX_ITEMS_PER_YEAR:
                        break
                    seen.add(it_lower)
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
                if found and len(found) > 0:
                    # Ensure at least 8 items, otherwise use default
                    if len(found) >= 5:
                        cleaned[ek] = found[:MAX_ITEMS_PER_YEAR]
                    else:
                        # Too few items parsed, use fallback
                        cleaned[ek] = default_steps[i]
                else:
                    cleaned[ek] = default_steps[i]
            
            # Cross-year deduplication: remove similar items that appear in multiple years
            all_seen_lower = {}  # track which year each item was added
            for ek in EXPECTED_KEYS:
                items_list = cleaned[ek]
                unique_items = []
                for it in items_list:
                    it_lower = it.lower()
                    if it_lower not in all_seen_lower:
                        unique_items.append(it)
                        all_seen_lower[it_lower] = ek
                    # else: item already in previous year, skip it
                cleaned[ek] = unique_items
            
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
    "3. Each key MUST map to an array (list) of STRINGS. Do NOT use objects/dicts inside the list.\n"
    "4. Provide **6 to 8 steps per year**. Always prefer medically relevant, risk-reducing, practical actions.\n"
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
    "If you cannot produce the required JSON, output an empty JSON object: {}\n"
)

def _generate_survival_plan_hf_uncached(features, model_name=HF_MODEL, max_retries=1):
    if client is None:
        logger.warning("HF client not initialized; returning fallback.")
        return generate_fallback_plan(), {"hf_raw": None, "error": "no_client"}

    # Build detailed feature description for better HF personalization
    feature_details = [
        f"Birth Weight: {features.get('birth_weight', 0.0)} kg",
        f"Maternal Age: {features.get('maternal_age', 0.0)} years",
        f"Immunization Status: {'Fully immunized' if features.get('immunized', 0) == 1 else 'Not fully immunized'}",
        f"Nutrition Score: {features.get('nutrition', 0.0)}/100",
        f"Socioeconomic Status: {features.get('socioeconomic', 0)} (0=low, 1=medium, 2=high)",
        f"Prenatal Visits: {features.get('prenatal_visits', 0.0)} visits"
    ]
    baby_info = "\n".join(feature_details)
    messages = [
        {"role": "system", "content": USER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Child features:\n{baby_info}\n\nProvide a personalized care and survival plan in JSON format as described."}
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

# --- caching wrapper + public generator for HF survival plan ---
@lru_cache(maxsize=256)
def _cached_generate_plan_tuple(features_tuple):
    features = {
        "birth_weight": features_tuple[0],
        "maternal_age": features_tuple[1],
        "immunized": features_tuple[2],
        "nutrition": features_tuple[3],
        "socioeconomic": features_tuple[4],
        "prenatal_visits": features_tuple[5]
    }
    plan, debug = _generate_survival_plan_hf_uncached(features)
    return json.dumps((plan, debug))

def generate_survival_plan_hf(features):
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

# === Explanation function (keeps your robust multi-method approach) ===
def explain_local_prediction(input_df):
    """
    Tries SHAP -> linear coef -> permutation -> heuristic.
    Returns dict with:
      - method
      - feature_contributions (raw)
      - feature_importance_human (list)
      - ui_html (user-friendly HTML snippet)
    """
    if model is None:
        return {"error": "no_local_model"}

    X = input_df.copy()
    feature_names = list(X.columns)
    standardized_feats = []
    method = "unknown"

    # SHAP
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
                    try:
                        safe_val = float(val)
                    except (ValueError, TypeError):
                        safe_val = 0.0
                    standardized_feats.append({"feature": fname, "value": safe_val, "contribution": float(c)})
                method = "shap"
    except Exception as e:
        logger.debug("SHAP failed: %s", e)

    # Linear coefficients
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

    # Permutation importance
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

    # Heuristic fallback
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

    standardized_feats = sorted(standardized_feats, key=lambda x: abs(x.get("contribution", 0.0)), reverse=True)

    FRIENDLY = {
        "birth_weight": ("Birth weight", "Provide feeding support and frequent weight checks; monitor growth."),
        "maternal_age": ("Maternal age", "Arrange maternal follow-up and counseling as needed."),
        "immunized": ("Immunization status", "Keep vaccinations on schedule and follow local immunization program."),
        "nutrition": ("Nutrition score", "Offer breastfeeding support and nutrient-rich complementary foods."),
        "socioeconomic": ("Socioeconomic status", "Link family to community resources and social support."),
        "prenatal_visits": ("Prenatal visits", "Arrange antenatal follow-up and skilled birth attendance.")
    }

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

    summary = " ".join(bullets) if bullets else "No specific risk drivers identified."

    # Calculate scale for bars
    max_contrib = max([abs(float(x.get("contribution", 0.0))) for x in standardized_feats] + [0.001])
    
    list_items_html = ""
    for it in humanified:
        contrib = float(it.get("contribution", 0.0))
        pct = (abs(contrib) / max_contrib) * 100.0
        
        # Color logic: Increase risk (bad) = Red, Decrease risk (good) = Green
        # Note: If high mortality risk is 1, then positive contrib = Risk ↑ (Red)
        if contrib > 0:
            bar_color = "#ef4444" # Red
            bg_color = "#fee2e2"
            icon = "▲"
            effect_text = "Increasing Risk"
        else:
            bar_color = "#10b981" # Green
            bg_color = "#d1fae5"
            icon = "▼"
            effect_text = "Reducing Risk"
            
        list_items_html += (
            f"<div style='margin-bottom:12px; padding:10px; background:white; border-radius:8px; border:1px solid #e5e7eb'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px'>"
                    f"<span style='font-weight:600; color:#1f2937'>{it['label']}</span>"
                    f"<span style='font-size:12px; font-weight:700; color:{bar_color}'>{icon} {effect_text}</span>"
                f"</div>"
                
                f"<div style='display:flex; align-items:center; gap:10px; font-size:12px; margin-bottom:4px'>"
                    f"<div style='flex-grow:1; background:#f3f4f6; height:8px; border-radius:4px; overflow:hidden'>"
                        f"<div style='width:{pct}%; height:100%; background:{bar_color}; border-radius:4px'></div>"
                    f"</div>"
                    f"<div style='min-width:40px; text-align:right; font-weight:600; color:#4b5563'>{it['relative_importance_pct']}%</div>"
                f"</div>"
                
                f"<div style='font-size:13px; color:#4b5563; margin-top:4px'>"
                    f"{it['recommendation']}"
                f"</div>"
            f"</div>"
        )

    recommendations_html = "".join([f"<li style='margin-bottom:4px'>• {r['recommendation']}</li>" for r in recommendations[:3]])
    tech_json = json.dumps({"feature_contributions": standardized_feats}, indent=2)
    
    # Method display name
    method_disp = "SMOTE-Enhanced SHAP" if method == "shap" else method.replace("_", " ").title()

    ui_html = (
        f"<div style='font-family:sans-serif'>"
            f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:12px'>"
                f"<h4 style='font-weight:700; font-size:16px; margin:0; color:#1f2937'>Contribution Analysis</h4>"
                f"<span style='font-size:10px; text-transform:uppercase; letter-spacing:0.5px; background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-weight:700'>{method_disp}</span>"
            f"</div>"
            
            f"<div style='background:#f9fafb; padding:12px; border-radius:8px; border:1px solid #e5e7eb; margin-bottom:16px'>"
                f"<div style='font-size:13px; color:#374151; font-weight:600; margin-bottom:4px'>Summary</div>"
                f"<div style='font-size:14px; color:#111827'>{summary}</div>"
            f"</div>"

            f"<div style='margin-bottom:16px'>{list_items_html}</div>"

            f"<div style='border-top:1px solid #e5e7eb; padding-top:12px'>"
                f"<div style='font-size:13px; font-weight:600; color:#374151; margin-bottom:6px'>Top Recommendations</div>"
                f"<ul style='padding-left:14px; margin:0; font-size:13px; color:#4b5563; line-height:1.5'>{recommendations_html or '<li>No specific actions required.</li>'}</ul>"
            f"</div>"
            
            f"<details style='margin-top:12px; border-top:1px solid #e5e7eb; padding-top:8px'>"
                f"<summary style='cursor:pointer; font-size:12px; font-weight:600; color:#6b7280; user-select:none'>Show Technical Data</summary>"
                f"<pre style='font-size:11px; white-space:pre-wrap; padding:8px; background:#111827; color:#10b981; border-radius:6px; margin-top:6px; overflow-x:auto'>{tech_json}</pre>"
            f"</details>"
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

    logger.info("Explanation method used: %s", method)
    return explanation_payload

# --- Caching wrapper to avoid repeated identical HF calls for survival plan (public) ---
@lru_cache(maxsize=256)
def _cached_generate_plan_tuple_public(features_tuple_with_model):
    features = {
        "birth_weight": features_tuple_with_model[0],
        "maternal_age": features_tuple_with_model[1],
        "immunized": features_tuple_with_model[2],
        "nutrition": features_tuple_with_model[3],
        "socioeconomic": features_tuple_with_model[4],
        "prenatal_visits": features_tuple_with_model[5]
    }
    plan, debug = _generate_survival_plan_hf_uncached(features)
    return json.dumps((plan, debug))

def generate_survival_plan_hf_public(features):
    tup = (
        float(features.get("birth_weight", 0.0)),
        float(features.get("maternal_age", 0.0)),
        int(features.get("immunized", 0)),
        float(features.get("nutrition", 0.0)),
        int(features.get("socioeconomic", 0)),
        float(features.get("prenatal_visits", 0.0)),
        HF_MODEL
    )
    try:
        cached = _cached_generate_plan_tuple_public(tup)
        plan, debug = json.loads(cached)
        return plan, debug
    except Exception:
        return _generate_survival_plan_hf_uncached({
            "birth_weight": tup[0],
            "maternal_age": tup[1],
            "immunized": tup[2],
            "nutrition": tup[3],
            "socioeconomic": tup[4],
            "prenatal_visits": tup[5]
        })

# --- Survival plan selector ---
def survival_plan(prediction, features=None, debug=False):
    if int(prediction) == 0:
        return {"risk_level": "low", "message": [
            "Continue routine preventive care and regular health checkups",
            "Complete immunization schedule on schedule",
            "Ensure balanced nutrition and adequate feeding",
            "Maintain safe sleeping environment (back sleeping, firm surface)",
            "Practice good hygiene and sanitation habits",
            "Monitor for early signs of illness and seek prompt care",
            "Provide safe drinking water and proper sanitation",
            "Ensure adequate rest and physical activity for development"
        ]}, {"branch": "low"}
    else:
        if features:
            plan, debug_info = generate_survival_plan_hf_public(features)
            return plan, {"branch": "high", **(debug_info or {})}
        else:
            return generate_fallback_plan(), {"branch": "high", "hf_raw": None, "error": "no_features"}

# ------------------- SQLite history DB helpers -------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def _get_db_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    # Custom timestamp converter to handle ISO format strings
    def convert_timestamp(val):
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        # Handle ISO format: 2025-12-06T21:07:19.604
        if 'T' in val:
            return val.replace('T', ' ')
        return val
    sqlite3.register_converter("timestamp", convert_timestamp)
    return conn

def init_history_db():
    conn = _get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_id TEXT,
        timestamp TIMESTAMP NOT NULL,
        features_json TEXT,
        probability REAL,
        prediction INTEGER,
        risk_category TEXT,
        survival_plan_json TEXT,
        explanation_json TEXT,
        hf_raw TEXT,
        model_used TEXT,
        explanation_method TEXT,
        patient_name TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    
    # Migration: ensure patient_name column exists
    try:
        cur.execute("ALTER TABLE history ADD COLUMN patient_name TEXT")
    except sqlite3.OperationalError:
        # column likely already exists
        pass

    conn.commit()
    conn.close()

init_history_db()

def _ensure_session_id():
    sid = flask_session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        flask_session["sid"] = sid
    return sid

def _current_user_id():
    return flask_session.get("user_id")

def save_prediction_to_history(features, probability, prediction, risk_category, survival_plan,
                               explanation=None, hf_raw=None, model_used=None, explanation_method=None, patient_name=None):
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        user_id = _current_user_id()
        session_id = _ensure_session_id()
        ts = datetime.utcnow().isoformat()
        cur.execute("""
            INSERT INTO history
            (user_id, session_id, timestamp, features_json, probability, prediction, risk_category,
             survival_plan_json, explanation_json, hf_raw, model_used, explanation_method, patient_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            session_id,
            ts,
            json.dumps(features, default=str),
            float(probability) if probability is not None else None,
            int(prediction) if prediction is not None else None,
            risk_category,
            json.dumps(survival_plan, default=str),
            json.dumps(explanation, default=str) if explanation is not None else None,
            hf_raw,
            model_used,
            explanation_method,
            patient_name
        ))
        conn.commit()
        rowid = cur.lastrowid
        conn.close()
        logger.info("Saved prediction history id=%s user_id=%s session=%s name=%s", rowid, user_id, session_id, patient_name)
        return rowid
    except Exception as e:
        logger.exception("Failed to save history: %s", e)
        return None

# ----------------- Small new helper route: whoami -----------------
@app.route("/api/auth/whoami", methods=["GET"])
def api_whoami():
    """
    Returns current session-based user info or 401 if not logged in.
    Frontend uses this to display login state without relying on tokens.
    """
    user_id = flask_session.get("user_id")
    if not user_id:
        return jsonify({"logged_in": False}), 200
    conn = _get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        # session contains stale user_id
        flask_session.pop("user_id", None)
        return jsonify({"logged_in": False}), 200
    return jsonify({"logged_in": True, "user_id": r["id"], "username": r["username"], "created_at": r["created_at"]}), 200

# ------------------- Auth & History endpoints -------------------
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error":"username & password required"}), 400
    conn = _get_db_conn()
    cur = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, pw_hash, datetime.utcnow().isoformat()))
        conn.commit()
        user_id = cur.lastrowid
        flask_session["user_id"] = user_id
        conn.close()
        return jsonify({"ok": True, "user_id": user_id})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error":"username already exists"}), 400
    except Exception as e:
        conn.close()
        logger.exception("register failed: %s", e)
        return jsonify({"error":"internal"}), 500

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username"); password = data.get("password")
    if not username or not password:
        return jsonify({"error":"username & password required"}), 400
    conn = _get_db_conn(); cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close(); return jsonify({"error":"invalid credentials"}), 401
    user_id = row["id"]; pw_hash = row["password_hash"]
    if not check_password_hash(pw_hash, password):
        conn.close(); return jsonify({"error":"invalid credentials"}), 401
    flask_session["user_id"] = user_id
    conn.close()
    return jsonify({"ok": True, "user_id": user_id})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    flask_session.pop("user_id", None)
    return jsonify({"ok": True})

@app.route("/api/history", methods=["GET"])
def api_history_list():
    limit = int(request.args.get("limit", 20))
    search_term = request.args.get("search", "").strip()
    user_id = _current_user_id()
    session_id = flask_session.get("sid")
    conn = _get_db_conn(); cur = conn.cursor()
    
    query = "SELECT * FROM history WHERE "
    params = []
    
    if user_id:
        query += "user_id = ? "
        params.append(user_id)
    else:
        query += "session_id = ? "
        params.append(session_id)
        
    if search_term:
        query += "AND patient_name LIKE ? "
        params.append(f"%{search_term}%")
        
    query += "ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    cur.execute(query, tuple(params))
    rows = cur.fetchall(); conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "features": json.loads(r["features_json"]) if r["features_json"] else None,
            "probability": r["probability"],
            "prediction": r["prediction"],
            "risk_category": r["risk_category"],
            "survival_plan": json.loads(r["survival_plan_json"]) if r["survival_plan_json"] else None,
            "explanation": json.loads(r["explanation_json"]) if r["explanation_json"] and request.args.get("debug","").lower()=="true" else None,
            "model_used": r["model_used"],
            "explanation_method": r["explanation_method"],
            "patient_name": r["patient_name"]
        })
    return jsonify({"history": out})

@app.route("/api/history/<int:hid>", methods=["GET"])
def api_history_get(hid):
    conn = _get_db_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM history WHERE id = ?", (hid,))
    r = cur.fetchone(); conn.close()
    if not r:
        return jsonify({"error":"not found"}), 404
    rec = {
        "id": r["id"],
        "timestamp": r["timestamp"],
        "features": json.loads(r["features_json"]) if r["features_json"] else None,
        "probability": r["probability"],
        "prediction": r["prediction"],
        "risk_category": r["risk_category"],
        "survival_plan": json.loads(r["survival_plan_json"]) if r["survival_plan_json"] else None,
        "explanation": json.loads(r["explanation_json"]) if r["explanation_json"] and request.args.get("debug","").lower()=="true" else None,
        "hf_raw": r["hf_raw"] if request.args.get("debug","").lower()=="true" else None,
        "model_used": r["model_used"],
        "explanation_method": r["explanation_method"],
        "patient_name": r["patient_name"]
    }
    return jsonify(rec)


@app.route("/api/history/<int:hid>", methods=["DELETE"])
def api_history_delete(hid):
    """Delete a single history record belonging to the current logged-in user."""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login required"}), 401
    conn = _get_db_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (hid, user_id))
    if cur.rowcount == 0:
        conn.close()
        return jsonify({"error": "not found or not authorized"}), 404
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/history/<int:hid>", methods=["PUT"])
def api_history_update(hid):
    """Update a single history record (e.g., patient_name) for the logged-in user."""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login required"}), 401
    
    payload = request.get_json() or {}
    new_name = payload.get("patient_name")
    if not new_name:
        return jsonify({"error": "missing patient_name"}), 400
        
    conn = _get_db_conn(); cur = conn.cursor()
    cur.execute("UPDATE history SET patient_name = ? WHERE id = ? AND user_id = ?", (new_name.strip(), hid, user_id))
    
    if cur.rowcount == 0:
        conn.close()
        return jsonify({"error": "not found or not authorized"}), 404
        
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/history", methods=["DELETE"])
def api_history_delete_all():
    """Delete all history for the current logged-in user."""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login required"}), 401
    conn = _get_db_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    conn.commit(); conn.close()
    return jsonify({"ok": True, "deleted": True})

# --- Merge local client history into logged-in user's history ---
@app.route("/api/history/merge", methods=["POST"])
def api_history_merge():
    """
    Accepts JSON: { "entries": [ { timestamp, input_json OR features, prediction_result OR prediction,
                                   survival_plan, explanation, probability } , ... ] }
    Requires user to be logged in (session-based). Returns inserted count.
    """
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login required"}), 401

    payload = request.get_json(silent=True) or {}
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        return jsonify({"error": "entries must be a list"}), 400

    inserted = 0
    conn = _get_db_conn()
    cur = conn.cursor()
    session_id = _ensure_session_id()
    for e in entries:
        try:
            ts = e.get("timestamp") or datetime.utcnow().isoformat()
            features = e.get("input_json") or e.get("features") or e.get("inputs") or {}
            try:
                features_json = json.dumps(features, default=str)
            except Exception:
                features_json = json.dumps({}, default=str)
            probability = e.get("probability")
            try:
                if probability is not None:
                    p = float(probability)
                    if p > 1.0 and p <= 100.0:
                        probability = p / 100.0
                    else:
                        probability = p
            except Exception:
                probability = None
            prediction = e.get("prediction") or (e.get("prediction_result") and (e.get("prediction_result").get("pred") if isinstance(e.get("prediction_result"), dict) else e.get("prediction_result")))
            prediction_val = int(prediction) if prediction is not None else None
            survival_plan = e.get("survival_plan") or e.get("plan") or {}
            explanation = e.get("explanation") or e.get("debug") or None

            cur.execute("""
                INSERT INTO history
                (user_id, session_id, timestamp, features_json, probability, prediction, risk_category, survival_plan_json, explanation_json, hf_raw, model_used, explanation_method, patient_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                session_id,
                ts,
                features_json,
                float(probability) if probability is not None else None,
                prediction_val,
                None,
                json.dumps(survival_plan, default=str),
                json.dumps(explanation, default=str) if explanation is not None else None,
                None,
                HF_MODEL,
                None,
                e.get("patient_name")
            ))
            inserted += 1
        except Exception as exc:
            logger.exception("merge entry failed: %s", exc)
            continue
    conn.commit()
    conn.close()
    return jsonify({"status": "merged", "inserted": inserted})

# --- Routes: index, health, reload_model ---
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

# --- Predict endpoint (integrates explanation + history save) ---
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
            # Clamp minimum risk to 0.5% and maximum to 99.9% to avoid absolutes
            if prob < 0.005:
                prob = 0.005
            elif prob > 0.999:
                prob = 0.999
        except Exception:
            prob = None
        pred = int(model.predict(df)[0])
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        return jsonify({"error": "Model inference failed", "details": str(e)}), 500

    features = df.iloc[0].to_dict()
    # survival plan
    plan_obj, debug_info = survival_plan(pred, features, debug=debug_flag)

    response = {
        "mortality_risk_probability": prob,
        "mortality_prediction": pred,
        "interpretation": "1 means higher predicted risk, 0 means lower predicted risk",
        "survival_plan": plan_obj,
        "debug": {"branch": debug_info.get("branch", "high" if pred == 1 else "low")}
    }

    # attach survival plan debug / HF raw if present
    if debug_flag:
        hf_raw = debug_info.get("hf_raw")
        if hf_raw:
            response["debug"]["hf_raw_truncated"] = hf_raw[:8000]
        if "error" in debug_info:
            response["debug"]["error"] = debug_info["error"]

    # attach explanation (attempt)
    explanation_payload = None
    try:
        explanation_payload = explain_local_prediction(df)
        if explanation_payload:
            # include explanation in debug only
            if debug_flag:
                response["debug"]["explanation"] = explanation_payload
                response["debug"]["explanation_summary"] = explanation_payload.get("summary")
    except Exception as ex:
        logger.exception("Explanation generation failed: %s", ex)
        if debug_flag:
            response["debug"]["explanation_error"] = str(ex)

    # prepare risk category
    risk_cat = "High" if pred == 1 or (prob is not None and prob >= 0.6) else ("Medium" if prob is not None and prob >= 0.35 else "Low")

    # Save history (best-effort)
    try:
        save_prediction_to_history(
            features=features,
            probability=prob,
            prediction=pred,
            risk_category=risk_cat,
            survival_plan=plan_obj,
            explanation=explanation_payload if debug_flag else (explanation_payload if False else None),
            hf_raw=debug_info.get("hf_raw") if isinstance(debug_info, dict) else None,
            model_used=HF_MODEL,
            explanation_method=(explanation_payload.get("method") if isinstance(explanation_payload, dict) else None),
            patient_name=data.get("patient_name")
        )
    except Exception as e:
        logger.exception("history save failed: %s", e)

    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict_alias():
    return predict_api()

# === Run server ===
if __name__ == "__main__":
    print("🚀 Starting Child Mortality API server on http://127.0.0.1:8000 ")
    app.run(host="0.0.0.0", port=8000, debug=False)
