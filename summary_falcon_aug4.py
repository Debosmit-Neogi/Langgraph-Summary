#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# langgraph_falcon_api.py
# Run:  python3 langgraph_falcon_api.py
# Test: curl -X POST http://0.0.0.0:5090/extract_lifestyle \
#         -H "Content-Type: application/json" \
#         --data @/path/to/test_input_fever.json

# Does not hallucinate much, but does not extract question-answers text a lot. Captures lifestyle factors though

# "followup_question_en": "Does the chest pain increase when walking or climbing stairs?",
# "response": "Sometimes it happens, but I drink a lot of water"
# The summary does not mention that the chest pain increases sometimes when walking or climbing stairs

import os, re, json, ast, logging, traceback, time
from typing import List, Dict, Tuple, TypedDict
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- Logging ----------------
logger = logging.getLogger("falcon_summary")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

app = Flask(__name__)


# ---------------- New similarity filter ----------------
# Put these near the other imports
from sentence_transformers import SentenceTransformer, util

# Load once at startup; model & threshold are configurable
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.25"))        # sensible default

embedder = SentenceTransformer(EMBEDDING_MODEL)

YES_NO_RE = re.compile(r"^\s*(yes|no|y|n|yeah|nah|na|haan|haanji|nahi)\b", re.I)

def _is_yes_no(ans: str) -> bool:
    return bool(YES_NO_RE.match(ans))

def _qa_semantic_match(q: str, a: str) -> bool:
    """
    Return True if 'a' is contextually relevant to 'q'.
    Logic:
      • Always accept explicit yes/no style replies.
      • Otherwise compute cosine-similarity of embeddings and compare
        against SIM_THRESHOLD (configurable).
    """
    if _is_yes_no(a):
        return True
    try:
        emb_q = embedder.encode(q, convert_to_tensor=True)
        emb_a = embedder.encode(a, convert_to_tensor=True)
        score = util.cos_sim(emb_q, emb_a).item()
        return score >= SIM_THRESHOLD
    except Exception as e:
        logger.warning(f"⚠️ similarity check failed; defaulting to False – {e}")
        return False


# ---------------- Model ----------------
MODEL_NAME = os.getenv("FALCON_MODEL", "tiiuae/falcon-7b-instruct")

class LocalLLM:
    def __init__(self, model_name=MODEL_NAME):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

local_llm = LocalLLM()

# ---------------- Prompt ----------------
SUMMARY_PROMPT = """You are a careful medical assistant. Summarize the following doctor-patient Q&A.
• Use only what is explicitly stated. Do not invent information.
• Include confirmed symptoms, their durations if available, triggers/impacts, and lifestyle factors or habits that are clearly mentioned.
• Avoid headings, quotes, or extra labels. Output one concise paragraph.
• Do **not** use speculative language (e.g. “suspects”, “may”, “might”, “likely”, “appears”, “seems”).  
• If an answer is nonsense / unrelated to its question, or has N/A as response, ignore that Q-A pair entirely.  
• Include only confirmed symptoms, their durations, triggers/impacts, and lifestyle factors.  
• Avoid headings, quotes, or extra labels. Produce one concise paragraph.


Q&A:
{qa}

Summary:"""

# ---------------- Helpers ----------------
FILLER_STRINGS = {"", "none", "null", "na", "n/a", "maybe", "no", "nothing", "ok", "okay", "fine"}
def normalize_space(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def try_parse_body() -> Dict:
    """
    Robust JSON loader:
    1) request.get_json(silent=True)
    2) raw bytes -> utf-8-sig -> json.loads
    3) ast.literal_eval for Python-like dicts with single quotes / None / True / False
    """
    # 1) Fast path
    obj = request.get_json(silent=True)
    if isinstance(obj, dict):
        return obj

    raw_bytes = request.get_data(cache=False) or b""
    if not raw_bytes:
        raise ValueError("Empty request body")

    text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ValueError("Empty request body (after decode)")

    # 2) Strict JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Python-literal fallback (single quotes, None, True/False)
    try:
        lit = ast.literal_eval(text)
        if isinstance(lit, dict):
            return lit
    except Exception:
        pass

    # 4) Last resort: replace single quotes with double quotes cautiously
    # Only attempt if it looks like a dict
    if text.startswith("{") and text.endswith("}"):
        repaired = (text
                    .replace("None", "null")
                    .replace("True", "true")
                    .replace("False", "false"))
        # naive single->double for keys/strings
        repaired = re.sub(r"(?<!\\)'", '"', repaired)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError("Failed to parse body as JSON or Python literal.")

def convert_history_to_qa(conversation_history: List[dict]) -> str:
    """
    Build a cleaned linear QA transcript:
      • Keeps only Q-A pairs whose answers are relevant by _qa_semantic_match.
      • First user utterance (chief complaint) is always included.
    """
    lines = []
    first_user_done = False

    for turn in conversation_history or []:
        if not isinstance(turn, dict):
            continue

        # Chief complaint (first 'user')
        if "user" in turn and not first_user_done:
            u = normalize_space(turn.get("user", ""))
            if u:
                lines.append("Doctor: Describe your symptoms?")
                lines.append(f"Patient: {u}")
                first_user_done = True
            continue

        q = normalize_space(turn.get("followup_question_en", ""))
        a = normalize_space(turn.get("response", ""))

        if q and a and _qa_semantic_match(q, a):
            lines.append(f"Doctor: {q}")
            lines.append(f"Patient: {a}")

    return "\n".join(lines)


NEG_TOKENS = r"(?:do\s*not|does\s*not|don't|never|no)\s+"
def extract_lifestyle_factors(conversation_history: List[dict]) -> List[str]:
    """
    Simple, strict regex-based extraction with negation handling.
    Returns normalized, deduped phrases in lowercase.
    """
    text_bits = []
    for t in conversation_history or []:
        if isinstance(t, dict):
            for k in ("user", "response", "followup_question_en"):
                v = t.get(k)
                if isinstance(v, str) and v.strip():
                    text_bits.append(v.lower())
    raw = " ".join(text_bits)
    raw = re.sub(r"\s+", " ", raw)

    factors = []

    # Smoking
    if re.search(rf"{NEG_TOKENS}smok\w*", raw):
        factors.append("does not smoke")
    elif re.search(r"\b(smok\w*|cig(ar|arette)s?|bidi|beedi)\b", raw):
        # refine heavy/occasional
        if re.search(r"\b(heavy|a lot|too much|many|chain)\b", raw):
            factors.append("smoking (heavy)")
        elif re.search(r"\b(occasion(al|ally)|sometimes|rarely|social(ly)?)\b", raw):
            factors.append("smoking (occasional)")
        else:
            factors.append("smoking")

    # Alcohol
    if re.search(rf"{NEG_TOKENS}(drink|take)\s+(alcohol|beer|whisk(e?)y|wine|rum|vodka|daru)", raw):
        factors.append("does not drink alcohol")
    elif re.search(r"\b(alcohol|beer|whisk(e?)y|wine|rum|vodka|daru|drinks?)\b", raw):
        if re.search(r"\b(daily|every\s*day|regular(ly)?)\b", raw):
            factors.append("alcohol use (daily)")
        elif re.search(r"\bsocial(ly)?\b", raw):
            factors.append("alcohol use (social)")
        else:
            factors.append("alcohol use")

    # Tobacco / pan masala / gutkha / hookah
    if re.search(rf"{NEG_TOKENS}(tobacco|pan\s*masala|gutkha|hookah|chew)", raw):
        factors.append("does not use tobacco")
    elif re.search(r"\b(tobacco|pan\s*masala|gutkha|hookah|chew(ing)?\s*tobacco)\b", raw):
        factors.append("tobacco use")

    # Caffeine / high intake
    if re.search(r"\b(caffeine|energy\s*drink)\b", raw) or \
       (re.search(r"\b(tea|coffee)\b", raw) and re.search(r"\b(too\s*much|excess)\b", raw)):
        factors.append("high caffeine intake")

    # Diet: junk / oily / salty / sugary drinks
    if re.search(r"\bsugary\s*drinks?|soft\s*drink|cold\s*drink|soda|cola|fizzy\b", raw):
        factors.append("sugary drinks")
    if re.search(r"\bjunk\b|\bfast\s*food\b", raw):
        factors.append("junk food")
    if re.search(r"\boily\b|\bgreasy\b", raw):
        factors.append("oily food")
    if re.search(r"\b(high\s*)?salt(y)?\b", raw):
        factors.append("high salt diet")

    # Medical history (diabetes / sugar, BP)
    if re.search(r"\b(diabetes|diabetic|sugar)\b", raw):
        # Negation:
        if re.search(rf"{NEG_TOKENS}(diabetes|sugar)", raw):
            factors.append("no diabetes")
        else:
            factors.append("diabetes")
    if re.search(r"\b(hypertension|high\s*bp|blood\s*pressure)\b", raw):
        if re.search(rf"{NEG_TOKENS}(hypertension|high\s*bp|blood\s*pressure)", raw):
            factors.append("no hypertension")
        else:
            factors.append("hypertension")

    # De-dup while preserving order
    seen = set()
    out = []
    for x in factors:
        x = normalize_space(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def make_summary_with_falcon(qa_text: str) -> str:
    """
    Generate twice and return the second summary chunk after the 'Summary:' marker.
    """
    prompt = SUMMARY_PROMPT.format(qa=qa_text)

    # 1st run (warm-up / variance)
    _ = local_llm.generate(prompt)
    # 2nd run
    full = local_llm.generate(prompt)

    # Extract text after final "Summary:"
    idx = full.rfind("Summary:")
    if idx != -1:
        summary = full[idx + len("Summary:"):].strip()
    else:
        summary = full.strip()

    # Strip enclosing quotes, cut any leading labels
    summary = re.sub(r'^["“”]+|["“”]+$', "", summary).strip()
    summary = re.sub(r'^(summary|output)\s*:?[\s-]*', '', summary, flags=re.I).strip()

    # If it's empty or meaningless, return ""
    if not summary or summary.lower() in {"summary", "n/a", "none"}:
        return ""
    # Avoid degenerate "The patient reports." etc.
    if re.fullmatch(r"The patient reports[\.!?]?", summary):
        return ""

    return summary if summary.endswith((".", "!", "?")) else summary + "."

def ensure_list(x) -> List[str]:
    if isinstance(x, list):
        return [normalize_space(s).lower() for s in x if isinstance(s, str) and normalize_space(s)]
    if isinstance(x, str) and normalize_space(x):
        return [normalize_space(x).lower()]
    return []

# ---------------- Route ----------------
@app.route("/extract_lifestyle", methods=["POST"])
def extract_lifestyle_handler():
    try:
        req = try_parse_body()
    except Exception as e:
        logger.exception("parse error")
        return jsonify({
            "error": "Failed to extract lifestyle/summary.",
            "detail": str(e),
            "trace": traceback.format_exc()
        }), 400

    try:
        conversation_history = req.get("conversation_history") or \
                               (req.get("finalResult", {}) or {}).get("conversation_history") or []
        if not isinstance(conversation_history, list):
            conversation_history = []

        # Summary (Falcon, use 2nd run)
        qa_text = convert_history_to_qa(conversation_history)
        summary = make_summary_with_falcon(qa_text)

        # --- simple fix: change “doctor” → “patient” ---
        if "doctor" in summary.lower():                 # case-insensitive check
            summary = re.sub(r"\bdoctor\b", "patient", summary, flags=re.I)

        if "chest pain" in summary.lower():                 # case-insensitive check
            summary = re.sub(r"\bchest pain\b", "pain", summary, flags=re.I)

        # Lifestyle factors (regex, strict with negation)
        lifestyle = extract_lifestyle_factors(conversation_history)

        # If summary ended up empty but lifestyle exists, still provide a short sentence
        if not summary and lifestyle:
            summary = "The patient reports " + ", ".join(lifestyle) + "."
        # If truly nothing, keep empty string
        if not summary:
            summary = ""

        return jsonify({
            "lifestyle_factors": summary
            #"summary": summary
        }), 200

    except Exception as e:
        logger.exception("extract handler error")
        return jsonify({
            "error": "Failed to extract lifestyle/summary.",
            "detail": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
