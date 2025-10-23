import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = "/content/outputs"
METRICS_JSON_PATH = "/content/outputs/metrics.json"
SEGMENTS_FILES = ["/content/raw_eval.json"]
KEYPOINTS_FILES = ["/content/GDPR_articles_obligations_keypoints_organized.json"]
MODEL_REPO = "openai/gpt-oss-20b"
BATCH_SIZE = 192
GEN_TEMPERATURE = 0.4
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOK = 96
MAX_SEQ_LEN = 4096
LABELS = ["COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"]
VALID = set(LABELS)
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.I)
FINAL_TAG = "<|channel|>final<|message|>"
END_TAG = "<|end|>"
JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}")
FEW_SHOTS = [
    {
        "article": 35,
        "label": "NOT_APPLICABLE",
        "excerpt": "Given the nature and scale of our processing activities, we do not conduct operations that are likely to result in a high risk to individuals' rights and freedoms. Therefore, we are not required to carry out a data protection impact assessment. Should our data processing practices change in the future in a way that may introduce such risks, we will promptly review our obligations and implement the necessary assessment procedures to ensure ongoing protection of personal data.",
        "keypoint": "Where a type of processing is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out a data protection impact assessment."
    },
    {
        "article": 15,
        "label": "NON_COMPLIANT",
        "excerpt": "We do not confirm whether or not we process your personal data upon request.",
        "keypoint": "The controller shall, upon request by the data subject, confirm whether or not personal data concerning the data subject are being processed."
    },
    {
        "article": 8,
        "label": "COMPLIANT",
        "excerpt": "We only offer information society services to users aged 16 or older. For users under 16, we process personal data only with verifiable parental consent.",
        "keypoint": "Where point (a) of Article 6(1) applies, in relation to the offer of information society services directly to a child, processing of a child's personal data is lawful where the child is at least 16 years old."
    },
]

SYSTEM_INSTRUCTIONS = "You are a GDPR compliance assistant. Respond with STRICT JSON only in your final message."
TASK_TEMPLATE = (
    "Analyze ONLY the key obligation below.\n\n"
    "You will see a few short EXAMPLES first. Study them, then answer for the TARGET excerpt only.\n"
    "Return a single-line JSON in your FINAL message. No code blocks. No extra text.\n\n"
    "EXAMPLES (format: Article → Key obligation → Excerpt → Expected JSON):\n"
    "{few_shots_block}\n"
    "---\n"
    "TARGET TASK\n"
    "Article {art} — Key obligation ({kp_id}):\n- {kp_text}\n\n"
    "Policy excerpt to evaluate:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
    "Decide whether this excerpt satisfies THIS specific obligation (ignore unrelated obligations).\n\n"
    "Definition:\n- NOT_APPLICABLE: The obligation is irrelevant to this excerpt’s context.\n\n"
    "Respond ONLY with this JSON (no extra text):\n"
    "{json_hint}"
)

def clip(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    return s if len(s) <= n else s[:n] + "…"

def parse_index_list(s: Any) -> set:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {int(x.strip()) for x in s.split(",") if x.strip().isdigit()}

def macro_metrics(y_true, y_pred, labels=LABELS) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"Macro-P": float(p), "Macro-R": float(r), "Macro-F1": float(f1), "Accuracy": float(acc)}

def load_segments_flat(files: List[str]) -> pd.DataFrame:
    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for s in data:
            if not all(k in s for k in ["gdpr_article_number", "label", "segment"]):
                continue
            rows.append(
                {
                    "gdpr_article_number": int(s["gdpr_article_number"]),
                    "segment": str(s["segment"]),
                    "label": str(s["label"]).strip().upper(),
                    "met_points": s.get("met_points", ""),
                    "unmet_points": s.get("unmet_points", ""),
                    "policy_id": s.get("policy_id"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No segments found.")
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df["segment_id"] = np.arange(1, len(df) + 1)
    return df

def load_keypoints(files: List[str]) -> Dict[int, List[str]]:
    kp_map: Dict[int, List[str]] = {}
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for obj in data:
            if isinstance(obj, dict) and "gdpr_article" in obj and "keypoints" in obj:
                art = int(obj["gdpr_article"])
                kp_map[art] = [str(x).strip() for x in obj["keypoints"]]
    if not kp_map:
        raise ValueError("No keypoints found.")
    return kp_map

def expand_pairs(segments_df: pd.DataFrame, kp_map: Dict[int, List[str]]) -> pd.DataFrame:
    rows = []
    for _, r in segments_df.iterrows():
        art = int(r["gdpr_article_number"])
        kps = kp_map.get(art)
        if not kps:
            continue
        met, unmet = parse_index_list(r["met_points"]), parse_index_list(r["unmet_points"])
        K = len(kps)
        for idx in range(1, K + 1):
            if idx in met:
                gold = "COMPLIANT"
            elif idx in unmet:
                gold = "NON_COMPLIANT"
            else:
                gold = "NOT_APPLICABLE"
            rows.append(
                {
                    "segment_id": int(r["segment_id"]),
                    "policy_id": r.get("policy_id"),
                    "gdpr_article_number": art,
                    "keypoint_index": idx,
                    "keypoint_text": kps[idx - 1],
                    "segment": r["segment"],
                    "gold_label": gold,
                }
            )
    return pd.DataFrame(rows)

def build_few_shots_block(shots: Optional[List[Dict[str, str]]] = None) -> str:
    shots = shots or FEW_SHOTS
    parts: List[str] = []
    for i, s in enumerate(shots, 1):
        parts.append(
            f"Example {i}:\n- Article: {s.get('article')}\n- Key obligation:\n{s.get('keypoint','').strip()}\n- Excerpt:\n\"\"\"\n{s.get('excerpt','').strip()}\n\"\"\"\n- Expected JSON:\n{{ \"label\": \"{s.get('label','').strip()}\" }}\n"
        )
    return "\n".join(parts)

def build_messages(excerpt: str, art: int, kp_id: str, kp_text: str) -> List[Dict[str, str]]:
    user_msg = TASK_TEMPLATE.format(
        art=art,
        kp_id=kp_id,
        kp_text=kp_text,
        excerpt=excerpt,
        json_hint='{"label": "<COMPLIANT|NON_COMPLIANT|NOT_APPLICABLE>"}',
        few_shots_block=build_few_shots_block(),
    )
    return [{"role": "system", "content": SYSTEM_INSTRUCTIONS}, {"role": "user", "content": user_msg}]

def strip_to_final_channel(txt: str) -> str:
    if not txt:
        return txt
    i = txt.rfind(FINAL_TAG)
    if i == -1:
        return txt
    s = txt[i + len(FINAL_TAG):]
    j = s.find(END_TAG)
    return s[:j] if j != -1 else s

def extract_label_from_text(txt: str) -> str:
    core = strip_to_final_channel(txt)
    m = JSON_BLOCK_RE.search(core)
    blob = m.group(0) if m else core
    try:
        lab = json.loads(blob).get("label", "").strip().upper()
    except Exception:
        m2 = LABEL_RE.search(blob)
        lab = m2.group(1).upper() if m2 else "UNKNOWN"
    return lab if lab in VALID else "NOT_APPLICABLE"

def encode_one(tok, messages: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = enc
        attn_mask = torch.ones_like(input_ids)
    except Exception:
        txt = f"<s>[INST] {messages[0]['content']} [/INST]"
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
    return input_ids.squeeze(0), attn_mask.squeeze(0)

def generate_batched(model, tok, device, rows: pd.DataFrame, batch_size: int = BATCH_SIZE) -> List[str]:
    preds: List[str] = []
    cache_ids: List[torch.Tensor] = []
    cache_masks: List[torch.Tensor] = []

    def flush():
        nonlocal preds, cache_ids, cache_masks
        if not cache_ids:
            return
        ids = pad_sequence(cache_ids, batch_first=True, padding_value=tok.pad_token_id)
        att = pad_sequence(cache_masks, batch_first=True, padding_value=0)
        ids = ids.to(device, non_blocking=True)
        att = att.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model.generate(
                ids,
                attention_mask=att,
                max_new_tokens=GEN_MAX_NEW_TOK,
                do_sample=False,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        prompt_len = ids.shape[1]
        out = out.to("cpu")
        for bi in range(out.size(0)):
            gen = out[bi, prompt_len:]
            text = tok.decode(gen, skip_special_tokens=False)
            preds.append(extract_label_from_text(text))
        del ids, att, out
        cache_ids.clear()
        cache_masks.clear()

    for _, r in rows.iterrows():
        messages = build_messages(
            excerpt=r["segment"],
            art=int(r["gdpr_article_number"]),
            kp_id=str(r["keypoint_index"]),
            kp_text=r["keypoint_text"],
        )
        input_ids, attn = encode_one(tok, messages)
        cache_ids.append(input_ids)
        cache_masks.append(attn)
        if len(cache_ids) >= batch_size:
            flush()
    flush()
    return preds

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    segments_df = load_segments_flat(SEGMENTS_FILES)
    keypoints_map = load_keypoints(KEYPOINTS_FILES)
    pairs_df = expand_pairs(segments_df, keypoints_map)
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "<s>{% for m in messages %}"
            "{% if m['role']=='user' %}[INST] {{ m['content'] }} [/INST]"
            "{% elif m['role']=='assistant' %} {{ m['content'] }}{% endif %}"
            "{% endfor %}"
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, device_map="auto", torch_dtype="auto")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = generate_batched(model, tok, device, pairs_df, BATCH_SIZE)
    pred_pairs = pairs_df.copy()
    pred_pairs["pred_label"] = preds
    m1 = macro_metrics(pred_pairs["gold_label"], pred_pairs["pred_label"])
    if pred_pairs.get("policy_id", pd.Series(dtype=object)).notna().any():
        group_keys = ["policy_id", "gdpr_article_number"]
    else:
        group_keys = ["segment_id", "gdpr_article_number"]
    def aggregate_labels(labels: List[str]) -> str:
        if all(l == "NOT_APPLICABLE" for l in labels):
            return "NOT_APPLICABLE"
        share_c = sum(l == "COMPLIANT" for l in labels) / len(labels)
        return "COMPLIANT" if share_c > 0.5 else "NON_COMPLIANT"
    agg_rows = []
    for _, grp in pred_pairs.groupby(group_keys):
        gold = aggregate_labels(grp["gold_label"].tolist())
        pred = aggregate_labels(grp["pred_label"].tolist())
        agg_rows.append({"article": int(grp["gdpr_article_number"].iloc[0]), "gold_article_label": gold, "pred_article_label": pred})
    agg_df = pd.DataFrame(agg_rows)
    m2 = macro_metrics(agg_df["gold_article_label"], agg_df["pred_article_label"])
    def per_article_f1(article_ids, y_true, y_pred) -> pd.DataFrame:
        df_tmp = pd.DataFrame({"article": article_ids, "y_true": y_true, "y_pred": y_pred})
        rows = []
        for a, grp in df_tmp.groupby("article"):
            p, r, f1, _ = precision_recall_fscore_support(grp["y_true"], grp["y_pred"], labels=LABELS, average="macro", zero_division=0)
            rows.append({"article": int(a), "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1), "support": int(len(grp))})
        return pd.DataFrame(rows).sort_values("article")
    scores_by_article = per_article_f1(
        article_ids=agg_df["article"].tolist(),
        y_true=agg_df["gold_article_label"].tolist(),
        y_pred=agg_df["pred_article_label"].tolist(),
    )
    macro_over_articles = float("nan") if scores_by_article.empty else float(scores_by_article["macro_f1"].mean())
    metrics_json = {
        "stage1": {"metrics": m1},
        "stage2": {"metrics": m2},
        "stage3": {"macro_f1_across_articles": float(macro_over_articles), "per_article": scores_by_article.to_dict(orient="records")},
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
