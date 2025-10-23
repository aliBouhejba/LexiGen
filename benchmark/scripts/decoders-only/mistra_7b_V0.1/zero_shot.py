import os, json, re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = "/content/outputs"
SEGMENTS_FILES = ["/content/raw_eval.json"]
KEYPOINTS_FILES = ["/content/GDPR_articles_obligations_keypoints_organized.json"]
BASE_REPO = "mistralai/Mistral-7B-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 224
GEN_TEMPERATURE = 0.4
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOK = 24
LABELS = ["COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"]
VALID = set(LABELS)
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.I)

os.environ["WANDB_DISABLED"] = "true"
os.makedirs(OUT_DIR, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

def free_cuda():
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

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
            rows.append({
                "gdpr_article_number": int(s["gdpr_article_number"]),
                "segment": str(s["segment"]),
                "label": str(s["label"]).strip().upper(),
                "met_points": s.get("met_points", ""),
                "unmet_points": s.get("unmet_points", ""),
                "policy_id": s.get("policy_id")
            })
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

def parse_index_list(s: Any) -> set:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {int(x.strip()) for x in s.split(",") if x.strip().isdigit()}

def expand_pairs(segments_df: pd.DataFrame, keypoints_map: Dict[int, List[str]]) -> pd.DataFrame:
    expanded = []
    for _, r in segments_df.iterrows():
        art = int(r["gdpr_article_number"])
        kps = keypoints_map.get(art)
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
            expanded.append({
                "segment_id": int(r["segment_id"]),
                "policy_id": r.get("policy_id"),
                "gdpr_article_number": art,
                "keypoint_index": idx,
                "keypoint_text": kps[idx - 1],
                "segment": r["segment"],
                "gold_label": gold
            })
    df = pd.DataFrame(expanded)
    if df.empty:
        raise ValueError("No (segment, keypoint) pairs after expansion.")
    return df

def build_messages(excerpt: str, art: int, kp_id: str, kp_text: str) -> List[Dict[str, str]]:
    msg = (
        "You are a GDPR compliance assistant. Analyse ONLY the key obligation below.\n\n"
        f"Article {art} – Key obligation ({kp_id}):\n"
        f"- {kp_text}\n\n"
        "Policy excerpt to evaluate:\n"
        '"""\n' + excerpt + '\n"""\n\n'
        "Decide whether this excerpt satisfies this specific obligation.\n\n"
        "Definition:\n"
        "- NOT_APPLICABLE: The obligation is irrelevant to this excerpt’s context.\n\n"
        'Respond ONLY with: { "label": "<COMPLIANT|NON_COMPLIANT|NOT_APPLICABLE>" }'
    )
    return [{"role": "user", "content": msg}]

def encode_one(tok, messages: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = enc
        attn_mask = torch.ones_like(input_ids)
    except Exception:
        txt = f"<s>[INST] {messages[0]['content']} [/INST]"
        enc = tok(txt, return_tensors="pt")
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
    return input_ids.squeeze(0), attn_mask.squeeze(0)

def extract_label(txt: str) -> str:
    try:
        lab = json.loads(txt).get("label", "").strip().upper()
    except Exception:
        m = LABEL_RE.search(txt or "")
        lab = m.group(1).upper() if m else "UNKNOWN"
    return lab if lab in VALID else "NOT_APPLICABLE"

def generate_batched(model, tok, rows: pd.DataFrame, batch_size: int) -> List[str]:
    preds: List[str] = []
    cache_ids: List[torch.Tensor] = []
    cache_masks: List[torch.Tensor] = []
    def flush():
        nonlocal preds, cache_ids, cache_masks
        if not cache_ids:
            return
        ids = pad_sequence(cache_ids, batch_first=True, padding_value=tok.pad_token_id)
        att = pad_sequence(cache_masks, batch_first=True, padding_value=0)
        ids = ids.to(DEVICE, non_blocking=True)
        att = att.to(DEVICE, non_blocking=True)
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
                use_cache=True
            )
        prompt_len = ids.shape[1]
        out = out.to("cpu")
        for bi in range(out.size(0)):
            gen = out[bi, prompt_len:]
            text = tok.decode(gen, skip_special_tokens=True)
            preds.append(extract_label(text))
        del ids, att, out
        cache_ids.clear(); cache_masks.clear()
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="generation"):
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

def macro_metrics(y_true, y_pred, labels=LABELS) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"Macro-P": float(p), "Macro-R": float(r), "Macro-F1": float(f1), "Accuracy": float(acc)}

def save_confusion(y_true, y_pred, labels, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=220)
    plt.close()
    return cm.tolist()

def per_article_f1(article_ids, y_true, y_pred) -> pd.DataFrame:
    df_tmp = pd.DataFrame({"article": article_ids, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for a, grp in df_tmp.groupby("article"):
        p, r, f1, _ = precision_recall_fscore_support(grp["y_true"], grp["y_pred"], labels=LABELS, average="macro", zero_division=0)
        rows.append({"article": int(a), "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1), "support": int(len(grp))})
    return pd.DataFrame(rows).sort_values("article")

def save_per_article_bar(df_scores: pd.DataFrame, fname_png: str, title: str):
    if df_scores.empty:
        return None
    df_sorted = df_scores.sort_values("article")
    fig_h = max(6, 0.18 * len(df_sorted))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    y = df_sorted["article"].astype(str).values
    x = df_sorted["macro_f1"].values
    ax.barh(y, x)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Per-Article Macro-F1")
    ax.set_ylabel("GDPR Article")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname_png)
    plt.savefig(path, dpi=220)
    plt.close()
    return path

def main():
    segments_df = load_segments_flat(SEGMENTS_FILES)
    keypoints_map = load_keypoints(KEYPOINTS_FILES)
    pairs_df = expand_pairs(segments_df, keypoints_map)
    tok = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
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
    model = AutoModelForCausalLM.from_pretrained(BASE_REPO, load_in_8bit=True, device_map="auto")
    model.eval(); model.config.use_cache = True
    preds = generate_batched(model, tok, pairs_df, BATCH_SIZE)
    pred_pairs = pairs_df.copy()
    pred_pairs["pred_label"] = preds
    m1 = macro_metrics(pred_pairs["gold_label"], pred_pairs["pred_label"])
    cm1 = save_confusion(pred_pairs["gold_label"], pred_pairs["pred_label"], LABELS, "stage1_confusion.png")
    if pred_pairs["policy_id"].notna().any():
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
    cm2 = save_confusion(agg_df["gold_article_label"], agg_df["pred_article_label"], LABELS, "stage2_confusion.png")
    scores_by_article = per_article_f1(
        article_ids=agg_df["article"].tolist(),
        y_true=agg_df["gold_article_label"].tolist(),
        y_pred=agg_df["pred_article_label"].tolist()
    )
    macro_over_articles = float("nan") if scores_by_article.empty else float(scores_by_article["macro_f1"].mean())
    save_per_article_bar(scores_by_article, "stage3_per_article_f1.png", "Per-Article Macro-F1")
    metrics_json = {
        "stage1": {"metrics": m1, "confusion": cm1},
        "stage2": {"metrics": m2, "confusion": cm2},
        "stage3": {"macro_f1_across_articles": float(macro_over_articles), "per_article": scores_by_article.to_dict(orient="records")}
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    free_cuda()

if __name__ == "__main__":
    main()
