#!/usr/bin/env python
import json
import random
import math
import openai

OPENAI_API_KEY = "REPLACE_WITH_YOUR_KEY"
SOURCE_FILE = "GDPR_Articles_Obligations_Keypoints_Organized.json"
OUTPUT_JSON_PATH = "synthetic_gdpr_corpus_batches.json"
TOTAL_SEGMENTS = 8977
ARTICLES_PER_BATCH = 47
LOG_EVERY = 100
SHOW_PROMPT = 2

GDPR_ARTICLES = [
    "5","6","7","8","9","10","11","12","13","14","15","16","17",
    "18","19","20","21","22","24","25","26","28","29","30","32",
    "33","34","35","36","37","38","39","44","45","46","47","48",
    "49","77","78","79","80","82","89","91","95","96"
]

CONDITIONAL_ARTICLES = {
    "7","8","9","10","11","26","35","36","37","38","39",
    "44","45","46","47","48","49","89","91","95","96"
}


def word_cap(num_kp: int) -> int:
    return 70 if num_kp <= 10 else 120 if num_kp <= 20 else 150


def token_cap(words: int) -> int:
    return math.ceil(words * 1.3) + 10


def ask_gpt(prompt: str, cap_words: int) -> str:
    rsp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=token_cap(cap_words),
        temperature=0.7,
    )
    return rsp.choices[0].message.content.strip()


def pick_label(article_id: str) -> str:
    if article_id in CONDITIONAL_ARTICLES:
        return "NOT_APPLICABLE" if random.random() < 0.5 else random.choice(["COMPLIANT", "NON_COMPLIANT"])
    return random.choice(["COMPLIANT", "NON_COMPLIANT"])


def build_eval(article_id: str, seg_no: int, article_kp: dict) -> dict:
    kps = article_kp[article_id]
    total_kp = len(kps)
    idx_all = list(range(1, total_kp + 1))
    label = pick_label(article_id)
    if label == "NOT_APPLICABLE":
        met_idx, unmet_idx, score = [], [], 0.0
    elif label == "COMPLIANT":
        met_idx = sorted(random.sample(idx_all, random.randint((total_kp // 2) + 1, total_kp)))
        unmet_idx = [i for i in idx_all if i not in met_idx]
        score = len(met_idx) / total_kp
    else:
        met_idx = sorted(random.sample(idx_all, random.randint(0, total_kp // 2)))
        unmet_idx = [i for i in idx_all if i not in met_idx]
        score = len(met_idx) / total_kp
    met_text = "; ".join(kps[i - 1] for i in met_idx) or "N/A"
    unmet_text = "; ".join(kps[i - 1] for i in unmet_idx) or "N/A"
    justification = (
        f"{len(met_idx)} of {total_kp} obligations addressed; compliance score {score:.2f}."
        if label != "NOT_APPLICABLE"
        else "This article does not apply to this organisation or its activities."
    )
    cap_words = word_cap(total_kp)
    prompt = (
        "You are a GDPR compliance assistant. Invent a realistic privacy-policy excerpt (\"segment\") "
        "matching the given GDPR article, compliance label, and key points.\n\n"
        f"gdpr_article_number: {article_id}\n"
        f"label: {label}\n"
        f"compliance_score: {score:.2f}\n"
        f"met_points: {met_text}\n"
        f"unmet_points: {unmet_text}\n"
        f"justification: {justification}\n\n"
        "IMPORTANT – Use NOT_APPLICABLE only for context-dependent articles and only when the organisation’s situation makes the duty irrelevant.\n\n"
        f"Return ONLY the segment (max {cap_words} words). Avoid naming GDPR, article numbers, or real companies."
    )
    if seg_no <= SHOW_PROMPT:
        print(prompt[:350] + " ...")
    segment = ask_gpt(prompt, cap_words)
    return {
        "gdpr_article_number": int(article_id),
        "label": label,
        "compliance_score": round(score, 2),
        "met_points": ", ".join(map(str, met_idx)),
        "unmet_points": ", ".join(map(str, unmet_idx)),
        "justification": justification,
        "segment": segment,
    }


def main() -> None:
    openai.api_key = OPENAI_API_KEY
    with open(SOURCE_FILE, encoding="utf-8") as fh:
        articles = json.load(fh)
    article_kp = {str(a["gdpr_article"]): a["keypoints"] for a in articles}
    batch_count = math.ceil(TOTAL_SEGMENTS / ARTICLES_PER_BATCH)
    batches = []
    seg_ptr = 0
    for b in range(1, batch_count + 1):
        evals = []
        for art in GDPR_ARTICLES:
            if seg_ptr >= TOTAL_SEGMENTS:
                break
            evals.append(build_eval(art, seg_ptr + 1, article_kp))
            seg_ptr += 1
            if seg_ptr % LOG_EVERY == 0:
                print(f"Generated {seg_ptr}/{TOTAL_SEGMENTS} segments")
        batches.append({"batch_id": f"B{b}", "segments": evals})
        if seg_ptr >= TOTAL_SEGMENTS:
            break
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(batches, fh, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT_JSON_PATH}: {len(batches)} batches, {sum(len(b['segments']) for b in batches)} segments")


if __name__ == "__main__":
    main()
