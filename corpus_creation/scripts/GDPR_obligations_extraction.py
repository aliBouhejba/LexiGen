#!/usr/bin/env python
import json
import time
import textwrap
from pathlib import Path
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

JSON_IN = "/Users/med/PycharmProjects/OPP115/Data/GDPR_Articles/gdpr_articles.json"
JSON_OUT = "/Users/med/PycharmProjects/OPP115/Data/GDPR_Articles/GDPR_Articles_Obligations_Keypoints.json"
OPENAI_API_KEY = "REPLACE_WITH_YOUR_KEY"
MODEL = "gpt-4.1"
TEMPERATURE = 0
MAX_TOKENS = 24000
PAUSE_SEC = 0.8

PROMPT_TEMPLATE = textwrap.dedent("""
You are a legal analysis assistant specialised in EU data-protection law.

Your task is to extract all distinct legal obligations from the following
GDPR article and to rewrite them as a clear, ordered list of bullet points.
Each bullet must capture a single stand-alone requirement.

Rules:
• Preserve the legal nuance (conditions, scope, subjects).
• Do not paraphrase away legal language; stay precise.
• One bullet per obligation. For nested obligations, use indentation.

Input:
---
GDPR Article {article_number}:
{article_text}
---

Output:
A bullet-point summary of the legal obligations in Article {article_number}:
""").strip()


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def gpt_bullets(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)
    articles = json.loads(Path(JSON_IN).read_text("utf-8"))
    output = []
    for art in articles:
        num = int(art["article_num"])
        text = art["text"].strip()
        try:
            bullets = gpt_bullets(client, PROMPT_TEMPLATE.format(article_number=num, article_text=text))
        except Exception as exc:
            bullets = f"ERROR: {exc}"
        print(f"Article {num} processed.")
        lines = [ln.strip(" -•*") for ln in bullets.splitlines() if ln.strip()]
        structured = {}
        index = 1
        i = 0
        current_intro = None
        while i < len(lines):
            line = lines[i]
            if line.endswith(":"):
                current_intro = line
                i += 1
                continue
            if current_intro:
                full_line = f"{current_intro} {line}"
                structured[str(index)] = full_line
            else:
                structured[str(index)] = line
            index += 1
            i += 1
        output.append(
            {
                "GDPR Article": num,
                "Structured Text JSON": json.dumps(structured, ensure_ascii=False),
            }
        )
        time.sleep(PAUSE_SEC)
    Path(JSON_OUT).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {JSON_OUT}")


if __name__ == "__main__":
    main()
