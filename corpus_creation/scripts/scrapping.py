#!/usr/bin/env python3
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://gdpr-info.eu/"
OUTPUT_CSV_PATH = "gdpr_articles_structured.csv"
REQUEST_SLEEP_SECONDS = 0.3

GDPR_ARTICLES = [
    "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "20", "21", "22", "24", "25", "26", "28", "29", "30", "32",
    "33", "34", "35", "36", "37", "38", "39", "44", "45", "46", "47", "48",
    "49", "77", "78", "79", "80", "82", "89", "91", "95", "96"
]


def fetch_article_index(base_url: str) -> pd.DataFrame:
    resp = requests.get(base_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    toc = soup.find("div", class_="liste-inhaltsuebersicht dsgvo")
    rows = []
    for art_div in toc.select("div.artikel"):
        a = art_div.find("a", href=True)
        num = art_div.find("span", class_="nummer").get_text(strip=True).split()[-1]
        rows.append({"GDPR Article": num, "Article URL": a["href"]})
    return pd.DataFrame(rows)


def extract_subparts(tag: Tag, prefix: str = "") -> dict:
    parts = {}
    count = 0
    for li in tag.find_all("li", recursive=False):
        count += 1
        label = f"{prefix}{chr(96 + count)}" if prefix else f"{count}"
        text = li.get_text(" ", strip=True)
        parts[f"Subpart {label}"] = text
        sublist = li.find(["ol", "ul"])
        if sublist:
            sub_parts = extract_subparts(sublist, prefix=label)
            parts.update(sub_parts)
    return parts


def fetch_article_structure(url: str) -> dict:
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    entry = soup.find("div", class_="entry-content")
    if not entry:
        return {}
    subparts = {}
    index = 0
    for child in entry.children:
        if isinstance(child, Tag) and child.name == "div" and "empfehlung-erwaegungsgruende" in child.get("class", []):
            break
        if isinstance(child, Tag) and child.name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                index += 1
                subparts[f"Subpart {index}"] = text
        elif isinstance(child, Tag) and child.name in ["ol", "ul"]:
            nested = extract_subparts(child)
            subparts.update(nested)
    return subparts


def main() -> None:
    toc_df = fetch_article_index(BASE_URL)
    toc_df = toc_df[toc_df["GDPR Article"].isin(GDPR_ARTICLES)]

    cache = {}
    for art, url in toc_df.itertuples(index=False):
        try:
            structure = fetch_article_structure(url)
            cache[art] = structure
            print(f"Article {art} parsed with {len(structure)} subparts")
        except Exception as e:
            print(f"Error with Article {art}: {e}")
            cache[art] = {}
        time.sleep(REQUEST_SLEEP_SECONDS)

    out = pd.DataFrame(
        {
            "GDPR Article": list(cache.keys()),
            "Structured Text JSON": [json.dumps(v, ensure_ascii=False, indent=2) for v in cache.values()],
        }
    )
    out.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(out)} articles to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
