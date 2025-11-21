import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Sequence

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


STOPWORDS = [
    "the","and","of","on","to","for","in","at","with","a","an","by","from","over","under","as","is","includes","will","all","this","that","be","per","include","new",
    "existing","work","project","projects","construction","contract","road","roads","state","route","routes","highway","sr","us","county","various","location","locations",
    "proposed","existing","improvements","improvement","design","services","approximately","mile","miles","length","mp","number","known","description","include","including",
]


def read_descriptions(path: Path) -> List[str]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [row["description"] for row in reader]


def normalize_texts(descriptions: Sequence[str]) -> List[str]:
    normed = []
    for d in descriptions:
        text = d or ""
        text = re.sub(r"\s+", " ", text)
        normed.append(text.strip())
    return normed


def build_vectorizer(max_features: int):
    return TfidfVectorizer(
        stop_words=STOPWORDS,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.6,
    )


def topic_keywords(model: NMF, feature_names: List[str], top_n: int) -> List[List[str]]:
    topics = []
    min_len = 3 
    for comp in model.components_:
        indices = comp.argsort()[::-1][:top_n]
        raw_keywords = [feature_names[i] for i in indices]
        filtered = [
            kw
            for kw in raw_keywords
            if len(kw.replace(" ", "")) >= min_len
            and not kw.isdigit()
            and not re.fullmatch(r"[0-9]+(\\.[0-9]+)?", kw)
        ]
        topics.append(filtered[:top_n])
    return topics


def make_topic_id(keywords: List[str], idx: int) -> str:
    if keywords:
        slug = re.sub(r"[^a-z0-9]+", "_", keywords[0].lower()).strip("_")
        if slug:
            return f"{slug}_{idx+1}"
    return f"topic_{idx+1}"


def make_topic_name(keywords: List[str]) -> str:
    if not keywords:
        return "Untitled"
    primary = keywords[0].title()
    secondary = keywords[1].title() if len(keywords) > 1 else ""
    return f"{primary} & {secondary}".strip(" &")


def make_definition(keywords: List[str]) -> str:
    if not keywords:
        return "Automatically discovered topic from proposal descriptions."
    lead = ", ".join(keywords[:5])
    return f"Discovered topic characterized by: {lead}."


def build_taxonomy(descriptions: List[str], num_topics: int, max_features: int, top_keywords: int) -> List[dict]:
    texts = normalize_texts(descriptions)
    vectorizer = build_vectorizer(max_features)
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    model = NMF(
        n_components=num_topics,
        init="nndsvda",
        random_state=42,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        max_iter=600,
    )
    model.fit(tfidf)
    topic_terms = topic_keywords(model, feature_names.tolist(), top_keywords)

    taxonomy = []
    for i, keywords in enumerate(topic_terms):
        taxonomy.append(
            {
                "id": make_topic_id(keywords, i),
                "name": make_topic_name(keywords),
                "definition": make_definition(keywords),
                "keywords": keywords,
            }
        )
    return taxonomy


def write_taxonomy(path: Path, taxonomy: list) -> None:
    path.write_text(json.dumps(taxonomy, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Discover a compact taxonomy from proposal descriptions.")
    parser.add_argument("--input", type=Path, default=Path("inputs/proposals.csv"), help="Input proposals CSV.")
    parser.add_argument("--output", type=Path, default=Path("inputs/taxonomy/taxonomy.json"), help="Path to write the taxonomy JSON.")
    parser.add_argument("--topics", type=int, default=8, help="Number of topics/categories to discover.")
    parser.add_argument("--max-features", type=int, default=2000, help="Max vocabulary size for TF-IDF.")
    parser.add_argument("--top-keywords", type=int, default=10, help="Number of keywords per topic.")
    args = parser.parse_args()

    descriptions = read_descriptions(args.input)
    taxonomy = build_taxonomy(
        descriptions,
        num_topics=args.topics,
        max_features=args.max_features,
        top_keywords=args.top_keywords,
    )
    write_taxonomy(args.output, taxonomy)
    print(f"Wrote taxonomy with {len(taxonomy)} topics to {args.output}")


if __name__ == "__main__":
    main()
