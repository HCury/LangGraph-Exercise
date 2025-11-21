import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

# Helper structures ----------------------------------------------------------


@dataclass
class TagResult:
    tag_id: str
    tag_name: str
    hits: List[Tuple[str, int]]


class PipelineState(TypedDict, total=False):
    proposals: List[Dict[str, str]]
    taxonomy: List[dict]
    results: List[Dict[str, str]]


# Core logic -----------------------------------------------------------------


def load_taxonomy(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def match_keywords(text: str, keyword: str) -> List[int]:
    positions = []
    start = 0
    lower_text = text.lower()
    lower_kw = keyword.lower()
    while True:
        idx = lower_text.find(lower_kw, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(lower_kw)
    return positions


def extract_evidence(text: str, positions: List[Tuple[str, int]]) -> str:
    if not positions:
        return ""
    keyword, pos = sorted(positions, key=lambda x: x[1])[0]
    start = max(0, pos - 60)
    end = min(len(text), pos + 80)
    snippet = text[start:end].strip()
    return f"Matched '{keyword}' ... {snippet}"


def tag_proposal(description: str, taxonomy: List[dict]) -> Tuple[List[TagResult], float, str]:
    text = description or ""
    lower_text = text.lower()
    tag_results: List[TagResult] = []
    all_hits: List[Tuple[str, int]] = []

    for entry in taxonomy:
        hits_for_entry: List[Tuple[str, int]] = []
        for kw in entry["keywords"]:
            for pos in match_keywords(lower_text, kw):
                hits_for_entry.append((kw, pos))
                all_hits.append((kw, pos))
        if hits_for_entry:
            tag_results.append(TagResult(entry["id"], entry["name"], hits_for_entry))

    total_words = sum(len(entry["keywords"]) for entry in taxonomy)
    confidence = round(min(1.0, (((len(tag_results)) / 8) + (len(all_hits) / total_words)) / 2), 2)
    evidence = extract_evidence(text, all_hits)
    return tag_results, confidence, evidence


def infer_threshold(confidences: List[float], quantile: float, floor: float = 0.35, ceiling: float = 1.0) -> float:
    """Pick a data-driven threshold from confidence samples at the given quantile."""
    if not confidences:
        return floor
    sorted_vals = sorted(confidences)
    idx = int(min(len(sorted_vals) - 1, max(0, quantile * (len(sorted_vals) - 1))))
    value = sorted_vals[idx]
    return float(max(floor, min(ceiling, value)))


def decide_publish(tags: List[TagResult], confidence: float, threshold: float) -> str:
    if tags and confidence >= threshold:
        return "publish"
    return "hold"


# LangGraph nodes ------------------------------------------------------------


def ingest_node(state: PipelineState, config: RunnableConfig | None = None) -> PipelineState:
    config = config or {}
    input_path = Path(config.get("input_path", "inputs/proposals.csv"))
    taxonomy_path = Path(config.get("taxonomy_path", "inputs/taxonomy/taxonomy.json"))

    taxonomy = load_taxonomy(taxonomy_path)
    with input_path.open() as f:
        reader = csv.DictReader(f)
        proposals = list(reader)

    return {"proposals": proposals, "taxonomy": taxonomy}


def tag_node(state: PipelineState, config: RunnableConfig | None = None) -> PipelineState:
    config = config or {}
    threshold = config.get("publish_threshold")
    infer = bool(config.get("infer_threshold", False))
    quantile = float(config.get("threshold_quantile", 0.6))
    taxonomy = state["taxonomy"]
    proposals = state["proposals"]
    results: List[Dict[str, str]] = []
    confidences_for_threshold: List[float] = []

    tagged = []
    for row in proposals:
        tags, confidence, evidence = tag_proposal(row["description"], taxonomy)
        if tags:
            confidences_for_threshold.append(confidence)
        tagged.append((row, tags, confidence, evidence))

    if threshold is None or infer:
        threshold = infer_threshold(confidences_for_threshold, quantile=quantile)
        print(threshold)
    else:
        threshold = float(threshold)

    for row, tags, confidence, evidence in tagged:
        decision = decide_publish(tags, confidence, threshold)
        tag_names = "; ".join(tr.tag_name for tr in tags)
        matched_keywords = "; ".join(sorted({kw for tr in tags for kw, _ in tr.hits}))
        results.append(
            {
                "proposalId": row["proposalId"],
                "final_tags": tag_names,
                "matched_keywords": matched_keywords,
                "decision": decision,
                "confidence": confidence,
                "evidence": evidence,
            }
        )

    return {"results": results}


def write_output_node(state: PipelineState, config: RunnableConfig | None = None) -> PipelineState:
    config = config or {}
    output_path = Path(config.get("output_path", "outputs/tagged_results.csv"))
    output_fields = ["proposalId", "final_tags", "matched_keywords", "decision", "confidence", "evidence"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter=",")
        writer.writeheader()
        writer.writerows(state["results"])

    return {}


# Graph assembly -------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("tag", tag_node)
    graph.add_node("write", write_output_node)

    graph.add_edge("ingest", "tag")
    graph.add_edge("tag", "write")
    graph.add_edge("write", END)
    graph.set_entry_point("ingest")
    return graph


def main():
    parser = argparse.ArgumentParser(description="LangGraph tagging pipeline.")
    parser.add_argument("--input", type=Path, default=Path("inputs/proposals.csv"), help="Input proposals CSV")
    parser.add_argument("--output", type=Path, default=Path("outputs/tagged_results.csv"), help="Output CSV path")
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("inputs/taxonomy/taxonomy.json"), help="Taxonomy definition file (JSON)"
    )
    parser.add_argument(
        "--infer-threshold",
        action="store_true",
        help="Infer publish threshold from confidence distribution instead of using a fixed value.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.6,
        help="Quantile of confidence scores to use when inferring threshold (0..1).",
    )
    parser.add_argument(
        "--publish-threshold",
        type=float,
        default=None,
        help="Fixed confidence required to publish; otherwise the proposal is held. Leave unset to infer from data.",
    )
    args = parser.parse_args()

    app = build_graph().compile()
    app.invoke(
        {},
        config={
            "input_path": args.input,
            "output_path": args.output,
            "taxonomy_path": args.taxonomy,
            "publish_threshold": args.publish_threshold,
            "infer_threshold": args.infer_threshold,
            "threshold_quantile": args.threshold_quantile,
        },
    )


if __name__ == "__main__":
    main()
