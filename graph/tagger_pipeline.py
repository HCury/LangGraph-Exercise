import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig
from langchain_openai import ChatOpenAI

# Load environment variables -------------------------------------------------
load_dotenv()  # Loads OPENAI_API_KEY from .env if present


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TagResult:
    tag_id: str
    tag_name: str
    hits: List[Tuple[str, int]]


class PipelineState(TypedDict, total=False):
    proposals: List[Dict[str, str]]
    taxonomy: List[dict]
    current_index: int
    current_proposal: Dict[str, str]
    tag_results: List[TagResult]
    confidence: float
    evidence: str
    decision: str
    results: List[Dict[str, str]]
    threshold: float
    needs_refinement: bool
    refinement_attempts: int


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def match_keywords(text: str, kw: str) -> List[int]:
    positions = []
    start = 0
    t = text.lower()
    k = kw.lower()

    while True:
        idx = t.find(k, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(k)

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
    lower = text.lower()

    tag_results: List[TagResult] = []
    all_hits = []

    for entry in taxonomy:
        hits = []
        for kw in entry["keywords"]:
            for pos in match_keywords(lower, kw):
                hits.append((kw, pos))
                all_hits.append((kw, pos))
        if hits:
            tag_results.append(TagResult(entry["id"], entry["name"], hits))

    total_words = sum(len(entry["keywords"]) for entry in taxonomy)
    # NEW CONFIDENCE SCORING
    num_categories = len(tag_results)
    num_taxonomy = len(taxonomy)

    coverage = num_categories / max(1, num_taxonomy)

    distinct_kw = len(set(kw for kw, _ in all_hits))
    density = min(1.0, len(all_hits) / 5)
    diversity = min(1.0, distinct_kw / 5)

    confidence = round(
        0.5 * coverage +
        0.3 * density +
        0.2 * diversity,
        2
    )

    evidence = extract_evidence(text, all_hits)

    return tag_results, confidence, evidence


def decide_publish_llm(llm: ChatOpenAI, tags: List[TagResult], confidence: float, threshold: float) -> str:
    tag_names = ", ".join([t.tag_name for t in tags])

    prompt = f"""
You are an agent deciding if a proposal should be published.

Tags found: {tag_names}
Confidence score: {confidence}
Threshold: {threshold}

Should we PUBLISH or HOLD this proposal?
Respond with exactly one word: "publish" or "hold".
"""

    resp = llm.invoke(prompt)
    text = resp.content.lower()
    return "publish" if "publish" in text else "hold"


def refine_tags_llm(llm: ChatOpenAI, description: str, tag_results: List[TagResult], confidence: float) -> bool:
    tag_names = ", ".join([t.tag_name for t in tag_results])

    prompt = f"""
You are reviewing tagging results.

Description: {description}
Existing tags: {tag_names}
Confidence: {confidence}

Should I refine the tagging (answer "yes") or proceed (answer "no")?
Respond with exactly one word: "yes" or "no".
"""

    resp = llm.invoke(prompt)
    return "yes" in resp.content.lower()


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def ingest_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    cfg = config["configurable"]

    input_path = Path(cfg["input_path"])
    taxonomy_path = Path(cfg["taxonomy_path"])

    with input_path.open() as f:
        proposals = list(csv.DictReader(f))

    taxonomy = load_taxonomy(taxonomy_path)

    return {
        "proposals": proposals,
        "taxonomy": taxonomy,
        "current_index": 0,
        "results": [],
        "refinement_attempts": 0,
    }


def tag_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    idx = state["current_index"]
    proposals = state["proposals"]
    taxonomy = state["taxonomy"]

    if idx >= len(proposals):
        # Nothing left to tag; just pass state through
        return state

    proposal = proposals[idx]
    tags, confidence, evidence = tag_proposal(proposal["description"], taxonomy)

    return {
        **state,
        "current_proposal": proposal,
        "tag_results": tags,
        "confidence": confidence,
        "evidence": evidence,
    }


def agent_decision_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    cfg = config["configurable"]
    threshold = float(cfg.get("publish_threshold", 0.5))
    tags = state["tag_results"]
    confidence = state["confidence"]
    desc = state["current_proposal"]["description"]

    # Only allow ONE refinement attempt per proposal
    if state.get("refinement_attempts", 0) >= 1:
        needs_refinement = False
    else:
        needs_refinement = refine_tags_llm(llm, desc, tags, confidence)

    decision = decide_publish_llm(llm, tags, confidence, threshold)

    return {
        **state,
        "needs_refinement": needs_refinement,
        "decision": decision,
    }


def refinement_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    proposal = state["current_proposal"]
    taxonomy = state["taxonomy"]

    tags, confidence, evidence = tag_proposal(proposal["description"], taxonomy)

    return {
        **state,
        "tag_results": tags,
        "confidence": confidence,
        "evidence": evidence,
        "needs_refinement": False,
        "refinement_attempts": state.get("refinement_attempts", 0) + 1,
    }


def write_result_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    cfg = config["configurable"]
    output_path = Path(cfg["output_path"])

    results = state["results"]
    proposal = state["current_proposal"]

    tag_names = "; ".join(tr.tag_name for tr in state["tag_results"])
    matched_keywords = "; ".join(sorted({kw for tr in state["tag_results"] for kw, _ in tr.hits}))

    results.append(
        {
            "proposalId": proposal["proposalId"],
            "final_tags": tag_names,
            "matched_keywords": matched_keywords,
            "decision": state["decision"],
            "confidence": state["confidence"],
            "evidence": state["evidence"],
        }
    )

    # Write entire results list to CSV on each write (simple, idempotent)
    output_fields = ["proposalId", "final_tags", "matched_keywords", "decision", "confidence", "evidence"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    return {**state, "results": results}


def next_proposal_node(state: PipelineState, config: RunnableConfig) -> PipelineState:
    # Move to next proposal and reset refinement counter for the new one
    return {
        **state,
        "current_index": state["current_index"] + 1,
        "refinement_attempts": 0,
    }


def end_condition(state: PipelineState) -> bool:
    return state["current_index"] >= len(state["proposals"])


# ---------------------------------------------------------------------------
# Build LangGraph Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("tag", tag_node)
    graph.add_node("agent_decision", agent_decision_node)
    graph.add_node("refine", refinement_node)
    graph.add_node("write", write_result_node)
    graph.add_node("next", next_proposal_node)

    graph.add_edge("ingest", "tag")
    graph.add_edge("tag", "agent_decision")

    graph.add_conditional_edges(
        "agent_decision",
        lambda s: "refine" if s.get("needs_refinement", False) else "write",
    )

    graph.add_edge("refine", "tag")
    graph.add_edge("write", "next")

    graph.add_conditional_edges(
        "next",
        lambda s: END if end_condition(s) else "tag",
    )

    graph.set_entry_point("ingest")
    return graph


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic LangGraph tagging pipeline (OpenAI).")
    parser.add_argument("--input", type=Path, default=Path("inputs/proposals.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tagged_results.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("inputs/taxonomy/taxonomy.json"))
    parser.add_argument("--publish-threshold", type=float, default=0.5)
    args = parser.parse_args()

    app = build_graph().compile()
    app.invoke(
        {},
        config={
            "configurable": {
                "input_path": str(args.input),
                "output_path": str(args.output),
                "taxonomy_path": str(args.taxonomy),
                "publish_threshold": args.publish_threshold,
            },
            # Allow enough steps for looping over many proposals
            "recursion_limit": 1000,
        },
    )


if __name__ == "__main__":
    main()
