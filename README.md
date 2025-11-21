# ML Engineer Tagging Exercise

Local, keyword-grounded tagging workflow for the provided proposal descriptions.

## Contents
- `taxonomy.json` — compact taxonomy with names, definitions, and keywords.
- `tagger_pipeline.py` — LangGraph pipeline that ingests the CSV, proposes tags, scores confidence, and makes publish/hold decisions.
- `tagged_results.csv` — machine-readable output (one row per proposal).

## Taxonomy (derived from the sample data)
- **Bridge & Replace** — Bridge replacement or deck rehabilitation projects, often with extensions and approach roadway fixes along river crossings.
- **Pavement & Marking** — Pavement and marking packages covering concrete/asphalt sections with base repair and related drainage items.
- **Resurfacing & Shoulder** — Mill-and-overlay or resurfacing jobs that rebuild shoulders and handle spot repairs across project mileposts.
- **Preservation & Surface** — Surface preservation and maintenance programs across multiple counties, including concrete repairs and patching.
- **River & South** — River corridor work involving grading, paving, and associated river-bridge elements, often with federal participation.
- **Near & Valley** — Location-specific improvements near valley/district boundary lines, typically defined by begin/end markers.
- **Interchange & Replacement** —  Interchange or roadway replacements with milling tied to specific mileposts and connecting streets
- **Creek & Town** — Creek/town corridor improvements involving drainage, lane work, and bridge-related components.

## LangGraph workflow
Graph: `ingest → tag → write → END`
- **ingest**: read `proposals.csv` and `taxonomy.json` into state.
- **tag**: keyword match per taxonomy entry, compute confidence, decide publish/hold using the threshold.
- **write**: emit `tagged_results.csv` with `proposalId`, `final_tags`, `decision`, `confidence`, and an evidence snippet.

## Confidence:
How is confidence determined?:
The current algoirthm to determine confidence is as follows: The (number of tags we matched / the number of taxonomies categories we have) + (all the tag hits / the total number of possible words) then we take the average.
`confidence = round(min(1.0, (((len(tag_results)) / 8) + (len(all_hits) / total_words)) / 2), 2)`

## Threhold:
With the confidence now set, we need a threshold of confidence to determine what should be approved and rejected. We do this by selecting the 60th Quantile of the sorted confidences. Anything above that confidence number is approved.
Anything below that number is on hold. This makes our threshold dynamic to our data for scalability.

## Running it'
You will need to pip install the required packages. You can do this by doing:
```bash
pip install -r requirements.txt
```
Once installed you can rune the pipeline: 
```bash
python graph/tagger-pipeline.py
```

## Notes and limitations
- The workflow is deterministic and local (no external services or keys).
- Keyword matching is intentionally simple to keep decisions inspectable. Very terse descriptions without matching terms will be held.
- To add tags, update `taxonomy.json`; the script will automatically enforce the new taxonomy on the next run.
