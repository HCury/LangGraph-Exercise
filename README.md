# ML Engineer Tagging Exercise

Local, keyword-grounded tagging workflow for the provided proposal descriptions.

## Contents
- `taxonomy.json` — compact taxonomy with names, definitions, and keywords.
- `tagger.py` — LangGraph pipeline that ingests the CSV, proposes tags, scores confidence, and makes publish/hold decisions.
- `tagged_results.csv` — machine-readable output (one row per proposal).

## Taxonomy (derived from the sample data)
- **Bridges & Structures** — Bridge replacement or rehab, culverts, structural repairs, painting, and approach work.
- **Roadway Rehabilitation** — Resurfacing, mill-and-overlay, reconstruction, and pavement preservation.
- **Capacity Expansion & Widening** — Adding lanes, roundabouts, new alignments, or other widening work.
- **Operations & Maintenance** — Routine upkeep such as mowing, striping contracts, cleaning/painting, and traffic maintenance.
- **Multimodal & Active Transportation** — Pedestrian/bicycle/shared-use facilities (sidewalks, paths, tunnels).
- **Drainage & Flood Control** — Drainage, stormwater, flooding, storm sewer, culverts, erosion protection.
- **Transport Facilities & Buildings** — Transportation-related facilities/buildings (terminals, stations, tunnels, civic square preservation).
- **Signals, ITS & Lighting** — ITS, signals, lighting, and communications upgrades.

## LangGraph workflow
Graph: `ingest → tag → write → END`
- **ingest**: read `proposals.csv` and `taxonomy.json` into state.
- **tag**: keyword match per taxonomy entry, compute confidence, decide publish/hold using the threshold.
- **write**: emit `tagged_results.csv` with `proposalId`, `final_tags`, `decision`, `confidence`, and an evidence snippet.

## Running it
```bash
python tagger.py --input proposals.csv --output tagged_results.csv --publish-threshold 0.35
```
Adjust `--publish-threshold` (higher = stricter) or edit `taxonomy.json` keywords/definitions to tune behavior.

## Notes and limitations
- The workflow is deterministic and local (no external services or keys).
- Keyword matching is intentionally simple to keep decisions inspectable. Very terse descriptions without matching terms will be held.
- To add tags, update `taxonomy.json`; the script will automatically enforce the new taxonomy on the next run.
