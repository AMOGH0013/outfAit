# Wardrobe Recommendation System (MVP)

FastAPI backend + server-served dev UI for a wardrobe/outfit recommender:

- Scan clothing image → (optional SAM mask) → color palette (CV) + CLIP embedding
- Wardrobe CRUD + manual semantics correction (category/item_type)
- Outfit generation with explainable scoring (rules + feedback + diversity + fit)
- Offline evaluation harness (`evaluation/`) with synthetic users (A/B baseline vs personalized)

## Quickstart

1) Install deps:

```bash
pip install -r requirements.txt
```

2) Run the API + dev UI:

```bash
uvicorn app.main:app --reload
```

Open:
- Dev UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`

## SAM weights (not committed)

This repo expects a SAM ViT-B checkpoint at:

- `sam_models/sam_vit_b.pth`

The `.pth` file is intentionally ignored by git because it is large.

## Offline evaluation (synthetic)

Run the simulator from the repo root, for example:

```bash
python -m evaluation.simulate_users --days 100 --replicates 5
python -m evaluation.report
```
