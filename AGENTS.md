# AGENTS.md — ML Crypto Signal App

## Objective
Turn this repository into a production-ready FastAPI app that trains a LightGBM model with triple-barrier labeling and serves:
- /api/train, /api/oof, /api/live, /api/telegram
- Minimal dashboard at "/"

## Tech
- Python 3.10+, FastAPI, Uvicorn
- ccxt, pandas, numpy, scikit-learn, lightgbm, joblib
- pytest + coverage
- Docker & docker-compose
- GitHub Actions CI

## Setup (run in sandbox)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q

## Tasks
1. Refactor `app.py` into a package:
   - src/app/__init__.py, api/routes.py, ml/features.py, ml/labeling.py, ml/model.py, services/data.py, core/config.py
2. Add tests:
   - tests/test_routes.py (200/JSON shape for /api/*)
   - tests/test_labeling.py (triple-barrier edge cases)
   - tests/test_features.py (no NaNs, shapes)
3. Add Docker and compose:
   - Dockerfile (slim, non-root, uvicorn)
   - docker-compose.yml (app + volume for ./models)
4. CI:
   - .github/workflows/ci.yml: lint (ruff), type-check (mypy optional), pytest with coverage gate (>=80%)
5. Robustness:
   - Rate-limit/backoff on OHLCV fetch
   - Config via env (fees, timeframe, symbols)
   - Save artifacts under ./models
6. Docs:
   - README.md: local run, Docker, CI, environment vars, safety note (“Paper trading first — not financial advice”)

## Commands
- Run: uvicorn app.main:app --host 0.0.0.0 --port 8000
- Tests: pytest -q
- Build: docker build -t ml-crypto-signal-app .
- Compose: docker compose up --build

## Acceptance Criteria
- All endpoints return 200 and expected schema
- CI passes on clean clone
- `codex` can run tests in sandbox and show passing output
- OOF report saved to models/oof_report.json after /api/train
