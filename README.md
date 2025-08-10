# ML Crypto Signal App

FastAPI app for high-precision crypto signals. Fetches OHLCV, engineers features (EMA/RSI/MACD/ATR/VWAP, Bollinger, candles, regime), uses ATR-based triple-barrier labeling and LightGBM with purged walk-forward CV + isotonic calibration. Threshold search, OOF backtest, live polling, Telegram alerts, simple dashboard. Paper trading first. NFA. Beta.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:8000
```
