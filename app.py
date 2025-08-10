from __future__ import annotations
import os
import math
import json
import time
import dataclasses as dc
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import ccxt

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from joblib import dump, load

# =====================
# Config
# =====================
@dc.dataclass
class Config:
    exchange_primary: str = "bybit"       # public OHLCV endpoints are fine
    exchange_fallback: str = "binanceusdm"
    symbols: Tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "15m"
    days_history: int = 365
    fetch_limit: int = 1500

    # Costs
    fee_per_side: float = 0.0006
    slippage_frac: float = 0.0004

    # Labeling
    horizon_bars: int = 16
    atr_len: int = 14
    up_mult: float = 1.2
    dn_mult: float = 1.2
    deadband_bps: float = 5

    # Model
    learning_rate: float = 0.03
    num_leaves: int = 64
    n_estimators: int = 1200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 100
    n_splits: int = 5

    # Threshold search
    opt_metric: str = "precision"
    min_prob: float = 0.60

    # Live
    poll_seconds: int = 30

    # Paths
    model_dir: str = "models"
    model_file: str = "models/lgbm_agent.txt"
    calib_file: str = "models/calibrator.joblib"
    threshold_file: str = "models/thresholds.json"
    oof_report_file: str = "models/oof_report.json"

CFG = Config()

# =====================
# Indicators / features
# =====================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = ema(series, fast)
    es = ema(series, slow)
    m = ef - es
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr

def atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    return true_range(h,l,c).ewm(alpha=1/length, adjust=False).mean()

def vwap(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
    typical = (h + l + c) / 3
    cum_pv = (typical * v).cumsum()
    cum_v = v.cumsum() + 1e-12
    return cum_pv / cum_v

FEATURE_COLS = [
    "ret_1","ret_3","ret_6","ret_12","vol_chg_6",
    "ema20","ema50","ema200","rsi14","macd","macd_sig","macd_hist",
    "atr14","vwap","vwap_dist","bb_width","body_pct","upper_wick","lower_wick","trend_up"
]

# =====================
# Data fetching
# =====================
def ex_by_name(name: str):
    return getattr(ccxt, name)({"enableRateLimit": True})

def fetch_ohlcv_paginated(ex, symbol: str, timeframe: str, days: int, limit: int) -> pd.DataFrame:
    ms_per_day = 24*60*60*1000
    since = ex.milliseconds() - days*ms_per_day
    rows = []
    while True:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not o:
            break
        rows += o
        since = o[-1][0] + 1
        if len(o) < limit:
            break
        time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df.astype(float)

def load_all_symbols(cfg: Config) -> pd.DataFrame:
    pri = ex_by_name(cfg.exchange_primary)
    frames = []
    for sym in cfg.symbols:
        try:
            df = fetch_ohlcv_paginated(pri, sym, cfg.timeframe, cfg.days_history, cfg.fetch_limit)
        except Exception:
            fall = ex_by_name(cfg.exchange_fallback)
            df = fetch_ohlcv_paginated(fall, sym, cfg.timeframe, cfg.days_history, cfg.fetch_limit)
        df["symbol"] = sym
        frames.append(df)
    data = pd.concat(frames).sort_index()
    return data

# =====================
# Features & labels
# =====================
def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_6"] = out["close"].pct_change(6)
    out["ret_12"] = out["close"].pct_change(12)
    out["vol_chg_6"] = out["volume"].pct_change(6)

    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["macd"], out["macd_sig"], out["macd_hist"] = macd(out["close"], 12, 26, 9)
    out["atr14"] = atr(out["high"], out["low"], out["close"], cfg.atr_len)
    out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    out["vwap_dist"] = (out["close"] - out["vwap"]) / out["close"]

    m20 = out["close"].rolling(20).mean()
    s20 = out["close"].rolling(20).std(ddof=0)
    out["bb_width"] = (2 * s20) / (m20 + 1e-12)

    body = (out["close"] - out["open"]).abs()
    rng = (out["high"] - out["low"]) + 1e-12
    out["body_pct"] = body / rng
    out["upper_wick"] = (out["high"] - out[["close","open"]].max(axis=1)) / rng
    out["lower_wick"] = (out[["close","open"]].min(axis=1) - out["low"]) / rng

    out["trend_up"] = (out["ema50"] > out["ema200"]).astype(int)
    return out.dropna()

def compute_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["atr14"] = atr(out["high"], out["low"], out["close"], cfg.atr_len)
    labels = []
    idx = out.index
    H = cfg.horizon_bars
    for i in range(len(out)):
        if i + H >= len(out):
            labels.append(np.nan)
            continue
        c0 = out["close"].iloc[i]
        a0 = out["atr14"].iloc[i]
        up = c0 + cfg.up_mult * a0
        dn = c0 - cfg.dn_mult * a0
        hi = out["high"].iloc[i+1:i+1+H].max()
        lo = out["low"].iloc[i+1:i+1+H].min()
        if hi >= up and lo <= dn:
            sub = out.iloc[i+1:i+1+H]
            t_up = sub.index[(sub["high"] >= up)].min()
            t_dn = sub.index[(sub["low"] <= dn)].min()
            label = 1 if t_up <= t_dn else -1
        elif hi >= up:
            label = 1
        elif lo <= dn:
            label = -1
        else:
            cT = out["close"].iloc[i+H]
            ret = (cT - c0)/c0
            if abs(ret) * 1e4 < cfg.deadband_bps:
                label = 0
            else:
                label = 1 if ret > 0 else -1
        labels.append(label)
    out["label"] = labels
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out

# =====================
# Dataset
# =====================
def build_dataset(cfg: Config):
    raw = load_all_symbols(cfg)
    frames = []
    for sym, g in raw.groupby("symbol"):
        f = add_features(g.drop(columns=["symbol"]), cfg)
        f["symbol"] = sym
        frames.append(f)
    feat = pd.concat(frames).sort_index()

    framesL = []
    for sym, g in raw.groupby("symbol"):
        L = compute_labels(g.drop(columns=["symbol"]), cfg)
        L["symbol"] = sym
        framesL.append(L[["label","symbol"]])
    lab = pd.concat(framesL).sort_index()

    ds = feat.join(lab, how="inner")

    sym_d = pd.get_dummies(ds["symbol"], prefix="sym")
    X = pd.concat([ds[FEATURE_COLS], sym_d], axis=1).replace([np.inf,-np.inf], np.nan).dropna()
    y = ds.loc[X.index, "label"].astype(int)
    return X, y, ds.loc[X.index]

# =====================
# CV & training
# =====================
def purged_time_series_splits(n_samples: int, n_splits: int, embargo: int):
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = fold_size * (k+1)
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > n_samples:
            break
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        splits.append((tr, te))
    return splits

def fit_model_cv(X: pd.DataFrame, y: pd.Series, cfg: Config):
    n = len(X)
    embargo = cfg.horizon_bars
    splits = purged_time_series_splits(n, cfg.n_splits, embargo)

    base = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=42,
        n_jobs=-1,
    )

    oof_proba = np.zeros((n, 3))
    metrics = []

    for (tr, te) in splits:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        model = base
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(stopping_rounds=CFG.early_stopping_rounds, verbose=False)])
        proba = model.predict_proba(Xte)
        oof_proba[te] = proba
        yhat = proba.argmax(1) - 1  # map back to {-1,0,1}
        acc = accuracy_score(yte.map({-1:0,0:1,1:2}), yhat+1)
        prec = precision_score(yte, np.array([-1,0,1])[yhat], average="macro", zero_division=0)
        rec  = recall_score(yte,   np.array([-1,0,1])[yhat], average="macro", zero_division=0)
        f1   = f1_score(yte,       np.array([-1,0,1])[yhat], average="macro", zero_division=0)
        metrics.append((acc,prec,rec,f1))

    # final fit on all data
    final = base
    final.fit(X, y)

    os.makedirs(cfg.model_dir, exist_ok=True)
    final.booster_.save_model(cfg.model_file)

    calib = CalibratedClassifierCV(final, cv="prefit", method="isotonic")
    calib.fit(X, y)
    dump(calib, cfg.calib_file)

    df_oof = pd.DataFrame(oof_proba, index=X.index, columns=["p_dn","p_flat","p_up"]).replace([np.inf,-np.inf], np.nan).dropna()
    y_aligned = y.loc[df_oof.index]
    thr = optimize_thresholds(df_oof, y_aligned, cfg)

    import numpy as _np
    m = _np.array(metrics)
    cv_summary = {
        "folds": int(len(metrics)),
        "acc_mean": float(m[:,0].mean()) if len(metrics) else None,
        "prec_mean": float(m[:,1].mean()) if len(metrics) else None,
        "rec_mean": float(m[:,2].mean()) if len(metrics) else None,
        "f1_mean": float(m[:,3].mean()) if len(metrics) else None,
    }
    return final, calib, thr, df_oof, y_aligned, cv_summary

# =====================
# Thresholds & backtest
# =====================
def proba_to_signal(df: pd.DataFrame, p_up_min: float, p_dn_min: float) -> pd.Series:
    sig = pd.Series(0, index=df.index)
    sig = sig.mask(df["p_up"] >= p_up_min, 1)
    sig = sig.mask(df["p_dn"] >= p_dn_min, -1)
    return sig

def optimize_thresholds(proba_df: pd.DataFrame, y: pd.Series, cfg: Config) -> Dict[str, float]:>
    best = {"p_up": cfg.min_prob, "p_dn": cfg.min_prob, "score": -1.0}
    for pu in np.linspace(0.55, 0.85, 7):
        for pdn in np.linspace(0.55, 0.85, 7):
            sig = proba_to_signal(proba_df, pu, pdn)
            mask = sig != 0
            if mask.sum() < 50:
                continue
            if cfg.opt_metric == "precision":
                sc = precision_score(y[mask], sig[mask], average="macro", zero_division=0)
            elif cfg.opt_metric == "f1":
                sc = f1_score(y[mask], sig[mask], average="macro", zero_division=0)
            else:
                sc = accuracy_score(y[mask], sig[mask])
            if sc > best["score"]:
                best = {"p_up": float(pu), "p_dn": float(pdn), "score": float(sc)}
    with open(CFG.threshold_file, "w") as f:
        json.dump(best, f)
    return best

def simple_backtest(y_true: pd.Series, sig: pd.Series, ds: pd.DataFrame, cfg: Config) -> Dict[str,float]:
    idx = sig.index.intersection(y_true.index)
    y_true = y_true.loc[idx]
    sig = sig.loc[idx]
    close = ds.loc[idx, "close"]

    H = cfg.horizon_bars
    rets = []
    for i, t in enumerate(idx):
        j = ds.index.get_loc(t)
        if j + H >= len(ds.index):
            continue
        c0 = close.iloc[i]
        cT = ds["close"].iloc[j+H]
        if sig.iloc[i] == 1:
            r = (cT - c0)/c0
        elif sig.iloc[i] == -1:
            r = (c0 - cT)/c0
        else:
            continue
        cost = 2*(cfg.fee_per_side + cfg.slippage_frac)
        rets.append(r - cost)
    if not rets:
        return {"trades": 0, "winrate": 0.0, "avg_ret": 0.0, "sharpe_like": 0.0}
    rets = np.array(rets)
    return {
        "trades": int(len(rets)),
        "winrate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "sharpe_like": float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(365*24*4/H)),
    }

# =====================
# Model I/O
# =====================
def load_model(cfg: Config):
    booster = lgb.Booster(model_file=cfg.model_file)
    model = lgb.LGBMClassifier()
    model._Booster = booster
    calib = load(cfg.calib_file)
    with open(cfg.threshold_file, "r") as f:
        thr = json.load(f)
    return model, calib, thr

# =====================
# FastAPI app
# =====================
app = FastAPI(title="ML Crypto Signal App", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, object] = {"last_live": None}

class TrainRequest(BaseModel):
    symbols: List[str] | None = None
    timeframe: str | None = None
    days_history: int | None = None

@app.get("/")
def index() -> HTMLResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ML Crypto Signal App</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"> 
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0d12;color:#e8eef9;margin:0}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}
    .row{display:flex;gap:16px;flex-wrap:wrap}
    .card{background:#121623;border:1px solid #1e2437;border-radius:16px;padding:16px;flex:1;min-width:260px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    h1{font-size:22px;margin:0 0 12px 0}
    h2{font-size:16px;margin:0 0 8px 0;color:#9fb3ff}
    button{background:#4f7cff;border:none;padding:10px 14px;border-radius:10px;color:#fff;font-weight:600;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .kpi{font-size:28px;font-weight:800}
    .muted{color:#94a3b8;font-size:12px}
    .green{color:#4ade80} .red{color:#f87171} .gray{color:#94a3b8}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{padding:8px;border-bottom:1px solid #1e2437;text-align:left}
    .row > .card.small{flex:0 0 260px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ML Crypto Signal App</h1>
    <div class="row">
      <div class="card small">
        <h2>Actions</h2>
        <button id="trainBtn" onclick="train()">Train model</button>
        <button onclick="refreshLive()">Refresh live</button>
        <div class="muted" style="margin-top:8px">Paper trading first. Not financial advice.</div>
      </div>
      <div class="card">
        <h2>Latest signal</h2>
        <div id="sigBox" class="kpi">—</div>
        <div class="muted" id="sigMeta">Waiting…</div>
      </div>
      <div class="card">
        <h2>OOF Backtest</h2>
        <div id="btTrades" class="kpi">—</div>
        <div class="muted">Trades • <span id="btWin">—</span> win • <span id="btAvg">—</span> avg • <span id="btSharpe">—</span> S~</div>
      </div>
    </div>

    <div class="row" style="margin-top:16px">
      <div class="card">
        <h2>Signal history</h2>
        <table id="sigTable"><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>p_up</th><th>p_dn</th><th>Price</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
<script>
async function train(){
  const btn = document.getElementById('trainBtn');
  btn.disabled = true; btn.innerText = 'Training…';
  try{
    const r = await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
    const j = await r.json();
    alert('Training done! Precision='+(j.cv?.prec_mean?.toFixed(3))); 
    await loadOOF();
  }catch(e){ alert('Train error: '+e); }
  finally{ btn.disabled=false; btn.innerText='Train model'; }
}
async function refreshLive(){
  const r = await fetch('/api/live');
  const j = await r.json();
  const sigMap = {1:'LONG', '-1':'SHORT', 0:'FLAT'};
  const sigEl = document.getElementById('sigBox');
  sigEl.textContent = sigMap[j.signal] + ' @ ' + j.price?.toFixed(2);
  sigEl.className = 'kpi ' + (j.signal==1?'green':(j.signal==-1?'red':'gray'));
  document.getElementById('sigMeta').textContent = `${j.symbol} • p_up ${j.p_up?.toFixed(2)} • p_dn ${j.p_dn?.toFixed(2)} • ${j.time}`;
  const tb = document.querySelector('#sigTable tbody');
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${j.time}</td><td>${j.symbol}</td><td>${sigMap[j.signal]}</td><td>${j.p_up?.toFixed(2)}</td><td>${j.p_dn?.toFixed(2)}</td><td>${j.price?.toFixed(2)}</td>`;
  tb.prepend(tr);
}
async function loadOOF(){
  const r = await fetch('/api/oof'); const j = await r.json();
  document.getElementById('btTrades').textContent = j.trades ?? '—';
  document.getElementById('btWin').textContent = j.winrate? (100*j.winrate).toFixed(1)+'%':'—';
  document.getElementById('btAvg').textContent = j.avg_ret? (100*j.avg_ret).toFixed(3)+'%':'—';
  document.getElementById('btSharpe').textContent = j.sharpe_like? j.sharpe_like.toFixed(2):'—';
}
setInterval(refreshLive, 30000);
loadOOF(); refreshLive();
</script>
</body>
</html>"""
    return HTMLResponse(html)

@app.get("/api/config")
def get_config():
    return JSONResponse(dc.asdict(CFG))

class TrainRequest(BaseModel):
    symbols: List[str] | None = None
    timeframe: str | None = None
    days_history: int | None = None

@app.post("/api/train")
def api_train(req: TrainRequest):
    if req.symbols:
        CFG.symbols = tuple(req.symbols)
    if req.timeframe:
        CFG.timeframe = req.timeframe
    if req.days_history:
        CFG.days_history = int(req.days_history)

    X, y, ds = build_dataset(CFG)
    model, calib, thr, oof_proba, y_aligned, cv = fit_model_cv(X, y, CFG)

    sig = proba_to_signal(oof_proba, thr["p_up"], thr["p_dn"])
    rep = simple_backtest(y_aligned, sig, ds, CFG)

    os.makedirs(CFG.model_dir, exist_ok=True)
    with open(CFG.oof_report_file, "w") as f:
        json.dump(rep, f)

    out = {"thresholds": thr, "cv": cv, "oof": rep}
    return JSONResponse(out)

@app.get("/api/oof")
def api_oof():
    if os.path.exists(CFG.oof_report_file):
        with open(CFG.oof_report_file, "r") as f:
            rep = json.load(f)
        return JSONResponse(rep)
    return JSONResponse({"trades": 0, "winrate": None, "avg_ret": None, "sharpe_like": None})

@app.get("/api/live")
def api_live():
    raw = load_all_symbols(CFG)
    frames = []
    for sym, g in raw.groupby("symbol"):
        f = add_features(g.drop(columns=["symbol"]), CFG)
        f["symbol"] = sym
        frames.append(f)
    feat = pd.concat(frames).sort_index()

    sym_d = pd.get_dummies(feat["symbol"], prefix="sym")
    X = pd.concat([feat[FEATURE_COLS], sym_d], axis=1).replace([np.inf,-np.inf], np.nan).dropna()

    model, calib, thr = load_model(CFG)
    proba = calib.predict_proba(X)
    proba_df = pd.DataFrame(proba, index=X.index, columns=["p_dn","p_flat","p_up"]).replace([np.inf,-np.inf], np.nan).dropna()

    sig = proba_to_signal(proba_df, thr["p_up"], thr["p_dn"])
    last_t = sig.index[-1]
    sym = CFG.symbols[0]
    price = float(raw[raw["symbol"]==sym]["close"].iloc[-1])

    data = {
        "time": str(last_t),
        "symbol": sym,
        "price": price,
        "p_up": float(proba_df.iloc[-1]["p_up"]),
        "p_dn": float(proba_df.iloc[-1]["p_dn"]),
        "signal": int(sig.iloc[-1]),
    }
    STATE["last_live"] = data
    return JSONResponse(data)

# Optional Telegram (simple JSON body: {"bot":"...","chat":"...","text":"..."})
@app.post("/api/telegram")
def api_telegram(payload: Dict[str,str]):
    bot = payload.get("bot", ""); chat = payload.get("chat", ""); text = payload.get("text", "")
    if not bot or not chat or not text:
        return JSONResponse({"ok": False, "error": "Missing bot/chat/text"}, status_code=400)
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{bot}/sendMessage"
        data = json.dumps({"chat_id": chat, "text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
