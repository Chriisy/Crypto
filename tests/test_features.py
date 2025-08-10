import importlib
from ._helpers import make_synth

# import add_features from either style
try:
    mod = importlib.import_module("app")
    add_features = getattr(mod, "add_features")
    CFG = getattr(mod, "CFG")
except Exception:
    add_features = importlib.import_module("app.ml.features").add_features
    CFG = importlib.import_module("app.core.config").CFG

def test_features_basic():
    raw = make_synth(symbols=("BTC/USDT",), n=200)
    df = raw.drop(columns=["symbol"])
    f = add_features(df, CFG)
    required = {"ema20","ema50","ema200","rsi14","macd","macd_sig","macd_hist","atr14","vwap","vwap_dist","bb_width","body_pct","upper_wick","lower_wick","trend_up"}
    assert required.issubset(set(f.columns))
    assert not f.isna().any().any()
    assert len(f) < len(df)
