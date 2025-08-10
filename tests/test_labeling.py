import importlib
import numpy as np
from ._helpers import make_synth

# import compute_labels from either style
try:
    mod = importlib.import_module("app")
    compute_labels = getattr(mod, "compute_labels")
    CFG = getattr(mod, "CFG")
except Exception:
    compute_labels = importlib.import_module("app.ml.labeling").compute_labels
    CFG = importlib.import_module("app.core.config").CFG

def test_triple_barrier_labels():
    raw = make_synth(symbols=("BTC/USDT",), n=200)
    df = raw.drop(columns=["symbol"])
    out = compute_labels(df, CFG)
    assert "label" in out.columns
    assert set(np.unique(out["label"].values)).issubset({-1,0,1})
    assert len(out) > 10
