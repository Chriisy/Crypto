import numpy as np
import pandas as pd

def make_synth(symbols=("BTC/USDT","ETH/USDT"), n=400, start=10000.0, step=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    frames = []
    for sym in symbols:
        price = start + np.cumsum(np.random.randn(n) * step)
        open_ = price + np.random.randn(n) * 0.1
        close = price + np.random.randn(n) * 0.1
        high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.2)
        low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.2)
        vol = np.abs(1000 + 100*np.random.randn(n))
        df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=idx)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames).sort_index()
