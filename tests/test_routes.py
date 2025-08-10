import importlib, os
from fastapi.testclient import TestClient
from ._helpers import make_synth

# Try to import both module styles
try:
    APP_MODULE = importlib.import_module("app")
    app = getattr(APP_MODULE, "app")
except Exception:
    APP_MODULE = importlib.import_module("app.main")
    app = getattr(APP_MODULE, "app")

# Monkeypatch data loader
def fake_loader(cfg):
    return make_synth(symbols=("BTC/USDT","ETH/USDT"), n=240)

# Try attach to whichever module owns it
for modname in ("app", "app.services.data"):
    try:
        m = importlib.import_module(modname)
        setattr(m, "load_all_symbols", fake_loader)
    except Exception:
        pass

client = TestClient(app)

def test_train_and_live_endpoints(tmp_path):
    models_dir = tmp_path / "models"
    os.makedirs(models_dir, exist_ok=True)

    # Redirect paths if CFG exists
    for modname in ("app", "app.core.config"):
        try:
            m = importlib.import_module(modname)
            CFG = getattr(m, "CFG")
            CFG.model_dir = str(models_dir)
            CFG.model_file = str(models_dir / "lgbm_agent.txt")
            CFG.calib_file = str(models_dir / "calibrator.joblib")
            CFG.threshold_file = str(models_dir / "thresholds.json")
            CFG.oof_report_file = str(models_dir / "oof_report.json")
        except Exception:
            pass

    r = client.post("/api/train", json={})
    assert r.status_code == 200
    r = client.get("/api/live")
    assert r.status_code == 200
