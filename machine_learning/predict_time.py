"""Edge travel-time estimates from the trained demonstration model."""
from pathlib import Path
import joblib
from machine_learning.train_model import MODEL_PATH, road_code

_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            from machine_learning.train_model import train
            train()
        _bundle = joblib.load(MODEL_PATH)
    return _bundle["model"]


def predict_edge_seconds(edge, hour=9, weekend=0):
    model = load_model()
    features = [[edge.length_m, int(hour), road_code(edge.road_type), int(weekend)]]
    return max(1.0, float(model.predict(features)[0]))
