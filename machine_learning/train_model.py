"""Train a demonstration travel-time model (synthetic labels, not live traffic)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "travel_time_model.joblib"
ROAD_CODES = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3,
              "tertiary": 4, "residential": 5, "unclassified": 6, "service": 7,
              "other": 8}


def road_code(road_type):
    return ROAD_CODES.get(str(road_type).lower(), 8)


def train():
    rng = np.random.default_rng(42)
    rows = []
    road_speeds = [90, 70, 50, 40, 35, 25, 25, 15, 25]
    for _ in range(6000):
        length = float(rng.uniform(40, 3500))
        hour = int(rng.integers(0, 24))
        road = int(rng.integers(0, 9))
        weekend = int(rng.integers(0, 2))
        peak = 1.0 if (8 <= hour <= 11 or 17 <= hour <= 21) else 0.0
        congestion = (0.35 + 0.28 * peak + 0.08 * weekend) * (1.0 if road >= 2 else 0.55)
        speed = road_speeds[road] * (1 - congestion) * rng.uniform(0.82, 1.12)
        seconds = (length / 1000) / max(speed, 5) * 3600
        rows.append([length, hour, road, weekend, seconds])
    data = pd.DataFrame(rows, columns=["length_m", "hour", "road_code", "weekend", "seconds"])
    x = data[["length_m", "hour", "road_code", "weekend"]]
    y = data["seconds"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(x_test))
    joblib.dump({"model": model, "road_codes": ROAD_CODES}, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Validation MAE on synthetic test data: {mae:.1f} seconds")
    print("Important: this is a demonstration model, not a live-traffic prediction.")


if __name__ == "__main__":
    train()
