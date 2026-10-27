"""Synthetic traffic 'ground truth' + a small trained model that predicts edge travel time (minutes)."""
import math
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROADS = ["highway", "arterial", "local"]
SPEED = {"highway": 90.0, "arterial": 55.0, "local": 30.0}      # km/h free flow
DELAY = {"highway": 0.0, "arterial": 0.15, "local": 0.25}       # min per segment (lights/junctions)
CONGESTION = {"highway": 3.0, "arterial": 1.8, "local": 0.4}    # sensitivity to rush hour
V_MAX = max(SPEED.values())
CENTER_SCALE = 3.0   # km: how far from the centre congestion reaches


def rush_level(h):
    """0..1 - peaks at 8:30 and 18:00."""
    return max(math.exp(-((h - 8.5) / 1.3) ** 2), math.exp(-((h - 18.0) / 1.5) ** 2))


def true_time(edge, hour, rng=None):
    """Hidden real-world travel time. Congestion is worst near the city centre on big roads."""
    centre = math.exp(-(edge.center_dist / CENTER_SCALE) ** 2)
    t = edge.length / SPEED[edge.road] * 60 * (1 + CONGESTION[edge.road] * rush_level(hour) * centre)
    t += DELAY[edge.road]
    return t * rng.lognormvariate(0, 0.05) if rng else t


def features(length, road, center_dist, hour):
    ang = 2 * math.pi * hour / 24
    return [length, *(1.0 if road == r else 0.0 for r in ROADS), math.sin(ang), math.cos(ang), center_dist]


def train_model(graph, n=30000, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        e = rng.choice(graph.edges)
        hr = rng.uniform(0, 24)
        X.append(features(e.length, e.road, e.center_dist, hr))
        y.append(true_time(e, hr, rng))
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {"MAE_min": mean_absolute_error(yte, pred), "R2": r2_score(yte, pred)}
    return model, metrics


def predict_table(model, graph, hour):
    """Predicted minutes for EVERY edge at this hour (one batched call -> fast routing)."""
    X = np.array([features(e.length, e.road, e.center_dist, hour) for e in graph.edges])
    return np.maximum(model.predict(X), 0.01)
