"""Individual Random Forest predictions, memoized by edge and departure settings."""
from functools import lru_cache
import threading

import joblib
from machine_learning.train_model import MODEL_PATH, road_code

_bundle = None
_model_stamp = None
_model_lock = threading.RLock()
_prediction_locks = [threading.Lock() for _ in range(64)]


def load_model():
    """Reload a replaced/retrained model; old route/prediction keys cannot match it."""
    global _bundle, _model_stamp
    with _model_lock:
        if not MODEL_PATH.exists():
            from machine_learning.train_model import train
            train()
        stat = MODEL_PATH.stat()
        stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        if _bundle is None or stamp != _model_stamp:
            bundle = joblib.load(MODEL_PATH)
            model = bundle['model']
            # A single sample does not benefit from starting workers for 100 trees.
            # All fitted trees and prediction features remain unchanged.
            model.n_jobs = 1
            model.minimum_seconds = max(1.0, sum(
                float(tree.tree_.value[tree.tree_.children_left == -1].min())
                for tree in model.estimators_
            ) / len(model.estimators_))
            _bundle = bundle
            _model_stamp = stamp
            _cached_prediction.cache_clear()
        return _bundle['model']


@lru_cache(maxsize=200_000)
def _cached_prediction(model, edge_id, length, road, hour, weekend):
    # Deliberately one edge per predict call: no batching.
    return max(1.0, float(model.predict([[length, hour, road, weekend]])[0]))


def make_predictor(hour=9, weekend=0, model=None):
    """Bind one model snapshot and reuse costs during search and reconstruction."""
    if model is None:
        model = load_model()
    hour, weekend = int(hour), int(weekend)
    costs = {}

    def predict(edge):
        edge_id = edge.edge_id if edge.edge_id is not None else id(edge)
        key = (model, edge_id, edge.length_m, road_code(edge.road_type), hour, weekend)
        if key not in costs:
            # Serialize identical misses across Flask request threads. Distinct
            # stripes can still predict independently; the LRU bounds shared memory.
            with _prediction_locks[hash(key) % len(_prediction_locks)]:
                costs[key] = _cached_prediction(*key)
        return costs[key]

    predict.minimum_seconds = getattr(model, 'minimum_seconds', 1.0)
    return predict


def predict_edge_seconds(edge, hour=9, weekend=0):
    return make_predictor(hour, weekend)(edge)
