"""Repeatable API benchmark; run before/after using the same cached graph/model."""
import hashlib
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import warnings
import platform
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--algorithm', choices=['astar', 'dijkstra'])
parser.add_argument('--repeats', type=int, default=2)
parser.add_argument('--baseline-ref', help='Read original Python modules from a Git revision without modifying the checkout')
args = parser.parse_args()
if args.baseline_ref:
    for name in ('algorithms.graph', 'algorithms.routing', 'machine_learning.predict_time', 'app'):
        relative = name.replace('.', '/') + '.py'
        if '.' in name:
            importlib.import_module(name.rsplit('.', 1)[0])
        source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:{relative}'], cwd=ROOT)
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / relative)
        sys.modules[name] = module
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(sys.modules[parent], child, module)
import app
from machine_learning import predict_time

warnings.filterwarnings('ignore', message='X does not have valid feature names')
OUTPUT = args.output
START = {'name': 'Sports Club of Gujarat', 'lat': 23.0408384, 'lon': 72.5646398}
END = {'name': 'Gordhan Thal', 'lat': 23.0359596, 'lon': 72.5111196}
result = {'start': START, 'end': END, 'hour': 9, 'weekend': 0,
          'python': platform.python_version(), 'baseline_ref': args.baseline_ref, 'runs': []}
t0 = time.perf_counter()
app.load_road_graph()
result['graph_loading_seconds'] = time.perf_counter() - t0
original_nearest = app.nearest_node
original_predictor = app.make_predicted_time
original_algorithms = {name: getattr(app, name) for name in ('astar', 'dijkstra')}
current = {}

def nearest(*args, **kwargs):
    t = time.perf_counter()
    value = original_nearest(*args, **kwargs)
    current['nearest_seconds'] = current.get('nearest_seconds', 0) + time.perf_counter() - t
    current.setdefault('nearest_nodes', []).append(int(value))
    return value

def predictor(*args, **kwargs):
    predict = original_predictor(*args, **kwargs)
    def wrapped(edge):
        current['prediction_calls'] = current.get('prediction_calls', 0) + 1
        return predict(edge)
    # Preserve certified heuristic lower bound, when supplied by optimized code.
    wrapped.minimum_seconds = getattr(predict, 'minimum_seconds', 0.0)
    return wrapped

def algorithm_wrapper(name):
    def run(*args, **kwargs):
        t = time.perf_counter()
        path, cost = original_algorithms[name](*args, **kwargs)
        current['routing_seconds'] = time.perf_counter() - t
        current['path'] = [int(node) for node in path]
        current['cost'] = cost
        return path, cost
    return run

app.nearest_node = nearest
app.make_predicted_time = predictor
for name in original_algorithms:
    setattr(app, name, algorithm_wrapper(name))
client = app.app.test_client()
for name in ([args.algorithm] if args.algorithm else ('astar', 'dijkstra')):
    if hasattr(app, 'invalidate_route_cache'):
        app.invalidate_route_cache()
        predict_time._cached_prediction.cache_clear()
    for mode in ('distance', 'time'):
        for repeat in range(args.repeats):
            current = {'algorithm': name, 'mode': mode, 'repeat': repeat}
            prediction_cache = getattr(predict_time, '_cached_prediction', None)
            misses_before = prediction_cache.cache_info().misses if prediction_cache else 0
            t0 = time.perf_counter()
            response = client.post('/api/route', json={
                'start': START, 'end': END, 'algorithm': name, 'mode': mode,
                'hour': 9, 'weekend': 0,
            })
            current['total_seconds'] = time.perf_counter() - t0
            if prediction_cache:
                current['actual_ml_predictions'] = prediction_cache.cache_info().misses - misses_before
            current['status'] = response.status_code
            body = response.get_json()
            coordinates = body.pop('coordinates', [])
            current['geometry_sha256'] = hashlib.sha256(json.dumps(coordinates).encode()).hexdigest()
            current['response'] = body
            current['server_timing'] = response.headers.get('Server-Timing')
            result['runs'].append(current)
            OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf8')
            print(json.dumps({k:v for k,v in current.items() if k not in ('path',)}), flush=True)
            if response.status_code != 200:
                raise RuntimeError(body)
