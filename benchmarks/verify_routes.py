"""Verify measured real routes, exact geometry, chosen-edge costs and RF accuracy."""
import hashlib
import json
from pathlib import Path
import sys
import warnings
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import joblib
from machine_learning.predict_time import load_model, make_predictor
from machine_learning.train_model import MODEL_PATH, road_code

warnings.filterwarnings('ignore', message='X does not have valid feature names')
base = Path(__file__).resolve().parent
before = json.loads((base / 'before.json').read_text())['runs']
if (base / 'before_dijkstra.json').exists():
    before += json.loads((base / 'before_dijkstra.json').read_text())['runs']
after = json.loads((base / 'after.json').read_text())
app.load_road_graph()
graph = app.road_graph
model = load_model()
predict = make_predictor(after['hour'], after['weekend'], model)
original = joblib.load(MODEL_PATH)['model']
checks = []
max_difference = 0.0
for run in after['runs']:
    previous = next(r for r in before if r['algorithm'] == run['algorithm'] and r['mode'] == run['mode'])
    assert run['response'] == previous['response'], (run['algorithm'], run['mode'], 'response mismatch')
    assert run['geometry_sha256'] == previous['geometry_sha256'], 'geometry changed'
    if run['repeat']:
        assert run['actual_ml_predictions'] == 0
        continue
    assert run['nearest_nodes'] == previous['nearest_nodes']
    assert run['path'] == previous['path']
    assert math.isclose(run['cost'], previous['cost'], rel_tol=1e-12, abs_tol=1e-9)
    path = run['path']
    edges = []
    for source, target in zip(path, path[1:]):
        candidates = [edge for edge in graph.neighbors(source) if edge.target == target]
        assert candidates
        edges.append(min(candidates, key=predict if run['mode'] == 'time' else lambda edge: edge.length_m))
    coordinates, distance = app.build_route(path, edges, graph)
    duration = sum(predict(edge) for edge in edges)
    assert round(distance / 1000, 2) == run['response']['distance_km']
    assert round(duration / 60, 1) == run['response']['estimated_minutes']
    assert math.isclose(duration if run['mode'] == 'time' else distance, run['cost'], rel_tol=1e-12)
    assert hashlib.sha256(json.dumps(coordinates).encode()).hexdigest() == run['geometry_sha256']
    for edge in edges:
        # Compare optimized individual predictions against original parallel RF.
        expected = max(1.0, float(original.predict([[edge.length_m, after['hour'],
                              road_code(edge.road_type), after['weekend']]])[0]))
        difference = abs(expected - predict(edge))
        max_difference = max(max_difference, difference)
        assert math.isclose(expected, predict(edge), rel_tol=1e-12, abs_tol=1e-9)
    checks.append({'algorithm': run['algorithm'], 'mode': run['mode'], 'distance_m': distance,
                   'estimated_seconds': duration, 'path_and_geometry_match_baseline': True})
for mode in ('distance', 'time'):
    runs = [r for r in after['runs'] if r['mode'] == mode and r['repeat'] == 0]
    assert math.isclose(runs[0]['cost'], runs[1]['cost'], rel_tol=1e-12)
summary = {'checks': checks, 'max_rf_difference_seconds': max_difference,
           'astar_and_dijkstra_optimal_costs_agree': True,
           'all_cached_requests_used_zero_predictions': True}
(base / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
print(json.dumps(summary, indent=2))
