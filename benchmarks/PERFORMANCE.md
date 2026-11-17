# SmartRoute route-performance verification

Tested on 17 November 2026 using Python 3.13.7, the existing cached Ahmedabad road graph (59,710 nodes / 142,706 directed edges), and the existing trained Random Forest. No new training data, retraining, batch predictions, or UI changes were introduced.

## Route and method

- Sports Club of Gujarat: `23.0408384, 72.5646398`
- Gordhan Thal: `23.0359596, 72.5111196`
- Coordinates came from the project's existing Nominatim place search.
- Departure: **09:00, weekday**.
- Flask test-client request time measured with `time.perf_counter()`. External geocoding, network transit, map tiles, browser painting, and the one-time graph load are outside request timings.
- Baseline backend: Git revision `0498b97b502c730103823caaecdddce96ffdfb93`.
- One first-request measurement for each algorithm/mode; optimized identical repeats measure completed-route cache hits. The extra original A* distance repeat is retained in raw data. The second original A* time request was stopped after the first completed measurement because it would repeat a seven-minute calculation; no partial timing is reported.
- Optimized prediction/route caches were cleared between algorithms. Within each algorithm, distance runs precede time runs, allowing the shared edge cache to behave as it does in the application. The model is loaded by the first distance request. These are local measurements, not statistical estimates or universal latency guarantees.

## Implementation

- [Prediction code](../machine_learning/predict_time.py): at most 200,000 shared LRU entries; keys include model identity, `(source, target, parallel-edge key)`, length, road code, departure hour, and weekday/weekend. Request-local costs avoid repeat cache lookups during reconstruction. Locks coalesce concurrent identical prediction misses. Predictions still process **one edge at a time**. Inference uses `n_jobs=1` to avoid repeatedly creating parallel worker tasks for single samples; fitted trees and features are unchanged.
- [Routing code](../algorithms/routing.py): reject stale queue entries, avoid costs that cannot improve an already cheaper path, reuse adjacency lists and A* heuristic values, and retain the selected predecessor edge. A* can reopen improved nodes. No node-count, distance-radius, or timeout cutoffs are used.
- A* bounds are conservative: distance uses the minimum edge-length/geographic-step ratio. Time uses a lower bound from the forest's leaf values and the largest graph step. This is safe even where the former assumed 130 km/h bound would not be justified by model outputs. It can explore more nodes than the former heuristic; fewer prediction calls on every cold A* search are **not** claimed.
- [Graph code](../algorithms/graph.py) and [API code](../app.py): retain the existing adjacency-list graph and reuse a haversine BallTree built once per graph, with up to 4,096 nearest-node results.
- Completed-route cache: at most 256 serialized results, keyed by exact endpoint coordinates, algorithm, optimization mode, departure hour, day type, graph identity/revision, model identity, and cache generation. Identical concurrent requests share one calculation. Failures and loading responses are not cached.
- Graph/model file changes are detected using modification/creation timestamps and size; graph reloads and model replacements invalidate/version caches. Graph mutations through `add_node`/`add_edge` increment its revision. `invalidate_route_cache()` explicitly clears response/nearest caches; in-flight old generations cannot repopulate them. Loaded graph/model objects should not be modified directly behind these APIs.
- Reconstruction uses the exact chosen parallel edge rather than always taking the shortest edge for a pair of nodes. This preserves time-route geometry and totals when a longer parallel road is faster.
- Monitoring: successful responses expose `graph_loading`, `nearest_lookup`, `distance_routing`, `time_routing`, and `response_preparation` durations in `Server-Timing` (milliseconds), plus cache hit/miss status. The actual background graph-load duration is printed in the server console; request `graph_loading` measures the readiness check/scheduling only. Request timing dictionaries are also logged at INFO level.

## Tests and reproduction

The backend tests cover directed multigraph shortest paths against NetworkX, zero-cost edges, unreachable/same-node paths, stale priority-queue entries, parallel-road reconstruction, per-edge prediction caching, concurrent cache misses, all route-cache inputs, bounded caches, graph/model changes, and invalidation during calculation.

```powershell
& ./.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/route-ui.test.cjs
& ./.venv/Scripts/python.exe benchmarks/route_performance.py benchmarks/after.json
& ./.venv/Scripts/python.exe benchmarks/route_performance.py benchmarks/before.json --baseline-ref 0498b97b502c730103823caaecdddce96ffdfb93 --algorithm astar --repeats 1
& ./.venv/Scripts/python.exe benchmarks/route_performance.py benchmarks/before_dijkstra.json --baseline-ref 0498b97b502c730103823caaecdddce96ffdfb93 --algorithm dijkstra --repeats 1
& ./.venv/Scripts/python.exe benchmarks/verify_routes.py
```

The baseline runner loads the selected revision into its own Python process; it does not revert the working tree. Baseline time-routing runs take several minutes. The verification script compares saved baseline and optimized paths, coordinates, totals, and individual selected-edge predictions from the original and optimized Random Forest inference.

Raw measurements: [A* baseline](before.json), [Dijkstra baseline](before_dijkstra.json), [optimized](after.json). Measurements and final verification follow below.

## Measured request times

| Algorithm | Optimization | Before (s) | After, cache miss (s) | Speedup | Cached repeat (ms) |
|---|---|---:|---:|---:|---:|
| A* | Distance | 2.779 | 0.639 | 4.35x | 1.72 |
| A* | Time | 439.153 | 120.475 | 3.65x | 3.57 |
| Dijkstra | Distance | 2.798 | 0.639 | 4.38x | 3.16 |
| Dijkstra | Time | 521.997 | 117.974 | 4.42x | 5.69 |

All optimized cache-hit requests made **zero ML predictions**. The initial time-route requests still take roughly two minutes; this work does not claim instant uncached Random Forest routing.

## Stage measurements

| Algorithm / mode | Before nearest lookup (ms) | After nearest lookup (ms) | Before search (ms) | After search (ms) | After response preparation (ms) |
|---|---:|---:|---:|---:|---:|
| astar / distance | 447.721 | 3.735 | 5.473 | 4.959 | 494.684 |
| astar / time | 370.858 | 0.004 | 436974.868 | 120469.348 | 1.906 |
| dijkstra / distance | 382.524 | 2.830 | 33.127 | 48.022 | 587.026 |
| dijkstra / time | 267.512 | 0.005 | 519886.667 | 117963.664 | 3.182 |

Graph loading from the existing local GraphML file: **8.437 s before**, **8.686 s after**. The optimized startup includes building the reusable spatial index and heuristic bounds. Graph loading did not improve in this measurement. Dijkstra's distance-search loop alone also did not improve; its total request improved through nearest-node lookup and individual ML inference. Sub-millisecond stage differences should not be interpreted as stable benchmarks.

Baseline response-preparation time was not separately instrumented. Baseline total minus nearest/search includes response preparation, model loading and request overhead, so no isolated baseline response-preparation measurement is claimed.

## Prediction counts

| Algorithm / time route | Baseline prediction calls | Optimized predictor calls | Optimized actual ML predictions |
|---|---:|---:|---:|
| astar | 14971 | 18193 | 18068 |
| dijkstra | 18267 | 18226 | 18101 |

The optimized counts distinguish predictor calls from actual forest evaluations. Reconstruction and shared edges reuse costs. The more conservative certified A* heuristic explores more edges in this case; its cold speedup mainly comes from removing per-sample parallel-worker overhead. Repeated requests bypass search and inference entirely.

## Final correctness verification

All four first-request routes have **identical node paths and geometry hashes** before and after optimization, and identical API result fields. Both algorithms agree on optimal costs for each optimization mode. The verification script independently resummed the selected edges' distance and individual predicted time:

| Optimization | Distance (m) | Estimated time (s) | Displayed result |
|---|---:|---:|---|
| Distance | 6635.393100101696 | 2052.247796275745 | 6.64 km / 34.2 min |
| Time | 6733.515918185918 | 1555.9262862161333 | 6.73 km / 25.9 min |

Maximum observed per-edge difference between original parallel RF inference and optimized serial inference: **1.13686837721616e-13 seconds** (floating-point summation order; tested tolerance `rel_tol=1e-12`, `abs_tol=1e-9`). Trained tree values and displayed accuracy are unchanged. See [verification data](verification.json).

**13 backend tests and 3 existing frontend tests passed.** Python compilation, editor diagnostics, and `git diff --check` passed. No browser UI changes were made in this task. Restart the Flask process to load the updated backend.
