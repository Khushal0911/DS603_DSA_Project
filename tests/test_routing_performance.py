"""Correctness and cache regression tests; no downloads or production data needed."""
import concurrent.futures
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import networkx as nx
import numpy as np

import app
from algorithms.graph import RoadGraph, RoadEdge
from algorithms.routing import astar, dijkstra
from machine_learning import predict_time


def small_graph():
    graph = RoadGraph()
    for node in range(4):
        graph.add_node(node, 23 + node * .001, 72)
    for source, target, length, road in [(0, 1, 100, 'slow'), (0, 1, 150, 'fast'),
                                        (1, 3, 200, 'fast'), (0, 2, 220, 'fast'),
                                        (2, 3, 220, 'fast')]:
        graph.add_edge(source, RoadEdge(target, length, road,
                       [list(graph.coordinates[source]), list(graph.coordinates[target])]))
    return graph


def cost(edge):
    return edge.length_m / (1 if edge.road_type == 'slow' else 10)


class RoutingTests(unittest.TestCase):
    def test_parallel_edges_reconstruction(self):
        graph = small_graph()
        for algorithm in (astar, dijkstra):
            for mode, expected_distance, expected_time in [('distance', 300, 120), ('time', 350, 35)]:
                edges = []
                path, total = algorithm(graph, 0, 3, mode, cost, path_edges=edges)
                self.assertEqual(path, [0, 1, 3])
                self.assertEqual(sum(e.length_m for e in edges), expected_distance)
                self.assertEqual(sum(cost(e) for e in edges), expected_time)
                coords, distance = app.build_route(path, edges, graph)
                self.assertEqual(distance, expected_distance)
                self.assertEqual(total, expected_time if mode == 'time' else expected_distance)
                self.assertEqual(coords[-1], list(graph.coordinates[3]))

    def test_random_directed_multigraphs_against_reference(self):
        rng = random.Random(603)
        for trial in range(40):
            graph = RoadGraph()
            reference = nx.MultiDiGraph()
            for node in range(12):
                graph.add_node(node, 23 + rng.random() * .1, 72 + rng.random() * .1)
                reference.add_node(node)
            for _ in range(60):
                source, target = rng.randrange(12), rng.randrange(12)
                length = rng.randrange(0, 1000)
                road = rng.choice(['slow', 'fast'])
                edge = RoadEdge(target, length, road, [])
                graph.add_edge(source, edge)
                reference.add_edge(source, target, distance=length, time=cost(edge))
            def predict(edge):
                return cost(edge)
            predict.minimum_seconds = min(cost(edge) for edges in graph.adjacency.values() for edge in edges)
            for mode in ('distance', 'time'):
                try:
                    expected = nx.dijkstra_path_length(reference, 0, 11, weight=mode)
                except nx.NetworkXNoPath:
                    expected = None
                for algorithm in (astar, dijkstra):
                    edges = []
                    path, actual = algorithm(graph, 0, 11, mode, predict, path_edges=edges)
                    if expected is None:
                        self.assertEqual(path, [])
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual, expected)
                        self.assertAlmostEqual(sum(cost(e) if mode == 'time' else e.length_m for e in edges), expected)

    def test_same_node_and_disconnected(self):
        graph = small_graph()
        for algorithm in (astar, dijkstra):
            edges = ['stale']
            self.assertEqual(algorithm(graph, 0, 0, path_edges=edges), ([0], 0))
            self.assertEqual(edges, [])
            self.assertEqual(algorithm(graph, 3, 0), ([], None))

    def test_outdated_queue_entries_do_not_repeat_edge_work(self):
        graph = RoadGraph()
        for node in range(4):
            graph.add_node(node, 23, 72)
        for source, target, weight in [(0, 1, 10), (0, 2, 1), (2, 1, 1), (1, 3, 100)]:
            graph.add_edge(source, RoadEdge(target, weight, 'fast', []))
        for algorithm in (astar, dijkstra):
            calls = []
            def predict(edge):
                calls.append(edge.edge_id)
                return edge.length_m
            self.assertEqual(algorithm(graph, 0, 3, 'time', predict), ([0, 2, 1, 3], 102))
            self.assertEqual(len(calls), len(set(calls)))

    def test_nearest_index_reused_and_invalidated(self):
        graph = small_graph()
        self.assertEqual(app.nearest_node(graph, 23, 72), 0)
        index = graph._nearest_index
        self.assertEqual(app.nearest_node(graph, 23.002, 72), 2)
        self.assertIs(graph._nearest_index, index)
        graph.add_node(9, 23.00001, 72)
        self.assertEqual(app.nearest_node(graph, 23.00001, 72), 9)
        self.assertIsNot(graph._nearest_index, index)


class FakeModel:
    minimum_seconds = 1.0
    def __init__(self):
        self.calls = []
    def predict(self, features):
        self.calls.append(features)
        assert len(features) == 1, 'Batch prediction is forbidden'
        return [sum(features[0])]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        predict_time._cached_prediction.cache_clear()

    def test_all_inputs_and_local_reuse(self):
        model = FakeModel()
        edge = RoadEdge(1, 100, 'primary', [], (0, 1, 0))
        first = predict_time.make_predictor(9, 0, model)
        self.assertEqual(first(edge), first(edge))
        self.assertEqual(predict_time.make_predictor(9, 0, model)(edge), first(edge))
        self.assertEqual(len(model.calls), 1)
        predict_time.make_predictor(10, 0, model)(edge)
        predict_time.make_predictor(9, 1, model)(edge)
        edge.length_m = 101
        first(edge)
        edge.road_type = 'service'
        first(edge)
        edge.edge_id = (0, 1, 1)
        first(edge)
        self.assertEqual(len(model.calls), 6)
        replacement = FakeModel()
        predict_time.make_predictor(9, 0, replacement)(edge)
        self.assertEqual(len(replacement.calls), 1)

    def test_simultaneous_prediction_misses_are_shared(self):
        model = FakeModel()
        edge = RoadEdge(1, 100, 'primary', [], (0, 1, 0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            values = list(executor.map(lambda _: predict_time.make_predictor(9, 0, model)(edge), range(32)))
        self.assertEqual(len(set(values)), 1)
        self.assertEqual(len(model.calls), 1)

    def test_model_file_replacement_invalidates_predictions(self):
        model = FakeModel()
        model.estimators_ = [SimpleNamespace(tree_=SimpleNamespace(
            children_left=np.array([-1]), value=np.array([[[1.0]]]))) ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump({'model': model}, path)
            with patch.object(predict_time, 'MODEL_PATH', path), \
                 patch.object(predict_time, '_bundle', None), \
                 patch.object(predict_time, '_model_stamp', None):
                first = predict_time.load_model()
                edge = RoadEdge(1, 100, 'primary', [], (0, 1, 0))
                predict_time.make_predictor(9, 0, first)(edge)
                self.assertEqual(predict_time._cached_prediction.cache_info().currsize, 1)
                self.assertIs(predict_time.load_model(), first)
                joblib.dump({'model': model, 'version': 'replacement'}, path)
                self.assertIsNot(predict_time.load_model(), first)
                self.assertEqual(predict_time._cached_prediction.cache_info().currsize, 0)


class APICacheTests(unittest.TestCase):
    def setUp(self):
        self.graph = small_graph()
        self.model = FakeModel()
        self.original_graph_check = app.graph_is_current
        self.patches = [patch.object(app, 'road_graph', self.graph),
                        patch.object(app, 'graph_is_current', return_value=True),
                        patch.object(app, 'load_model', return_value=self.model),
                        patch.object(app, 'make_predicted_time', return_value=cost)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        app.invalidate_route_cache()
        self.payload = {'start': {'lat': 23, 'lon': 72}, 'end': {'lat': 23.003, 'lon': 72},
                        'algorithm': 'astar', 'mode': 'time', 'hour': 9, 'weekend': 0}

    def post(self, **changes):
        with app.app.test_client() as client:
            return client.post('/api/route', json={**self.payload, **changes})

    def test_cache_inputs_and_response_values(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['distance_km'], .35)
        self.assertIn('route_cache;desc="miss"', response.headers['Server-Timing'])
        self.assertIn('time_routing;dur=', response.headers['Server-Timing'])
        self.assertEqual(response.json, self.post().json)
        self.assertIn('route_cache;desc="hit"', self.post().headers['Server-Timing'])
        variants = [{'mode': 'distance'}, {'algorithm': 'dijkstra'}, {'hour': 10}, {'weekend': 1},
                    {'start': {'lat': 23.00001, 'lon': 72}}, {'end': {'lat': 23.00299, 'lon': 72}}]
        for variant in variants:
            self.assertIn('route_cache;desc="miss"', self.post(**variant).headers['Server-Timing'])
        with patch.object(app, 'load_model', return_value=FakeModel()):
            self.assertIn('route_cache;desc="miss"', self.post().headers['Server-Timing'])
        self.graph.add_edge(0, RoadEdge(3, 250, 'fast', [[23, 72], [23.003, 72]]))
        self.assertEqual(self.post().json['distance_km'], .25)
        app.invalidate_route_cache()
        self.assertIn('route_cache;desc="miss"', self.post().headers['Server-Timing'])

    def test_simultaneous_routes_computed_once(self):
        original = app.astar
        def slow_search(*args, **kwargs):
            time.sleep(.02)
            return original(*args, **kwargs)
        with patch.object(app, 'astar', side_effect=slow_search) as search:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                responses = list(executor.map(lambda _: self.post(), range(4)))
            self.assertEqual(search.call_count, 1)
            self.assertTrue(all(response.status_code == 200 for response in responses))

    def test_cache_bound_and_failures(self):
        with patch.object(app, 'ROUTE_CACHE_SIZE', 2):
            for hour in range(3):
                self.post(hour=hour)
            self.assertEqual(len(app.route_cache), 2)
        app.invalidate_route_cache()
        response = self.post(start=self.payload['end'], end=self.payload['start'])
        self.assertEqual(response.status_code, 404)
        self.assertEqual(len(app.route_cache), 0)
        self.assertEqual(self.post(algorithm='bad').status_code, 400)

    def test_inflight_invalidation_does_not_repopulate_cache(self):
        original = app.astar
        def invalidate_during_search(*args, **kwargs):
            app.invalidate_route_cache()
            return original(*args, **kwargs)
        with patch.object(app, 'astar', side_effect=invalidate_during_search):
            self.assertEqual(self.post().status_code, 200)
        self.assertEqual(len(app.route_cache), 0)

    def test_changed_graph_file_requires_reload(self):
        # Use the real version check, overriding the fixture's ready-graph mock.
        with patch.object(app, 'graph_is_current', self.original_graph_check):
            with patch.object(app, 'loaded_graph_stamp', (1, 2, 3)), \
                 patch.object(app, 'graph_file_stamp', return_value=(4, 5, 6)), \
                 patch.object(app, 'start_graph_loading') as start_loading:
                self.assertEqual(self.post().status_code, 202)
                start_loading.assert_called_once()


if __name__ == '__main__':
    unittest.main()
