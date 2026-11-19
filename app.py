
from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
from math import isfinite

import numpy as np
import networkx as nx
from sklearn.neighbors import BallTree
import threading
import time

import os
import requests
from flask import Flask, jsonify, render_template, request

from algorithms.graph import RoadGraph, RoadEdge
from algorithms.routing import astar, dijkstra
from machine_learning.predict_time import load_model, make_predictor


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
PACKAGED_GRAPH_FILE = BASE / "data" / "ahmedabad_drive.graphml"
RUNTIME_DATA_DIR = Path("/tmp/smartroute-data")
# Vercel instances have ephemeral /tmp storage. The repository already
# contains the complete road network, so prefer that immutable deployment
# asset and only use /tmp as a fallback for a missing local copy.
GRAPH_FILE = PACKAGED_GRAPH_FILE if PACKAGED_GRAPH_FILE.exists() else RUNTIME_DATA_DIR / "ahmedabad_drive.graphml"
DATA_DIR = GRAPH_FILE.parent

DATA_DIR.mkdir(exist_ok=True)

# Keep this path aligned with the tracked directory name.  This is
# case-sensitive on Linux (including Vercel), even though it appears to work
# locally on Windows when written as ``Public``.
PUBLIC_DIR = BASE / "public"
app = Flask(
    __name__,
    template_folder=str(PUBLIC_DIR),
    static_folder=str(PUBLIC_DIR),
    static_url_path="/public",
)

road_graph = None
osm_graph = None

graph_error = None
graph_loading = False

graph_state_lock = threading.Lock()
graph_build_lock = threading.Lock()
nearest_index_lock = threading.Lock()
route_cache_lock = threading.RLock()
route_locks = [threading.Lock() for _ in range(64)]
route_cache = OrderedDict()
ROUTE_CACHE_SIZE = 256
route_cache_version = None
route_cache_generation = 0
loaded_graph_stamp = None


def graph_file_stamp():
    if not GRAPH_FILE.exists():
        return None
    stat = GRAPH_FILE.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def graph_is_current():
    return road_graph is not None and loaded_graph_stamp == graph_file_stamp()


def invalidate_route_cache():
    """Clear response/nearest caches; graph and model reloads also version keys."""
    global route_cache_version, route_cache_generation
    with route_cache_lock:
        route_cache.clear()
        route_cache_version = None
        route_cache_generation += 1
    _nearest_node_cached.cache_clear()


# Ahmedabad urban-area bounding box
# Format: west, south, east, north
AHMEDABAD_BBOX = (72.45, 22.90, 72.75, 23.15)

AHMEDABAD_VIEWBOX = "72.45,23.15,72.75,22.90"


# --------------------------------------------------
# BUILD CUSTOM ROAD GRAPH
# --------------------------------------------------

def graphml_geometry_coordinates(geometry):
    """Convert GraphML WKT geometry to Leaflet [lat, lon] coordinates."""
    if not isinstance(geometry, str):
        return None
    text = geometry.strip()
    if not text.upper().startswith("LINESTRING"):
        return None
    body = text[text.find("(") + 1:text.rfind(")")]
    coordinates = []
    for pair in body.split(","):
        values = pair.strip().split()
        if len(values) >= 2:
            coordinates.append([float(values[1]), float(values[0])])
    return coordinates

def build_custom_graph(source_graph):

    custom = RoadGraph()

    # Add all road intersections
    for node_id, data in source_graph.nodes(data=True):

        custom.add_node(
            node_id,
            data["y"],
            data["x"]
        )

    # Add road connections
    for source, target, key, data in source_graph.edges(
        keys=True,
        data=True
    ):

        length = float(data.get("length", 50.0))

        highway = data.get("highway", "other")

        if isinstance(highway, list):
            highway = highway[0]

        geometry = data.get("geometry")

        wkt_coordinates = graphml_geometry_coordinates(geometry)

        if wkt_coordinates is not None:

            coordinates = wkt_coordinates

        elif geometry is not None:

            coordinates = [
                [float(lat), float(lon)]
                for lon, lat in geometry.coords
            ]

        else:

            coordinates = [
                list(custom.coordinates[source]),
                list(custom.coordinates[target])
            ]

        edge = RoadEdge(
            target=target,
            length_m=length,
            road_type=str(highway),
            coordinates=coordinates,
            edge_id=(source, target, key)
        )

        custom.add_edge(source, edge)

    return custom


# --------------------------------------------------
# DOWNLOAD AND LOAD AHMEDABAD ROAD NETWORK
# --------------------------------------------------

def load_road_graph():

    global road_graph
    global osm_graph
    global graph_error
    global loaded_graph_stamp

    if graph_is_current():
        return road_graph

    with graph_build_lock:

        if graph_is_current():
            return road_graph

        loading_started = time.perf_counter()
        try:
            source_stamp = None

            # --------------------------------------
            # LOAD EXISTING GRAPH
            # --------------------------------------

            if GRAPH_FILE.exists():

                print("\nLoading cached Ahmedabad road network...")

                source_stamp = graph_file_stamp()
                source_graph = nx.read_graphml(GRAPH_FILE, node_type=int)

            else:

                # ----------------------------------
                # DOWNLOAD WITH EXPONENTIAL BACKOFF
                # ----------------------------------

                # OSMnx is a development/download dependency. Production
                # deployments include the complete GraphML asset above.
                try:
                    import osmnx as ox
                except ImportError as error:
                    raise RuntimeError(
                        "Packaged road network is missing and OSMnx is not installed."
                    ) from error

                ox.settings.use_cache = True
                ox.settings.timeout = 180
                ox.settings.log_console = True
                source_graph = None

                for attempt in range(5):

                    try:

                        print(
                            f"\nDownloading Ahmedabad road network "
                            f"(attempt {attempt + 1}/5)..."
                        )

                        source_graph = ox.graph_from_bbox(
                            bbox=AHMEDABAD_BBOX,
                            network_type="drive",
                            simplify=True,
                            retain_all=False,
                            truncate_by_edge=True
                        )

                        if len(source_graph) == 0:

                            raise RuntimeError(
                                "Downloaded road network is empty."
                            )

                        print("\nRoad network downloaded successfully!")

                        break

                    except Exception as error:

                        print(f"\nDownload failed: {error}")

                        if attempt == 4:
                            raise

                        wait_time = 15 * (2 ** attempt)

                        print(
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                # Save downloaded graph locally
                print("\nSaving road network locally...")

                ox.save_graphml(
                    source_graph,
                    GRAPH_FILE
                )
                source_stamp = graph_file_stamp()

            # --------------------------------------
            # CREATE OUR OWN GRAPH STRUCTURE
            # --------------------------------------

            custom = build_custom_graph(source_graph)
            prepare_nearest_index(custom)
            osm_graph = source_graph
            loaded_graph_stamp = source_stamp
            road_graph = custom
            invalidate_route_cache()

            graph_error = None

            print("\n--------------------------------")
            print("AHMEDABAD ROAD NETWORK READY")
            print("--------------------------------")

            print(
                "Intersections:",
                len(road_graph.coordinates)
            )

            print(
                "Road connections:",
                sum(
                    len(edges)
                    for edges in road_graph.adjacency.values()
                )
            )

            print("--------------------------------\n")

            return road_graph

        except Exception as error:

            graph_error = str(error)

            print("\nROAD NETWORK ERROR:", graph_error)

            raise

        finally:
            elapsed = time.perf_counter() - loading_started
            app.logger.info("graph_loading=%.6fs", elapsed)
            print(f"PERF graph_loading={elapsed:.6f}s", flush=True)


# --------------------------------------------------
# BACKGROUND LOADING
# --------------------------------------------------

def graph_loading_worker():

    global graph_loading

    try:

        load_road_graph()

    except Exception:

        pass

    finally:

        with graph_state_lock:

            graph_loading = False


def start_graph_loading():

    global graph_loading
    global graph_error

    with graph_state_lock:

        if graph_is_current():
            return

        if graph_loading:
            return

        graph_loading = True
        graph_error = None

        thread = threading.Thread(
            target=graph_loading_worker,
            daemon=True
        )

        thread.start()


# --------------------------------------------------
# PLACE SEARCH
# --------------------------------------------------

def geocode(query):

    headers = {
        "User-Agent": "SmartRouteAcademicProject/1.1"
    }

    params = {

        "q": query + ", Ahmedabad, Gujarat, India",

        "format": "jsonv2",

        "limit": 6,

        "viewbox": AHMEDABAD_VIEWBOX,

        "bounded": 0
    }

    response = requests.get(

        "https://nominatim.openstreetmap.org/search",

        params=params,

        headers=headers,

        timeout=30
    )

    response.raise_for_status()

    results = response.json()

    return [

        {

            "name": item.get(
                "name",
                item.get("display_name", query)
            ),

            "display_name": item.get(
                "display_name",
                ""
            ),

            "lat": float(item["lat"]),

            "lon": float(item["lon"])

        }

        for item in results

    ]


# --------------------------------------------------
# FIND NEAREST ROAD INTERSECTION
# --------------------------------------------------

def prepare_nearest_index(graph):
    # Same haversine BallTree and node order as OSMnx, built once per graph.
    with nearest_index_lock:
        if graph._nearest_index is None:
            nodes = tuple(graph.coordinates)
            radians = np.deg2rad([graph.coordinates[node] for node in nodes])
            graph._nearest_index = (nodes, BallTree(radians, metric="haversine"))
    return graph._nearest_index


@lru_cache(maxsize=4096)
def _nearest_node_cached(graph, revision, lat, lon):
    nodes, tree = prepare_nearest_index(graph)
    _, positions = tree.query(np.deg2rad([[lat, lon]]), k=1)
    return int(nodes[positions[0, 0]])


def nearest_node(graph, lat, lon):
    return _nearest_node_cached(graph, graph.revision, float(lat), float(lon))


# --------------------------------------------------
# FIND ROAD EDGE
# --------------------------------------------------

def edge_for_pair(graph, source, target):

    candidates = [

        edge

        for edge in graph.neighbors(source)

        if edge.target == target

    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda edge: edge.length_m
    )


# --------------------------------------------------
# ML TRAVEL-TIME PREDICTION
# --------------------------------------------------

def make_predicted_time(hour, weekend, model=None):
    return make_predictor(hour, weekend, model)


# --------------------------------------------------
# RECONSTRUCT ROUTE GEOMETRY
# --------------------------------------------------


def build_route(path, edges=None, graph=None):

    graph = road_graph if graph is None else graph

    coordinates = []
    total_distance = 0.0

    if edges is None:
        edges = [edge_for_pair(graph, source, target)
                 for source, target in zip(path, path[1:])]

    for edge in edges:

        if edge is None:
            continue

        total_distance += edge.length_m

        segment = edge.coordinates

        if not segment:
            continue

        valid_segment = []

        for point in segment:

            if not isinstance(point, (list, tuple)):
                continue

            if len(point) < 2:
                continue

            try:
                lat = float(point[0])
                lon = float(point[1])
            except (TypeError, ValueError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            valid_segment.append([lat, lon])

        if not valid_segment:
            continue

        if coordinates and coordinates[-1] == valid_segment[0]:
            coordinates.extend(valid_segment[1:])
        else:
            coordinates.extend(valid_segment)

    if len(coordinates) < 2:
        coordinates = [
            list(graph.coordinates[node])
            for node in path
            if node in graph.coordinates
        ]

    return coordinates, total_distance


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# --------------------------------------------------
# PLACE SEARCH API
# --------------------------------------------------

@app.route("/api/search")
def search_places():

    query = request.args.get(
        "q",
        ""
    ).strip()

    if len(query) < 2:

        return jsonify([])

    try:

        results = geocode(query)

        return jsonify(results)

    except Exception as error:

        return jsonify({

            "error": f"Place search failed: {error}"

        }), 502


# --------------------------------------------------
# ROAD NETWORK STATUS API
# --------------------------------------------------

@app.route("/api/status")
def status():

    # Start warming the cached road graph as soon as the UI opens.  Previously
    # this work began only after the user clicked Find Route, which made the
    # first result appear unnecessarily slow.
    if road_graph is None and not graph_loading and graph_error is None:
        # A daemon thread can be stopped when a serverless invocation ends.
        # Load the packaged graph inside the request so status is truthful and
        # the next route request never waits on work from a dead invocation.
        if os.environ.get("VERCEL"):
            try:
                load_road_graph()
            except Exception:
                pass
        else:
            start_graph_loading()

    with graph_state_lock:

        loading = graph_loading

    if road_graph is not None:

        message = "Ahmedabad road network is ready."

    elif loading:

        message = (
            "Downloading Ahmedabad road network "
            "in the background..."
        )

    elif graph_error:

        message = (
            "Road network download failed. "
            "Check CMD for details."
        )

    else:

        message = (
            "Ready. Road network loads on first route request."
        )

    return jsonify({

        "ready": road_graph is not None,

        "loading": loading,

        "error": graph_error,

        "message": message

    })


# --------------------------------------------------
# ROUTE CALCULATION API
# --------------------------------------------------

def timed_response(body, timings, cache_hit=False, status_code=200):
    started = time.perf_counter()
    response = app.response_class(body, status=status_code, mimetype="application/json")
    timings["response_preparation"] += time.perf_counter() - started
    response.headers["Server-Timing"] = ", ".join(
        f"{stage};dur={seconds * 1000:.3f}" for stage, seconds in timings.items()
    ) + f', route_cache;desc="{"hit" if cache_hit else "miss"}"'
    app.logger.info("route_cache=%s %s", "hit" if cache_hit else "miss", timings)
    return response


@app.route("/api/route", methods=["POST"])
def calculate_route():
    global route_cache_version, route_cache_generation
    payload = request.get_json(silent=True) or {}
    timings = dict.fromkeys(("graph_loading", "nearest_lookup", "distance_routing",
                            "time_routing", "response_preparation"), 0.0)
    try:
        start, end = payload["start"], payload["end"]
        hour = int(payload.get("hour", 9))
        weekend = int(payload.get("weekend", 0))
        mode = payload.get("mode", "distance")
        algorithm = payload.get("algorithm", "astar")
        if mode not in ("distance", "time"):
            return jsonify({"error": "Invalid optimization mode."}), 400
        if algorithm not in ("astar", "dijkstra"):
            return jsonify({"error": "Invalid algorithm."}), 400
        points = tuple(float(value) for value in
                       (start["lat"], start["lon"], end["lat"], end["lon"]))
        if not all(isfinite(value) for value in points):
            return jsonify({"error": "Invalid coordinates."}), 400

        started = time.perf_counter()
        if not graph_is_current():
            if os.environ.get("VERCEL"):
                load_road_graph()
            else:
                start_graph_loading()
                timings["graph_loading"] = time.perf_counter() - started
                return timed_response(app.json.dumps({
                    "loading": True,
                    "message": "Preparing Ahmedabad road network. Please wait."
                }), timings, status_code=202)
        graph = road_graph  # Keep one graph/model snapshot for this entire request.
        timings["graph_loading"] = time.perf_counter() - started
        model = load_model()
        version = (graph, graph.revision, model)
        with route_cache_lock:
            if route_cache_version != version:
                route_cache.clear()
                route_cache_version = version
                route_cache_generation += 1
            generation = route_cache_generation
        key = (version, generation, points, algorithm, mode, hour, weekend)

        # Identical simultaneous requests share the completed result.
        with route_locks[hash(key) % len(route_locks)]:
            with route_cache_lock:
                body = route_cache.get(key)
                if body is not None:
                    route_cache.move_to_end(key)
            if body is not None:
                return timed_response(body, timings, cache_hit=True)

            started = time.perf_counter()
            start_node = nearest_node(graph, *points[:2])
            end_node = nearest_node(graph, *points[2:])
            timings["nearest_lookup"] = time.perf_counter() - started
            predictor = make_predicted_time(hour, weekend, model)
            search_algorithm = astar if algorithm == "astar" else dijkstra
            edges = []
            started = time.perf_counter()
            path, _ = search_algorithm(graph, start_node, end_node, mode, predictor,
                                       path_edges=edges)
            timings[f"{mode}_routing"] = time.perf_counter() - started
            if not path:
                return timed_response(app.json.dumps({
                    "error": "No connected road route found. Try different locations."
                }), timings, status_code=404)

            started = time.perf_counter()
            coordinates, distance_m = build_route(path, edges, graph)
            estimated_seconds = sum(predictor(edge) for edge in edges)
            if len(coordinates) < 2:
                return timed_response(app.json.dumps({
                    "error": "Route geometry could not be generated."
                }), timings, status_code=500)
            body = app.json.dumps({
                "coordinates": coordinates,
                "distance_km": round(distance_m / 1000, 2),
                "estimated_minutes": round(estimated_seconds / 60, 1),
                "nodes": len(path), "algorithm": algorithm, "mode": mode,
                "note": "Travel time is estimated using synthetic ML training data, not live traffic."
            })
            timings["response_preparation"] = time.perf_counter() - started
            with route_cache_lock:
                # A reload/invalidation during calculation must not repopulate old entries.
                if route_cache_version == version and route_cache_generation == generation:
                    route_cache[key] = body
                    route_cache.move_to_end(key)
                    if len(route_cache) > ROUTE_CACHE_SIZE:
                        route_cache.popitem(last=False)
            return timed_response(body, timings)
    except Exception as error:
        print("\nROUTE CALCULATION ERROR:", error)
        return jsonify({"error": str(error)}), 500


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    print("\n================================")
    print("       SMARTROUTE AHMEDABAD")
    print("================================")

    print(
        "\nOpen http://127.0.0.1:5000 in your browser"
    )

    print(
        "\nFirst route request downloads the road network."
    )

    print(
        "The successful download is cached locally."
    )

    app.run(

        debug=False,

        threaded=True

    )
