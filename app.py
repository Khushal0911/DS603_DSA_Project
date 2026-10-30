
from pathlib import Path
import threading
import time

import requests
import osmnx as ox
from flask import Flask, jsonify, render_template, request

from algorithms.graph import RoadGraph, RoadEdge
from algorithms.routing import astar, dijkstra
from machine_learning.predict_time import predict_edge_seconds


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
GRAPH_FILE = DATA_DIR / "ahmedabad_drive.graphml"

DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

road_graph = None
osm_graph = None

graph_error = None
graph_loading = False

graph_state_lock = threading.Lock()
graph_build_lock = threading.Lock()

# Ahmedabad urban-area bounding box
# Format: west, south, east, north
AHMEDABAD_BBOX = (72.45, 22.90, 72.75, 23.15)

AHMEDABAD_VIEWBOX = "72.45,23.15,72.75,22.90"


# --------------------------------------------------
# BUILD CUSTOM ROAD GRAPH
# --------------------------------------------------

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

        if geometry is not None:

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
            coordinates=coordinates
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

    if road_graph is not None:
        return road_graph

    with graph_build_lock:

        if road_graph is not None:
            return road_graph

        try:

            ox.settings.use_cache = True
            ox.settings.timeout = 180
            ox.settings.log_console = True

            # --------------------------------------
            # LOAD EXISTING GRAPH
            # --------------------------------------

            if GRAPH_FILE.exists():

                print("\nLoading cached Ahmedabad road network...")

                source_graph = ox.load_graphml(GRAPH_FILE)

            else:

                # ----------------------------------
                # DOWNLOAD WITH EXPONENTIAL BACKOFF
                # ----------------------------------

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

            # --------------------------------------
            # CREATE OUR OWN GRAPH STRUCTURE
            # --------------------------------------

            osm_graph = source_graph

            road_graph = build_custom_graph(source_graph)

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

        if road_graph is not None:
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

def nearest_node(graph, lat, lon):

    return ox.distance.nearest_nodes(

        osm_graph,

        X=float(lon),

        Y=float(lat)

    )


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

def make_predicted_time(hour, weekend):

    return lambda edge: predict_edge_seconds(
        edge,
        hour,
        weekend
    )


# --------------------------------------------------
# RECONSTRUCT ROUTE GEOMETRY
# --------------------------------------------------


def build_route(path):

    coordinates = []
    total_distance = 0.0

    for index in range(len(path) - 1):

        source = path[index]
        target = path[index + 1]

        edge = edge_for_pair(
            road_graph,
            source,
            target
        )

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
            list(road_graph.coordinates[node])
            for node in path
            if node in road_graph.coordinates
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

@app.route("/api/route", methods=["POST"])
def calculate_route():

    payload = request.get_json(silent=True) or {}

    try:

        start = payload["start"]

        end = payload["end"]

        hour = int(payload.get("hour", 9))

        weekend = int(payload.get("weekend", 0))

        mode = payload.get(
            "mode",
            "distance"
        )

        algorithm = payload.get(
            "algorithm",
            "astar"
        )

        if mode not in ("distance", "time"):

            return jsonify({
                "error": "Invalid optimization mode."
            }), 400

        if algorithm not in ("astar", "dijkstra"):

            return jsonify({
                "error": "Invalid algorithm."
            }), 400

        # Start background download if necessary
        if road_graph is None:

            start_graph_loading()

            return jsonify({

                "loading": True,

                "message": (
                    "Preparing Ahmedabad road network. "
                    "Please wait."
                )

            }), 202

        # Find nearest road intersections
        start_node = nearest_node(

            road_graph,

            start["lat"],

            start["lon"]

        )

        end_node = nearest_node(

            road_graph,

            end["lat"],

            end["lon"]

        )

        predictor = make_predicted_time(
            hour,
            weekend
        )

        # Select algorithm
        search_algorithm = (

            astar

            if algorithm == "astar"

            else dijkstra

        )

        # Calculate route
        path, _ = search_algorithm(

            road_graph,

            start_node,

            end_node,

            mode,

            predictor

        )

        if not path:

            return jsonify({

                "error": (
                    "No connected road route found. "
                    "Try different locations."
                )

            }), 404

        # Get route geometry and distance
        coordinates, distance_m = build_route(path)

        # Calculate estimated travel time
        estimated_seconds = 0.0

        for index in range(len(path) - 1):

            edge = edge_for_pair(

                road_graph,

                path[index],

                path[index + 1]

            )

            if edge is not None:

                estimated_seconds += predictor(edge)

        if len(coordinates) < 2:

            return jsonify({

                "error": "Route geometry could not be generated."

            }), 500

        return jsonify({

            "coordinates": coordinates,

            "distance_km": round(
                distance_m / 1000,
                2
            ),

            "estimated_minutes": round(
                estimated_seconds / 60,
                1
            ),

            "nodes": len(path),

            "algorithm": algorithm,

            "mode": mode,

            "note": (
                "Travel time is estimated using synthetic ML training "
                "data, not live traffic."
            )

        })

    except Exception as error:

        print("\nROUTE CALCULATION ERROR:", error)

        return jsonify({

            "error": str(error)

        }), 500


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
