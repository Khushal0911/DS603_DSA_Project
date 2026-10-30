# SmartRoute – Real Ahmedabad Route Planner

A Flask-based route planner using real road geometry from OpenStreetMap. The interface uses Leaflet and OpenStreetMap tiles. The route-search algorithms (Dijkstra and A*) are implemented manually over a Python adjacency-list graph.

## Features
- Real, zoomable Ahmedabad street map.
- Search for places using OpenStreetMap Nominatim.
- Download and cache the Ahmedabad driving-road network on the first route request.
- Dijkstra and A* implemented from scratch with a min-priority queue.
- Route optimization by distance or demonstration ML-estimated travel time.
- Route distance, estimated time, and highlighted road geometry.

## Requirements
- Python 3.10 or newer (Python 3.13 should work with current package versions).
- Internet connection for map tiles, place search, and the first road-network download.

## Windows CMD setup
Open CMD in the folder containing `app.py`, then run:

```cmd
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python machine_learning\train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

If the `py -3.13` launcher is unavailable, use `python -m venv venv` instead. Always install packages with `python -m pip` while the virtual environment is activated.

## First run
The base map appears immediately. When you calculate your first route, the application downloads the Ahmedabad driving network from OpenStreetMap and saves it to `data/ahmedabad_drive.graphml`. This may take a few minutes. Later runs load the cached graph instead of downloading it again.

## Data structures and algorithms
- **Adjacency list (`dict[node] -> list[RoadEdge]`)**: stores only existing road connections, so it is more memory-efficient than a dense adjacency matrix for a sparse road network.
- **Priority queue (`heapq`)**: efficiently selects the next lowest-cost candidate.
- **Dictionary**: stores best-known costs and predecessor nodes.
- **Set**: prevents reprocessing finalized nodes.
- **Dijkstra**: finds a minimum-cost route with non-negative edge weights.
- **A\***: uses straight-line distance as an admissible heuristic. For time routing, the heuristic assumes a deliberately high maximum road speed to remain a lower bound.

Both algorithms are coded in this project; OSMnx is used to obtain and cache road data, not to calculate the selected route.

## Machine-learning limitation
The Random Forest model is trained on synthetic demonstration data. It illustrates how edge attributes and departure time can influence estimated travel time. It is **not** trained on measured Ahmedabad traffic observations and does **not** provide live traffic. For a real traffic product, replace the synthetic training data with timestamped observed travel times from a licensed data provider.

## OpenStreetMap usage
Map tiles and road-network data are from OpenStreetMap. Follow the tile usage policy and Nominatim usage policy; use a descriptive User-Agent and do not send rapid repeated geocoding requests. This is an academic demo, not a high-volume public service.
