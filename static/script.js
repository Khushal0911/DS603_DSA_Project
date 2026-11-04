const ahmedabad = [23.0225, 72.5714];

const map = L.map('map', {
  zoomControl: true
}).setView(ahmedabad, 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

let startPlace = null;
let endPlace = null;
let routeLayer = null;
let startMarker = null;
let endMarker = null;

const $ = id => document.getElementById(id);

function setMessage(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

function markerIcon(color) {
  return L.divIcon({
    className: '',
    html: `<div style="width:17px;height:17px;background:${color};border:3px solid white;border-radius:50%;box-shadow:0 2px 8px #17243b66"></div>`,
    iconSize: [17, 17],
    iconAnchor: [8, 8]
  });
}

function clearRoute() {
  if (routeLayer) {
    map.removeLayer(routeLayer);
    routeLayer = null;
  }

  $('routeSummary').classList.add('hidden');
}

function updateMarkers() {
  if (startMarker) {
    map.removeLayer(startMarker);
    startMarker = null;
  }

  if (endMarker) {
    map.removeLayer(endMarker);
    endMarker = null;
  }

  if (startPlace) {
    startMarker = L.marker(
      [startPlace.lat, startPlace.lon],
      { icon: markerIcon('#1eb77a') }
    ).addTo(map).bindPopup('Starting point');
  }

  if (endPlace) {
    endMarker = L.marker(
      [endPlace.lat, endPlace.lon],
      { icon: markerIcon('#f05b63') }
    ).addTo(map).bindPopup('Destination');
  }

  const points = [startPlace, endPlace].filter(Boolean);

  if (points.length === 2) {
    map.fitBounds(
      points.map(p => [p.lat, p.lon]),
      { padding: [70, 70] }
    );
  } else if (points.length === 1) {
    map.setView([points[0].lat, points[0].lon], 14);
  }

  setTimeout(() => map.invalidateSize(), 150);
}

async function search(query) {
  const response = await fetch(
    `/api/search?q=${encodeURIComponent(query)}`
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || 'Search failed');
  }

  return data;
}

function attachSearch(inputId, resultsId, which) {
  const input = $(inputId);
  const box = $(resultsId);
  let timer;

  input.addEventListener('input', () => {
    clearTimeout(timer);

    const query = input.value.trim();

    if (which === 'start') {
      startPlace = null;
    } else {
      endPlace = null;
    }

    clearRoute();
    updateMarkers();

    if (query.length < 3) {
      box.classList.remove('open');
      return;
    }

    timer = setTimeout(async () => {
      try {
        const results = await search(query);

        box.innerHTML = '';

        results.forEach(place => {
          const item = document.createElement('div');
          item.className = 'suggestion';

          const title = document.createElement('div');
          title.textContent = place.name;

          const detail = document.createElement('small');
          detail.textContent = place.display_name;

          item.append(title, detail);

          item.addEventListener('click', () => {
            if (which === 'start') {
              startPlace = place;
            } else {
              endPlace = place;
            }

            input.value = place.name;
            box.classList.remove('open');

            clearRoute();
            updateMarkers();
          });

          box.appendChild(item);
        });

        if (results.length) {
          box.classList.add('open');
        } else {
          box.classList.remove('open');
        }

      } catch (error) {
        setMessage(error.message, true);
      }

    }, 450);
  });

  input.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      box.querySelector('.suggestion')?.click();
    }
  });

  document.addEventListener('click', event => {
    if (!box.parentElement.contains(event.target)) {
      box.classList.remove('open');
    }
  });
}

attachSearch('startSearch', 'startResults', 'start');
attachSearch('endSearch', 'endResults', 'end');

async function resolveDefaults() {
  try {
    const [a, b] = await Promise.all([
      search('Sabarmati Ashram'),
      search('Kankaria Lake')
    ]);

    if (a.length) {
      startPlace = a[0];
      $('startSearch').value = a[0].name;
    }

    if (b.length) {
      endPlace = b[0];
      $('endSearch').value = b[0].name;
    }

    updateMarkers();

  } catch (error) {
    setMessage(
      'Enter and select both locations using place search. ' +
      error.message,
      true
    );
  }
}

async function requestRoute(payload) {

  // Handle HTTP 202 while the road network is loading.
  for (let attempt = 0; attempt < 120; attempt++) {

    const response = await fetch('/api/route', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (response.status === 202 || data.loading) {
      setMessage(
        data.message || 'Loading Ahmedabad road network…'
      );

      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }

    if (!response.ok) {
      throw new Error(
        data.error || 'Route calculation failed'
      );
    }

    return data;
  }

  throw new Error(
    'Road network is taking too long to load. Check the CMD window and try again.'
  );
}

function getValidCoordinates(rawCoordinates) {
  if (!Array.isArray(rawCoordinates)) {
    return [];
  }

  return rawCoordinates
    .filter(point =>
      Array.isArray(point) &&
      point.length >= 2 &&
      Number.isFinite(Number(point[0])) &&
      Number.isFinite(Number(point[1])) &&
      Number(point[0]) >= -90 &&
      Number(point[0]) <= 90 &&
      Number(point[1]) >= -180 &&
      Number(point[1]) <= 180
    )
    .map(point => [
      Number(point[0]),
      Number(point[1])
    ]);
}

async function calculate(modeOverride = null) {

  if (!startPlace || !endPlace) {
    setMessage(
      'Select both locations from the search suggestions first.',
      true
    );
    return null;
  }

  $('mapLoading').classList.remove('hidden');
  $('routeButton').disabled = true;

  // Remove the previous route.
  clearRoute();

  setMessage('Calculating the route…');

  try {

    const payload = {
      start: startPlace,
      end: endPlace,
      algorithm: $('algorithm').value,
      mode: modeOverride || $('mode').value,
      hour: Number($('hour').value),
      weekend: Number($('day').value)
    };

    const data = await requestRoute(payload);

    const coordinates = getValidCoordinates(
      data.coordinates
    );

    if (coordinates.length < 2) {
      throw new Error(
        'Invalid route coordinates received. Please try another location.'
      );
    }

    // Ensure only one route layer exists.
    clearRoute();

    routeLayer = L.polyline(coordinates, {
      color: '#2864ed',
      weight: 6,
      opacity: 0.95,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(map);

    // Refresh map dimensions before fitting the route.
    map.invalidateSize();

    const bounds = routeLayer.getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [65, 65]
      });
    }

    $('distance').textContent =
      `${data.distance_km} km`;

    $('duration').textContent =
      `${data.estimated_minutes} min`;

    $('algoTag').textContent =
      $('algorithm').value === 'astar'
        ? 'A* Search'
        : 'Dijkstra';

    $('routeNote').textContent = data.note || '';

    $('routeSummary').classList.remove('hidden');

    $('networkStatus').textContent =
      'Ahmedabad road network loaded';

    setMessage(
      `Route found across ${Number(data.nodes).toLocaleString()} road intersections.`
    );

    setTimeout(() => map.invalidateSize(), 200);

    return data;

  } catch (error) {

    clearRoute();
    setMessage(error.message, true);

    return null;

  } finally {

    $('mapLoading').classList.add('hidden');
    $('routeButton').disabled = false;

  }
}

$('routeButton').addEventListener('click', () => {
  calculate();
});

$('compareButton').addEventListener('click', async () => {

  const distanceRoute = await calculate('distance');

  if (!distanceRoute) return;

  const timeRoute = await calculate('time');

  if (!timeRoute) return;

  setMessage(
    `Distance-optimized: ${distanceRoute.distance_km} km / ${distanceRoute.estimated_minutes} min. ` +
    `Time-optimized: ${timeRoute.distance_km} km / ${timeRoute.estimated_minutes} min. ` +
    'The time-optimized route is shown on the map.'
  );

});

fetch('/api/status')
  .then(response => response.json())
  .then(status => {
    $('networkStatus').textContent = status.message;
  })
  .catch(() => {
    $('networkStatus').textContent =
      'Road network status unavailable';
  });

window.addEventListener('resize', () => {
  setTimeout(() => map.invalidateSize(), 150);
});

setTimeout(() => map.invalidateSize(), 200);

resolveDefaults();
