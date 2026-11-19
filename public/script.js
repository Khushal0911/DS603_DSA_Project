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
let routeLayers = [];
let routeGeneration = 0;
let routeController = null;
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
  routeGeneration++;
  routeController?.abort();
  routeController = null;
  routeLayers.forEach(layer => map.removeLayer(layer));
  routeLayers = [];
  $('routeSummary').classList.add('hidden');
  $('routeSummary').querySelectorAll('strong, .tag, .route-note').forEach(el => { el.textContent = ''; });
  $('mapLoading').classList.add('hidden');
  $('routeButton').disabled = false;
  setMessage('Select locations, then find both routes.');
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
  const initialStart = $('startSearch').value;
  const initialEnd = $('endSearch').value;
  try {
    const [a, b] = await Promise.all([
      search('Sabarmati Ashram'),
      search('Kankaria Lake')
    ]);

    if (a.length && $('startSearch').value === initialStart && !startPlace) {
      startPlace = a[0];
      $('startSearch').value = a[0].name;
    }

    if (b.length && $('endSearch').value === initialEnd && !endPlace) {
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

async function requestRoute(payload, signal) {

  // Handle HTTP 202 while the road network is loading.
  for (let attempt = 0; attempt < 120; attempt++) {

    const response = await fetch('/api/route', {
      method: 'POST',
      signal,
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

      await new Promise((resolve, reject) => {
        const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
        const timer = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve(); }, 500);
        signal.addEventListener('abort', onAbort, { once: true });
        if (signal.aborted) onAbort();
      });
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

async function calculate() {
  clearRoute();
  if (!startPlace || !endPlace) {
    setMessage('Select both locations from the search suggestions first.', true);
    return;
  }
  const generation = routeGeneration;
  const controller = new AbortController();
  routeController = controller;
  $('mapLoading').classList.remove('hidden');
  $('routeButton').disabled = true;
  setMessage('Calculating distance-optimized and time-optimized routes...');
  const payload = {
    start: startPlace, end: endPlace, algorithm: $('algorithm').value,
    hour: Number($('hour').value), weekend: Number($('day').value)
  };
  try {
    const modes = ['distance', 'time'];
    const routes = await Promise.all(modes.map(mode => requestRoute({ ...payload, mode }, controller.signal)));
    if (generation !== routeGeneration) return;
    const points = routes.map(route => getValidCoordinates(route.coordinates));
    if (points.some(coords => coords.length < 2)) throw new Error('Invalid route coordinates received. Please try another location.');
    routes.forEach((data, index) => {
      const mode = modes[index];
      routeLayers.push(L.polyline(points[index], {
        color: index === 0 ? '#2864ed' : '#159957',
        weight: index === 0 ? 8 : 4, opacity: 0.95,
        dashArray: index === 1 ? '10 8' : null,
        lineCap: 'round', lineJoin: 'round'
      }).addTo(map));
      $(mode + 'Distance').textContent = data.distance_km + ' km';
      $(mode + 'Duration').textContent = data.estimated_minutes + ' min';
      $(mode + 'Algo').textContent = payload.algorithm === 'astar' ? 'A* Search' : 'Dijkstra';
      $(mode + 'Note').textContent = data.note || '';
    });
    $('routeSummary').classList.remove('hidden');
    $('networkStatus').textContent = 'Ahmedabad road network loaded';
    setMessage('Both routes calculated. Shared segments show dashed green over blue.');
    requestAnimationFrame(() => {
      if (generation !== routeGeneration) return;
      map.invalidateSize({ pan: false });
      map.fitBounds(L.latLngBounds(points.flat()), { padding: [40, 40], animate: false });
    });
  } catch (error) {
    if (generation !== routeGeneration) return;
    clearRoute();
    setMessage(error.message || 'Route calculation failed', true);
  } finally {
    if (generation === routeGeneration) {
      $('mapLoading').classList.add('hidden');
      $('routeButton').disabled = false;
      routeController = null;
    }
  }
}

$('routeButton').addEventListener('click', calculate);
['algorithm', 'hour', 'day'].forEach(id => $(id).addEventListener('change', clearRoute));

// Leaflet must measure its container after grid/layout changes, not only window resizes.
const mapResizeObserver = new ResizeObserver(() => {
  requestAnimationFrame(() => map.invalidateSize({ pan: false }));
});
mapResizeObserver.observe($('map'));

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
