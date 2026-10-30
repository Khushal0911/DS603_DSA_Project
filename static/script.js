const ahmedabad = [23.0225, 72.5714];
const map = L.map('map', {zoomControl:true}).setView(ahmedabad, 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:19, attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
let startPlace = null, endPlace = null, routeLayer = null, startMarker = null, endMarker = null;

const $ = id => document.getElementById(id);
function setMessage(text, error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function markerIcon(color){return L.divIcon({className:'',html:`<div style="width:17px;height:17px;background:${color};border:3px solid white;border-radius:50%;box-shadow:0 2px 8px #17243b66"></div>`,iconSize:[17,17],iconAnchor:[8,8]});}
function updateMarkers(){
  if(startMarker) map.removeLayer(startMarker); if(endMarker) map.removeLayer(endMarker);
  if(startPlace) startMarker=L.marker([startPlace.lat,startPlace.lon],{icon:markerIcon('#1eb77a')}).addTo(map).bindPopup('Starting point');
  if(endPlace) endMarker=L.marker([endPlace.lat,endPlace.lon],{icon:markerIcon('#f05b63')}).addTo(map).bindPopup('Destination');
  const pts=[startPlace,endPlace].filter(Boolean);
  if(pts.length===2) map.fitBounds(pts.map(p=>[p.lat,p.lon]),{padding:[70,70]});
  else if(pts.length===1) map.setView([pts[0].lat,pts[0].lon],14);
}
async function search(query){
  const response=await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const data=await response.json(); if(!response.ok) throw new Error(data.error||'Search failed'); return data;
}
function attachSearch(inputId, resultsId, which){
  const input=$(inputId), box=$(resultsId); let timer;
  input.addEventListener('input',()=>{
    clearTimeout(timer); const query=input.value.trim();
    if(query.length<3){box.classList.remove('open');return;}
    timer=setTimeout(async()=>{
      try{
        const results=await search(query); box.innerHTML='';
        results.forEach(place=>{
          const item=document.createElement('div');item.className='suggestion';
          const title=document.createElement('div');title.textContent=place.name;
          const detail=document.createElement('small');detail.textContent=place.display_name;
          item.append(title,detail);item.addEventListener('click',()=>{
            if(which==='start') startPlace=place; else endPlace=place;
            input.value=place.name;box.classList.remove('open');updateMarkers();
          });box.appendChild(item);
        });
        if(results.length) box.classList.add('open'); else box.classList.remove('open');
      }catch(error){setMessage(error.message,true);}
    },450);
  });
  input.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();box.querySelector('.suggestion')?.click();}});
  document.addEventListener('click',e=>{if(!box.parentElement.contains(e.target))box.classList.remove('open');});
}
attachSearch('startSearch','startResults','start');
attachSearch('endSearch','endResults','end');

async function resolveDefaults(){
  try{
    const [a,b]=await Promise.all([search('Sabarmati Ashram'),search('Kankaria Lake')]);
    if(a.length){startPlace=a[0];$('startSearch').value=a[0].name;}
    if(b.length){endPlace=b[0];$('endSearch').value=b[0].name;}
    updateMarkers();
  }catch(error){setMessage('Enter and select both locations using place search. '+error.message,true);}
}
async function calculate(modeOverride=null){
  if(!startPlace||!endPlace){setMessage('Select both locations from the search suggestions first.',true);return;}
  $('mapLoading').classList.remove('hidden');$('routeButton').disabled=true;
  setMessage('Preparing real Ahmedabad road data and calculating your route…');
  try{
    const mode=modeOverride||$('mode').value;
    const response=await fetch('/api/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      start:startPlace,end:endPlace,algorithm:$('algorithm').value,mode,
      hour:Number($('hour').value),weekend:Number($('day').value)
    })});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Route calculation failed');
    
const validCoordinates = (data.coordinates || [])
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

if (validCoordinates.length < 2) {
    throw new Error(
        "Invalid route coordinates received. Please try another location."
    );
}

if (routeLayer) {
    map.removeLayer(routeLayer);
}

routeLayer = L.polyline(validCoordinates, {
    color: '#2864ed',
    weight: 6,
    opacity: 0.95,
    lineCap: 'round',
    lineJoin: 'round'
}).addTo(map);

const bounds = routeLayer.getBounds();

if (bounds.isValid()) {
    map.fitBounds(bounds, {
        padding: [65, 65]
    });
}

$('distance').textContent = `${data.distance_km} km`;
$('duration').textContent = `${data.estimated_minutes} min`;

$('algoTag').textContent =
    $('algorithm').value === 'astar'
        ? 'A* Search'
        : 'Dijkstra';

$('routeNote').textContent = data.note || '';

$('routeSummary').classList.remove('hidden');

setMessage(
    `Route found across ${data.nodes.toLocaleString()} road intersections.`
);

    routeLayer=L.polyline(data.coordinates,{color:'#2864ed',weight:6,opacity:.9,lineCap:'round',lineJoin:'round'}).addTo(map);
    map.fitBounds(routeLayer.getBounds(),{padding:[65,65]});
    $('distance').textContent=`${data.distance_km} km`;
    $('duration').textContent=`${data.estimated_minutes} min`;
    $('algoTag').textContent=$('algorithm').value==='astar'?'A* Search':'Dijkstra';
    $('routeNote').textContent=data.note;
    $('routeSummary').classList.remove('hidden');
    setMessage(`Route found across ${data.nodes.toLocaleString()} road intersections.`);
    $('networkStatus').textContent='Ahmedabad road network loaded';
    return data;
  }catch(error){setMessage(error.message,true);return null;}
  finally{$('mapLoading').classList.add('hidden');$('routeButton').disabled=false;}
}
$('routeButton').addEventListener('click',()=>calculate());
$('compareButton').addEventListener('click',async()=>{
  const distanceRoute=await calculate('distance');
  if(!distanceRoute) return;
  const timeRoute=await calculate('time');
  if(!timeRoute) return;
  setMessage(`Distance-optimized: ${distanceRoute.distance_km} km / ${distanceRoute.estimated_minutes} min. Time-optimized: ${timeRoute.distance_km} km / ${timeRoute.estimated_minutes} min. The time-optimized route is shown on the map.`);
});
fetch('/api/status').then(r=>r.json()).then(s=>{$('networkStatus').textContent=s.message;});
resolveDefaults();
