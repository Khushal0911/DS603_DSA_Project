const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
function setup() {
  const elements = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id, { value: '', textContent: '', disabled: false, classList: { add(){}, remove(){}, toggle(){} }, addEventListener(){}, querySelectorAll(){return [];} });
    return elements.get(id);
  };
  el('algorithm').value='astar'; el('hour').value='9'; el('day').value='0';
  const layers = new Set(), requests=[];
  let refreshes=0, fits=0;
  const map={setView(){return this;},removeLayer(l){layers.delete(l);},invalidateSize(){refreshes++;},fitBounds(){fits++;}};
  const ctx=vm.createContext({document:{getElementById:el,addEventListener(){}},window:{addEventListener(){}}, AbortController, setTimeout,clearTimeout,requestAnimationFrame:fn=>fn(),ResizeObserver:class{observe(){}},L:{map:()=>map,tileLayer:()=>({addTo(){}}),polyline:(points,options)=>({points,options,addTo(){layers.add(this);return this;}}),latLngBounds:p=>p},fetch:async (url,options)=>{
    if(url==='/api/status') return {json:async()=>({message:'ready'})};
    if(url.startsWith('/api/search')) return {ok:true,json:async()=>[]};
    const payload=JSON.parse(options.body); requests.push(payload);
    return {ok:true,status:200,json:async()=>({coordinates:[[23,72],[23.1,72.1]],distance_km:payload.mode==='distance'?2:3,estimated_minutes:payload.mode==='distance'?8:6,note:'ML estimate'})};
  }});
  vm.runInContext(fs.readFileSync('public/script.js','utf8'),ctx);
  vm.runInContext('startPlace={lat:23,lon:72};endPlace={lat:23.1,lon:72.1};',ctx);
  return {ctx,el,layers,requests,counts:()=>({refreshes,fits})};
}
test('both modes use selected algorithm, render API metrics and fit both layers',async()=>{
  for(const algorithm of ['astar','dijkstra']) {
    const h=setup();h.el('algorithm').value=algorithm;
    await vm.runInContext('calculate()',h.ctx);
    assert.deepEqual(h.requests.map(r=>r.mode),['distance','time']);
    assert.ok(h.requests.every(r=>r.algorithm===algorithm));
    assert.equal(h.layers.size,2);
    assert.deepEqual([...h.layers].map(l=>l.options.color),['#2864ed','#159957']);
    assert.equal(h.el('distanceDistance').textContent,'2 km');
    assert.equal(h.el('timeDuration').textContent,'6 min');
    assert.ok(h.counts().refreshes>0 && h.counts().fits>0);
    await vm.runInContext('calculate()',h.ctx);
    assert.equal(h.layers.size,2);
    vm.runInContext('clearRoute()',h.ctx);
    assert.equal(h.layers.size,0);
  }
});
test('location changes suppress late responses and release loading state',async()=>{
  const h=setup();
  const pending=vm.runInContext('calculate()',h.ctx);
  vm.runInContext('clearRoute()',h.ctx);
  await pending;
  assert.equal(h.layers.size,0);
  assert.equal(h.el('routeButton').disabled,false);
});
test('failed requests do not leave partial routes',async()=>{
  const h=setup(); h.ctx.fetch=async()=>{throw new Error('Network unavailable');};
  await vm.runInContext('calculate()',h.ctx);
  assert.equal(h.layers.size,0);
  assert.equal(h.el('message').textContent,'Network unavailable');
  assert.equal(h.el('routeButton').disabled,false);
});
