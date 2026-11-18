import os, zipfile, json, time
from pathlib import Path
from functools import lru_cache
import numpy as np
import joblib
import folium
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from train import trip_names, read_trip
from decode import decode_trip
from route_reconstruction import reconstruct_route

ROOT=Path(__file__).resolve().parent
DATASET=Path(os.environ.get('PATHSNAP_DATASET',str(Path.home()/'Downloads'/'Student.zip'))).expanduser()
MODEL=Path(os.environ.get('PATHSNAP_MODEL',str(ROOT/'models'/'tuned_hist_gradient_boosting.joblib'))).expanduser()
if not MODEL.exists():
    MODEL=ROOT/'models'/'best_model.joblib'
app=FastAPI(title='PathSnap API',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET'],allow_headers=['*'])

@lru_cache(maxsize=1)
def get_model():
    if not MODEL.exists(): raise RuntimeError(f'Model missing: {MODEL}')
    return joblib.load(MODEL)

def trip_index():
    if not DATASET.exists(): raise RuntimeError(f'Dataset missing: {DATASET}')
    with zipfile.ZipFile(DATASET) as z:
        return {Path(name).name:name for name in trip_names(z,'student_test')}

@app.get('/health')
def health():
    return {'status':'ok','dataset_found':DATASET.exists(),'model_found':MODEL.exists(),'dataset':str(DATASET),'model':str(MODEL)}

@app.get('/trips')
def trips():
    try: return {'trips':sorted(trip_index())}
    except Exception as e: raise HTTPException(500,str(e))

@lru_cache(maxsize=12)
def infer(trip_id):
    names=trip_index()
    if trip_id not in names: raise ValueError('Unknown student-testing trip')
    with zipfile.ZipFile(DATASET) as z:
        track,nodes,arcs,route=read_trip(z,names[trip_id])
    start=time.perf_counter()
    df,obs,decoded,other=decode_trip(get_model(),track,nodes,arcs,route,trip_id,k=8,stride=8,max_points=90,transition_weight=1.0)
    if not obs: raise ValueError('No observations decoded')
    def arc_lines(ids):
        lines=[]
        for aid in ids:
            u,v=arcs[int(aid)]
            if 0<=u<len(nodes) and 0<=v<len(nodes):
                lines.append([[float(nodes[u,1]),float(nodes[u,0])],[float(nodes[v,1]),float(nodes[v,0])]])
        return lines
    # Reference route used only for evaluation/display, never to select predicted arcs.
    reference=[int(a) for a in route if 0<=a<len(arcs)]
    reconstructed=reconstruct_route(nodes,arcs,decoded, gps_points=[[float(track[i,1]),float(track[i,0])] for i in obs])
    return {'trip':trip_id,'observations_total':int(len(track)),'observations_decoded':len(obs),
            'runtime_seconds':round(time.perf_counter()-start,3),
            'metrics':other['metrics'],
            'gps':[[float(track[i,1]),float(track[i,0])] for i in obs],
            'greedy_segments':arc_lines(other['greedy']),
            'viterbi_segments':arc_lines(decoded),
            'reference_segments':arc_lines(reference),
            'reconstruction':reconstructed,
            'predictions':[{'obs':int(o),'greedy_arc':int(g),'viterbi_arc':int(v)} for o,g,v in zip(obs,other['greedy'],decoded)]}

@app.get('/predict/{trip_id}')
def predict(trip_id:str):
    try: return infer(trip_id)
    except ValueError as e: raise HTTPException(404,str(e))
    except Exception as e: raise HTTPException(500,str(e))

def make_map(data):
    gps=data['gps']; center=gps[len(gps)//2] if gps else [0,0]
    m=folium.Map(location=center,zoom_start=14,control_scale=True,tiles='OpenStreetMap')
    for name,key,color in [('Reference route','reference_segments','green'),('Greedy prediction','greedy_segments','orange'),('Viterbi prediction','viterbi_segments','blue')]:
        layer=folium.FeatureGroup(name=name,show=(name!='Greedy prediction'))
        for line in data[key]: folium.PolyLine(line,color=color,weight=4,opacity=.7).add_to(layer)
        layer.add_to(m)
    expanded=folium.FeatureGroup(name='Expanded road-network route',show=True)
    for part in data['reconstruction']['route_parts']:
        if len(part)>1: folium.PolyLine(part,color='purple',weight=6,opacity=.8).add_to(expanded)
    expanded.add_to(m)
    layer=folium.FeatureGroup(name='Sampled GPS points',show=True)
    for p in gps: folium.CircleMarker(p,radius=2,color='red',fill=True,fill_opacity=.7).add_to(layer)
    layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if gps: m.fit_bounds(gps)
    return m.get_root().render()

@app.get('/map/{trip_id}',response_class=HTMLResponse)
def map_view(trip_id:str):
    try: return HTMLResponse(make_map(infer(trip_id)))
    except ValueError as e: raise HTTPException(404,str(e))
    except Exception as e: raise HTTPException(500,str(e))
