import argparse, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, log_loss, accuracy_score
import joblib

FEATURES=['distance_m','heading_error','segment_length_m','gps_step_m','speed_mps','candidate_rank','projection_fraction']

def read_trip(z, prefix):
    def load(ext, dtype=float):
        with z.open(f'{prefix}/{Path(prefix).name}.{ext}') as f:
            return np.loadtxt(f, dtype=dtype, ndmin=2)
    return load('track'),load('nodes'),load('arcs',int),load('route',int).astype(int).ravel()

def xy(coords, lat0):
    return np.column_stack((coords[:,0]*111320*np.cos(np.deg2rad(lat0)), coords[:,1]*110540))

def create_rows(track,nodes,arcs,route,trip_id,k=8,stride=5,max_points=150,training=True):
    # KDTree of segment midpoints; this is a fast baseline, not guaranteed exact nearest-segment search.
    if len(track)==0 or len(arcs)==0: return pd.DataFrame()
    lat0=float(np.median(nodes[:,1])); pts=xy(track[:,:2],lat0); nd=xy(nodes,lat0)
    valid=(arcs>=0).all(axis=1)&(arcs<len(nd)).all(axis=1)
    arc_ids=np.flatnonzero(valid); edges=arcs[valid]
    a,b=nd[edges[:,0]],nd[edges[:,1]]
    mids=(a+b)/2; tree=cKDTree(mids)
    # Ground-truth route arcs are used ONLY to create proxy training/evaluation labels.
    route_ids=np.unique(route[(route>=0)&(route<len(arcs))]); route_ids=route_ids[valid[route_ids]]
    if training and len(route_ids)==0: return pd.DataFrame()
    route_a=nd[arcs[route_ids,0]] if len(route_ids) else np.empty((0,2))
    route_b=nd[arcs[route_ids,1]] if len(route_ids) else np.empty((0,2))
    route_tree=cKDTree((route_a+route_b)/2) if len(route_ids) else None
    idx=np.arange(0,len(pts),stride)
    if len(idx)>max_points: idx=idx[np.linspace(0,len(idx)-1,max_points).astype(int)]
    out=[]; previous=None
    for t in idx:
        p=pts[t]; prev=pts[max(t-1,0)]; delta=p-prev
        dt=max(float(track[t,2]-track[max(t-1,0),2]),1.)
        speed=float(np.linalg.norm(delta)/dt)
        # Include closest segment midpoint candidates; proxy labels may be missing.
        _,ix=tree.query(p,k=min(k,len(mids)))
        ix=np.atleast_1d(ix)
        # proxy reference: closest geometrical segment among route IDs (not true per-observation ground truth)
        proxy=-1
        if training:
            _,rix=route_tree.query(p,k=min(32,len(route_ids)))
            rix=np.atleast_1d(rix); u=route_b[rix]-route_a[rix]
            f=np.clip(np.sum((p-route_a[rix])*u,axis=1)/(np.sum(u*u,axis=1)+1e-9),0,1)
            d=np.linalg.norm(p-(route_a[rix]+f[:,None]*u),axis=1)
            proxy=int(route_ids[rix[np.argmin(d)]])
        for rank,j in enumerate(ix):
            v=b[j]-a[j]; seglen=float(np.linalg.norm(v)); frac=float(np.clip(np.dot(p-a[j],v)/(seglen**2+1e-9),0,1))
            dist=float(np.linalg.norm(p-(a[j]+frac*v)))
            cos=float(abs(np.dot(delta,v))/(np.linalg.norm(delta)*seglen+1e-9))
            out.append(dict(trip=trip_id,obs=int(t),arc_id=int(arc_ids[j]),distance_m=dist,
                heading_error=float(1-cos),segment_length_m=seglen,gps_step_m=float(np.linalg.norm(delta)),
                speed_mps=speed,candidate_rank=rank,projection_fraction=frac,
                label=int(arc_ids[j]==proxy) if training else -1,proxy_arc=proxy))
    return pd.DataFrame(out)

def trip_names(z,partition):
    prefix=f'Student folder/{partition}/'
    return sorted({ '/'.join(n.split('/')[:3]) for n in z.namelist() if n.startswith(prefix) and n.endswith('.track')})

def build_data(zip_path,partition,k,stride,max_points):
    frames=[]; stats=[]
    with zipfile.ZipFile(zip_path) as z:
        names=trip_names(z,partition)
        for i,name in enumerate(names):
            try:
                tr,no,ar,ro=read_trip(z,name)
                df=create_rows(tr,no,ar,ro,Path(name).name,k,stride,max_points)
                if len(df):
                    miss=1-df.groupby(['trip','obs']).label.max().mean()
                    stats.append({'trip':Path(name).name,'observations':len(tr),'nodes':len(no),'arcs':len(ar),'route_arcs':len(ro),'proxy_candidate_miss_rate':float(miss)})
                    # No positive candidate => exclude from supervised candidate training, still count in miss-rate.
                    valid_obs=df.groupby(['trip','obs']).label.transform('max').eq(1)
                    frames.append(df[valid_obs].copy())
                print(f'[{partition}] {i+1}/{len(names)} {Path(name).name}',flush=True)
            except Exception as e: print(f'WARNING skipped {name}: {e}',flush=True)
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),pd.DataFrame(stats)

def score(model,df):
    prob=np.clip(model.predict_proba(df[FEATURES])[:,1],1e-6,1-1e-6)
    pred=(prob>=0.5).astype(int); y=df.label.to_numpy()
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tmp=df[['trip','obs','label']].copy();tmp['prob']=prob
    best=tmp.loc[tmp.groupby(['trip','obs']).prob.idxmax()]
    return dict(precision=float(p),recall=float(r),f1=float(f),log_loss=float(log_loss(y,prob,labels=[0,1])),
                candidate_accuracy=float(accuracy_score(y,pred)),observation_top1_proxy_accuracy=float(best.label.mean()))

def models():
    return {
      'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=350,class_weight='balanced',random_state=42)),
      'decision_tree':DecisionTreeClassifier(max_depth=10,min_samples_leaf=15,class_weight='balanced',random_state=42),
      'random_forest':RandomForestClassifier(n_estimators=100,max_depth=14,min_samples_leaf=3,n_jobs=-1,class_weight='balanced_subsample',random_state=42),
      'extra_trees':ExtraTreesClassifier(n_estimators=100,max_depth=14,min_samples_leaf=3,n_jobs=-1,class_weight='balanced',random_state=42),
      'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=23,l2_regularization=1,random_state=42)
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',default='Student.zip');ap.add_argument('--out',default='results')
    ap.add_argument('--k',type=int,default=8);ap.add_argument('--stride',type=int,default=8);ap.add_argument('--max-points',type=int,default=90)
    ap.add_argument('--folds',type=int,default=3);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,ts=build_data(args.zip,'train',args.k,args.stride,args.max_points)
    test,ss=build_data(args.zip,'student_test',args.k,args.stride,args.max_points)
    if train.empty or test.empty: raise RuntimeError('No usable data; inspect ZIP structure and logs')
    ts.to_csv(out/'train_trip_stats.csv',index=False);ss.to_csv(out/'test_trip_stats.csv',index=False)
    cv=GroupKFold(n_splits=min(args.folds,train.trip.nunique()))
    rows=[];winning=(-1,None,None)
    for name,model in models().items():
        fold_scores=[];t0=time.perf_counter()
        for ix,jx in cv.split(train[FEATURES],train.label,groups=train.trip):
            model.fit(train.iloc[ix][FEATURES],train.iloc[ix].label)
            fold_scores.append(score(model,train.iloc[jx]))
        cv_top1=float(np.mean([v['observation_top1_proxy_accuracy'] for v in fold_scores]))
        model.fit(train[FEATURES],train.label)
        test_scores=score(model,test);elapsed=time.perf_counter()-t0
        row={'model':name,'cv_top1_proxy_accuracy':cv_top1,'train_and_cv_seconds':elapsed,**{'test_'+k:v for k,v in test_scores.items()}}
        rows.append(row);print(json.dumps(row),flush=True)
        joblib.dump(model,out/f'{name}.joblib')
        if cv_top1>winning[0]:winning=(cv_top1,name,model)
    pd.DataFrame(rows).sort_values('cv_top1_proxy_accuracy',ascending=False).to_csv(out/'model_comparison.csv',index=False)
    joblib.dump(winning[2],out/'best_model.joblib')
    (out/'metadata.json').write_text(json.dumps({'best_model':winning[1],'features':FEATURES,'seed':42,
      'label_caveat':'Proxy labels: closest reference-route segment by geometric distance, not verified observation ground truth',
      'candidate_caveat':'Midpoint KDTree can omit truly nearest segments; miss rate is against proxy labels',
      'selection':'highest group CV proxy observation top-1','args':vars(args)},indent=2))
    print('FINISHED. Selected:',winning[1],'results:',out.resolve())
if __name__=='__main__':main()
