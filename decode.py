import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from train import read_trip, trip_names, create_rows, xy, FEATURES


def decode_trip(model, track, nodes, arcs, route, trip_id, k, stride, max_points, transition_weight):
    df = create_rows(track, nodes, arcs, route, trip_id, k=k, stride=stride,
                     max_points=max_points, training=False)
    if df.empty: return df, [], [], {}
    df = df.sort_values(['obs','candidate_rank']).reset_index(drop=True)
    probs = np.clip(model.predict_proba(df[FEATURES])[:, 1], 1e-8, 1-1e-8)
    df['probability'] = probs
    lat0 = float(np.median(nodes[:, 1])); nd = xy(nodes, lat0)
    valid = (arcs >= 0).all(axis=1) & (arcs < len(nd)).all(axis=1)
    edge_lengths = np.zeros(len(arcs), dtype=float)
    edge_lengths[valid] = np.linalg.norm(nd[arcs[valid, 0]] - nd[arcs[valid, 1]], axis=1)
    # Undirected road graph: arc directionality is not supplied by this dataset.
    good = arcs[valid]
    rr = np.r_[good[:, 0], good[:, 1]]
    cc = np.r_[good[:, 1], good[:, 0]]
    ww = np.r_[edge_lengths[valid], edge_lengths[valid]]
    graph = coo_matrix((ww, (rr, cc)), shape=(len(nodes),len(nodes))).tocsr()
    graph.sum_duplicates()
    obs_ids = sorted(df.obs.unique())
    groups = [df[df.obs == o].copy().reset_index(drop=True) for o in obs_ids]
    # All candidate endpoint nodes are valid sources; cache distances for this trip.
    candidate_arcs = np.unique(df.arc_id.to_numpy(dtype=int))
    endpoint_nodes = np.unique(arcs[candidate_arcs].ravel())
    shortest = dijkstra(graph, directed=False, indices=endpoint_nodes)
    endpoint_index = {int(n): i for i,n in enumerate(endpoint_nodes)}
    def road_distance(aid, bid):
        if aid == bid: return 0.0
        u,v = arcs[aid]; x,y = arcs[bid]
        return min(shortest[endpoint_index[int(s)], int(t)] for s in (u,v) for t in (x,y))
    scores = [np.log(groups[0].probability.to_numpy())]
    backs = []
    pts = xy(track[:, :2], lat0)
    for i in range(1,len(groups)):
        prev, curr = groups[i-1],groups[i]
        step = float(np.linalg.norm(pts[obs_ids[i]] - pts[obs_ids[i-1]]))
        dt = max(float(track[obs_ids[i],2] - track[obs_ids[i-1],2]),1.0)
        # Transition penalty compares road-network distance with GPS displacement.
        # An additional speed bound penalizes implausible jumps.
        trans = np.empty((len(prev),len(curr)))
        for j,aid in enumerate(prev.arc_id):
            for h,bid in enumerate(curr.arc_id):
                rd = road_distance(int(aid),int(bid))
                if not np.isfinite(rd): trans[j,h] = -1e5
                else:
                    mismatch = abs(rd-step)/max(30.0,step)
                    speed_penalty = max(0.0, rd/dt - 45.0)/45.0
                    trans[j,h] = -transition_weight*(mismatch+speed_penalty)
        total = scores[-1][:,None] + trans
        backs.append(np.argmax(total,axis=0))
        scores.append(np.log(curr.probability.to_numpy())+np.max(total,axis=0))
    choice = [int(np.argmax(scores[-1]))]
    for back in reversed(backs): choice.append(int(back[choice[-1]]))
    choice.reverse()
    decoded = [int(g.iloc[c].arc_id) for g,c in zip(groups,choice)]
    greedy = [int(g.iloc[g.probability.argmax()].arc_id) for g in groups]
    # Proxy reference labels are evaluation-only; they never influence decoding.
    eval_df = create_rows(track,nodes,arcs,route,trip_id,k=k,stride=stride,max_points=max_points,training=True)
    proxy = eval_df.groupby('obs').proxy_arc.first().to_dict() if len(eval_df) else {}
    usable = [(o,p,g) for o,p,g in zip(obs_ids,decoded,greedy) if o in proxy and proxy[o]>=0]
    metrics = {'trip':trip_id,'observations_decoded':len(obs_ids),
               'greedy_proxy_top1':float(np.mean([g==proxy[o] for o,p,g in usable])) if usable else None,
               'viterbi_proxy_top1':float(np.mean([p==proxy[o] for o,p,g in usable])) if usable else None,
               'proxy_observations_scored':len(usable)}
    return df, obs_ids, decoded, {'greedy':greedy, 'metrics':metrics}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',default='Student.zip')
    ap.add_argument('--model',default='tuning_results/tuned_hist_gradient_boosting.joblib')
    ap.add_argument('--out',default='decoding_results')
    ap.add_argument('--partition',choices=['student_test','train'],default='student_test')
    ap.add_argument('--k',type=int,default=8)
    ap.add_argument('--stride',type=int,default=8)
    ap.add_argument('--max-points',type=int,default=90)
    ap.add_argument('--transition-weight',type=float,default=1.0)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model=joblib.load(args.model)
    summary=[]; predictions=[]
    with zipfile.ZipFile(args.zip) as z:
        names=trip_names(z,args.partition)
        if args.limit: names=names[:args.limit]
        for i,name in enumerate(names,1):
            trip=Path(name).name
            try:
                track,nodes,arcs,route=read_trip(z,name)
                df,obs,decoded,other=decode_trip(model,track,nodes,arcs,route,trip,args.k,args.stride,args.max_points,args.transition_weight)
                if not obs: continue
                summary.append(other['metrics'])
                for o,a,g in zip(obs,decoded,other['greedy']):
                    predictions.append({'trip':trip,'obs':o,'viterbi_arc':a,'greedy_arc':g})
                print(f'[{i}/{len(names)}] {trip} greedy={other["metrics"]["greedy_proxy_top1"]:.4f} viterbi={other["metrics"]["viterbi_proxy_top1"]:.4f}',flush=True)
            except Exception as exc:
                print(f'ERROR {trip}: {exc}',flush=True)
                raise
    pd.DataFrame(summary).to_csv(out/'trip_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    if summary:
        valid=[s for s in summary if s['proxy_observations_scored']]
        if valid:
            n=sum(s['proxy_observations_scored'] for s in valid)
            totals={key:sum(s[key]*s['proxy_observations_scored'] for s in valid)/n for key in ('greedy_proxy_top1','viterbi_proxy_top1')}
            print('Weighted proxy comparison:',json.dumps(totals,indent=2))
    print('Saved to:',out.resolve())

if __name__=='__main__': main()
