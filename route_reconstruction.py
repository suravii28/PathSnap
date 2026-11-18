"""Projection-aware road-network route expansion (undirected approximation).

Each observation is projected onto its selected road segment. Consecutive
projections are connected by the shortest graph walk, with partial travel on
first/last segments. No full traversal of each selected arc is forced.
"""
import heapq
import math
from collections import defaultdict
import numpy as np


def reconstruct_route(nodes, arcs, selected_arc_ids, gps_points=None):
    selected = [int(a) for a in selected_arc_ids]
    valid = (arcs >= 0).all(axis=1) & (arcs < len(nodes)).all(axis=1)
    if any(a < 0 or a >= len(arcs) or not valid[a] for a in selected):
        raise ValueError('Selected road arc contains invalid node IDs')
    if gps_points is not None and len(gps_points) != len(selected):
        raise ValueError('gps_points must correspond to selected_arc_ids')
    empty = dict(connected=False, gaps=[], route_arc_ids=[], route_coordinates=[],
                 route_parts=[], selected_arc_count=len(selected), expanded_arc_count=0,
                 network_distance_m=0., total_route_distance_m=0., gps_distance_m=None,
                 detour_ratio=None, transition_diagnostics=[], note='No selected arcs')
    if not selected:
        return empty

    lat0 = float(np.median(nodes[:, 1]))
    xy = np.column_stack((nodes[:, 0] * 111320 * np.cos(np.deg2rad(lat0)),
                          nodes[:, 1] * 110540))
    graph = defaultdict(list)
    lengths = {}
    for aid in np.flatnonzero(valid):
        u, v = map(int, arcs[aid, :2])
        length = float(np.linalg.norm(xy[u] - xy[v]))
        if length < 1e-8:
            continue
        lengths[int(aid)] = length
        graph[u].append((v, int(aid), length))
        graph[v].append((u, int(aid), length))

    def ll(point):
        return [float(point[1]), float(point[0])]

    projections = []
    for i, aid in enumerate(selected):
        u, v = map(int, arcs[aid, :2])
        a, b = xy[u], xy[v]
        if gps_points is None:
            f = 0.5
        else:
            gps = np.asarray(gps_points[i], dtype=float)
            # GPS points are supplied as [latitude, longitude], as in api.py.
            p = np.array([gps[1] * 111320 * np.cos(np.deg2rad(lat0)), gps[0] * 110540])
            f = float(np.clip(np.dot(p-a, b-a) / max(np.dot(b-a, b-a), 1e-12), 0, 1))
        projections.append((u, v, f, a + f * (b-a)))

    cache = {}
    def shortest(source, target):
        if source == target:
            return 0., [], [source]
        key = (source, target)
        if key in cache:
            return cache[key]
        heap = [(0., source)]
        dist = {source: 0.}
        prev = {}
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u] + 1e-9:
                continue
            if u == target:
                break
            for v, aid, w in graph[u]:
                nd = d + w
                if nd + 1e-9 < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, aid)
                    heapq.heappush(heap, (nd, v))
        if target not in dist:
            result = (math.inf, [], [])
        else:
            ids, verts, cur = [], [target], target
            while cur != source:
                parent, aid = prev[cur]
                ids.append(aid)
                verts.append(parent)
                cur = parent
            result = (dist[target], ids[::-1], verts[::-1])
        cache[key] = result
        return result

    parts = []
    current = [ll(projections[0][3])]
    route_ids = []
    gaps = []
    diagnostics = []
    total_distance = 0.
    bridge_distance = 0.
    gps_distance = 0.
    for i in range(1, len(selected)):
        u, v, f, p = projections[i-1]
        x, y, g, q = projections[i]
        start = ll(p)
        finish = ll(q)
        candidates = []
        # Same segment: move between the two projected positions directly.
        if selected[i-1] == selected[i]:
            candidates.append((abs(f-g) * lengths.get(selected[i], 0.),
                               [start, finish], [selected[i]], 0.))
        for source, partial_a in ((u, f * lengths.get(selected[i-1], 0.)),
                                  (v, (1-f) * lengths.get(selected[i-1], 0.))):
            for target, partial_b in ((x, g * lengths.get(selected[i], 0.)),
                                      (y, (1-g) * lengths.get(selected[i], 0.))):
                d, ids, verts = shortest(source, target)
                if math.isfinite(d):
                    coords = [start, ll(xy[source])]
                    coords.extend(ll(xy[n]) for n in verts[1:])
                    coords.append(finish)
                    candidates.append((partial_a + d + partial_b, coords,
                                       [selected[i-1]] + ids + [selected[i]], d))
        if not candidates:
            gaps.append({'between_observations': [i-1, i],
                         'from_arc': selected[i-1], 'to_arc': selected[i]})
            parts.append(current)
            current = [finish]
            continue
        dist, coords, ids, internal = min(candidates, key=lambda item: item[0])
        total_distance += dist
        bridge_distance += internal
        route_ids.extend(ids)
        for coord in coords[1:]:
            if not current or coord != current[-1]:
                current.append(coord)
        gps_step = float(np.linalg.norm(p-q))
        gps_distance += gps_step
        diagnostics.append({'between_observations': [i-1, i],
                            'road_distance_m': round(dist, 2),
                            'gps_displacement_m': round(gps_step, 2),
                            'detour_ratio': round(dist/max(gps_step, 1.), 2)})
    parts.append(current)
    ratio = total_distance / max(gps_distance, 1.)
    return dict(connected=len(gaps) == 0 and len(parts) == 1,
                gaps=gaps, route_arc_ids=route_ids,
                route_coordinates=parts[0] if len(parts) == 1 else [],
                route_parts=parts, selected_arc_count=len(selected),
                expanded_arc_count=len(route_ids),
                network_distance_m=round(bridge_distance, 2),
                total_route_distance_m=round(total_distance, 2),
                gps_distance_m=round(gps_distance, 2),
                detour_ratio=round(ratio, 3),
                transition_diagnostics=diagnostics,
                note=('Projection-aware undirected shortest-path reconstruction. '
                      'Does not enforce one-way restrictions or prove driven-route accuracy. '
                      'Arc count includes traversed partial segments and repetitions.'))
