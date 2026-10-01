"""Palmira HTML game: turn raw OSM + Microsoft footprints into compact game data.

Inputs (raw, kept outside the repo):  osm.json (Overpass `out geom`), msft_centro.geojsonl
Output: ../game/palmira_centro.json

Coordinates: local Transverse Mercator centred on Parque Bolívar (GRS80, k=1),
1 unit = 1 m, x = East, y = North. Values are stored in decimetres (ints).

Fidelity flags on every element:
  roads   w_est=1 when width was estimated from road class (no width/lanes tag)
  roads   o: 1 / -1 one-way as tagged in OSM, 0 = two-way (tagged no or untagged -> estimated)
  bldg    s: 'o' OSM footprint, 'm' Microsoft footprint; hs: 'tag' (OSM levels/height) or 'est'
"""
import json, sys, math, hashlib, pathlib, collections
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from shapely import make_valid
from pyproj import Transformer, CRS

RAW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = pathlib.Path(__file__).resolve().parent.parent / "game" / "palmira_centro.json"
LAT0, LON0 = 3.5274, -76.3007
LOC = CRS.from_proj4(f"+proj=tmerc +lat_0={LAT0} +lon_0={LON0} +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
T = Transformer.from_crs(4326, LOC, always_xy=True)
# play area (metres, local): a little inside the downloaded bbox so nothing is cut at the edge
AREA = box(-1500, -1450, 1500, 1500)

CAR = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
       "living_street", "service", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
PED = {"pedestrian", "footway", "path", "steps", "cycleway"}
WIDTH = {"trunk": 14, "primary": 12, "secondary": 10, "tertiary": 9, "unclassified": 7.5,
         "residential": 7.5, "living_street": 6, "service": 4.5, "pedestrian": 7, "footway": 2.5,
         "path": 2, "steps": 2.5, "cycleway": 2.5}
for k in list(WIDTH):
    WIDTH.setdefault(k + "_link", WIDTH[k] * 0.7)
WIDTH.update({"trunk_link": 8, "primary_link": 8, "secondary_link": 7, "tertiary_link": 7})


def P(lon, lat):
    return T.transform(lon, lat)


def dm(v):
    return int(round(v * 10))


def flat(coords):
    out = []
    for x, y in coords:
        out += [dm(x), dm(y)]
    return out


def ring_out(poly, tol=0.15):
    poly = poly.simplify(tol, preserve_topology=True)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    ext = list(poly.exterior.coords)[:-1]
    if len(ext) < 3:
        return None
    holes = [flat(list(h.coords)[:-1]) for h in poly.interiors if len(h.coords) > 3]
    return {"p": flat(ext), "h": holes} if holes else {"p": flat(ext)}


def polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for gg in g.geoms for p in polys(gg)]
    return []


def hrand(s, n=1):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % n


osm = json.load(open(RAW / "osm.json"))
els = osm["elements"]
print("OSM", osm["osm3s"]["timestamp_osm_base"], len(els))

# ---------------------------------------------------------------- roads + graph
roads, ped = [], []
node_xy, node_use = {}, collections.Counter()
for e in els:
    t = e.get("tags", {})
    hw = t.get("highway")
    if e["type"] != "way" or not hw or "geometry" not in e:
        continue
    if hw not in CAR and hw not in PED:
        continue
    if t.get("area") == "yes":
        continue
    pts = [P(g["lon"], g["lat"]) for g in e["geometry"]]
    line = LineString(pts)
    if not line.intersects(AREA):
        continue
    w, w_est = WIDTH.get(hw, 6), 1
    if "width" in t:
        try:
            w, w_est = float(t["width"].split()[0]), 0
        except ValueError:
            pass
    elif "lanes" in t and hw in CAR:
        try:
            w, w_est = max(4.0, int(t["lanes"]) * 3.3 + 1.0), 0
        except ValueError:
            pass
    ow = t.get("oneway")
    o = 1 if ow in ("yes", "true", "1") else -1 if ow == "-1" else 0
    if t.get("junction") == "roundabout" and ow is None:
        o = 1
    rec = dict(id=e["id"], hw=hw, name=t.get("name", ""), w=w, w_est=w_est, o=o, nodes=e["nodes"], pts=pts,
               lanes=t.get("lanes"), surface=t.get("surface"), maxspeed=t.get("maxspeed"))
    (roads if hw in CAR else ped).append(rec)
    if hw in CAR:
        for nid, xy in zip(e["nodes"], pts):
            node_xy[nid] = xy
            node_use[nid] += 1
        node_use[e["nodes"][0]] += 1
        node_use[e["nodes"][-1]] += 1
print("car ways", len(roads), "ped ways", len(ped))

# split ways at junctions -> graph edges
junction = {n for n, c in node_use.items() if c >= 2}
gnode_index, gnodes = {}, []


def gid(n):
    if n not in gnode_index:
        gnode_index[n] = len(gnodes)
        x, y = node_xy[n]
        gnodes.append([dm(x), dm(y)])
    return gnode_index[n]


edges = []
for ri, r in enumerate(roads):
    seg = [0]
    for i in range(1, len(r["nodes"])):
        seg.append(i)
        if r["nodes"][i] in junction or i == len(r["nodes"]) - 1:
            if len(seg) >= 2:
                a, b = r["nodes"][seg[0]], r["nodes"][seg[-1]]
                pts = [r["pts"][k] for k in seg]
                L = LineString(pts).length
                if L > 0.5:
                    edges.append([gid(a), gid(b), ri, flat(pts), round(L, 1)])
            seg = [i]

# traffic signals snapped to graph nodes
signals = []
for e in els:
    t = e.get("tags", {})
    if e["type"] == "node" and t.get("highway") == "traffic_signals":
        if e["id"] in gnode_index:
            signals.append(gnode_index[e["id"]])
        else:
            x, y = P(e["lon"], e["lat"])
            best = min(range(len(gnodes)), key=lambda i: (gnodes[i][0] / 10 - x) ** 2 + (gnodes[i][1] / 10 - y) ** 2)
            if math.dist((gnodes[best][0] / 10, gnodes[best][1] / 10), (x, y)) < 15:
                signals.append(best)
signals = sorted(set(signals))
print("graph nodes", len(gnodes), "edges", len(edges), "signals", len(signals))

# ---------------------------------------------------------------- blocks (everything that is not carriageway)
road_buf = unary_union([LineString(r["pts"]).buffer(r["w"] / 2, cap_style=1, join_style=1, quad_segs=3) for r in roads])
blocks_geom = AREA.difference(road_buf)
blocks = [p for p in polys(blocks_geom) if p.area > 8]
print("blocks", len(blocks))

# pedestrian streets / plazas drawn as paving ribbons
ped_out = []
for r in ped:
    if r["hw"] in ("pedestrian", "footway", "steps") and LineString(r["pts"]).length > 3:
        ped_out.append({"w": round(r["w"], 1), "t": r["hw"][0], "p": flat(LineString(r["pts"]).intersection(AREA).coords) if LineString(r["pts"]).within(AREA) else flat(r["pts"]), "n": r["name"]})

# ---------------------------------------------------------------- areas: parks, landmarks
parks, landuse, landmarks = [], [], []
LANDMARK_NAMES = {
    "Parque Bolívar": "park", "CAMP": "camp", "Catedral de Nuestra Señora del Rosario del Palmar": "cathedral",
    "Galería Central de Palmira": "market", "La Santísima Trinidad": "church", "Iglesia De los Carmelos": "church",
    "Estadio Francisco Rivera Escobar": "stadium",
}
for e in els:
    t = e.get("tags", {})
    if e["type"] != "way" or "geometry" not in e or len(e["geometry"]) < 4:
        continue
    pts = [P(g["lon"], g["lat"]) for g in e["geometry"]]
    if pts[0] != pts[-1]:
        continue
    try:
        poly = make_valid(Polygon(pts))
    except Exception:
        continue
    poly = poly.intersection(AREA)
    if poly.is_empty:
        continue
    name = t.get("name", "")
    if name in LANDMARK_NAMES:
        for p in polys(poly):
            ro = ring_out(p, 0.05)
            if ro:
                landmarks.append({"k": LANDMARK_NAMES[name], "n": name, "id": f"osm:way/{e['id']}", **ro})
    if t.get("leisure") in ("park", "garden", "pitch", "playground") or t.get("landuse") in ("grass", "recreation_ground", "village_green") or t.get("leisure") == "stadium":
        kind = "pitch" if t.get("leisure") in ("pitch", "stadium") else "park"
        for p in polys(poly):
            ro = ring_out(p)
            if ro:
                parks.append({"k": kind, "n": name, **ro})
    elif t.get("landuse") in ("farmland", "meadow", "farmyard", "orchard") or t.get("natural") in ("grassland", "scrub", "wood"):
        for p in polys(poly):
            ro = ring_out(p, 0.5)
            if ro:
                landuse.append({"k": "field", **ro})

# ---------------------------------------------------------------- buildings
osm_b = []
for e in els:
    t = e.get("tags", {})
    if e["type"] == "way" and "building" in t and "geometry" in e and len(e["geometry"]) >= 4:
        pts = [P(g["lon"], g["lat"]) for g in e["geometry"]]
        try:
            poly = make_valid(Polygon(pts))
        except Exception:
            continue
        for p in polys(poly):
            if p.area > 6 and p.intersects(AREA):
                osm_b.append((p, t, e["id"]))
osm_tree = STRtree([p for p, _, _ in osm_b]) if osm_b else None

ms_b = []
for i, line in enumerate(open(RAW / "msft_centro.geojsonl")):
    g = json.loads(line)
    pts = [P(x, y) for x, y in g["geometry"]["coordinates"][0]]
    p = make_valid(Polygon(pts))
    for pp in polys(p):
        if pp.area > 6 and AREA.contains(pp.centroid):
            ms_b.append((pp, i))
print("osm bldgs", len(osm_b), "msft bldgs", len(ms_b))

lm_polys = {l["k"]: Polygon([(l["p"][i] / 10, l["p"][i + 1] / 10) for i in range(0, len(l["p"]), 2)]) for l in landmarks}
park_poly = lm_polys.get("park")


def est_height(poly, key):
    """Estimated height (m). Rule documented in README: storeys by footprint size and distance from the park."""
    a = poly.area
    d = math.hypot(poly.centroid.x, poly.centroid.y)
    r = hrand(key, 1000) / 1000
    if a < 40:
        st = 1
    elif a < 160:
        st = 1 + (r < (0.55 if d < 600 else 0.25))
    elif a < 600:
        st = 2 + (r < (0.45 if d < 600 else 0.15)) + (r < (0.15 if d < 400 else 0.0))
    else:
        st = 2 + int(r * (4 if d < 500 else 2))
    if st == 1:
        return 3.6 + r * 1.2, st
    return st * 3.1 + 0.8 + r * 0.6, st


bld_out = []
road_tree = STRtree([LineString(r["pts"]) for r in roads])


def on_road(p):
    # drop footprints whose interior is crossed by a carriageway centreline (misdetections)
    for idx in road_tree.query(p):
        if LineString(roads[idx]["pts"]).intersection(p.buffer(-1.0)).length > 2:
            return True
    return False


GRID_ROT = 8.5  # degrees, street grid rotation (clockwise from north)
LOT = 11.0      # approximate lot frontage used to split merged row-house footprints


def subdivide(p):
    """Split a large merged footprint into ~LOT m cells aligned with the street grid (approximation)."""
    from shapely import affinity
    from shapely.geometry import box as sbox
    c = p.centroid
    r = affinity.rotate(p, GRID_ROT, origin=c)          # undo the clockwise grid rotation
    x0, y0, x1, y1 = r.bounds
    cells = []
    nx, ny = max(1, round((x1 - x0) / LOT)), max(1, round((y1 - y0) / LOT))
    sx, sy = (x1 - x0) / nx, (y1 - y0) / ny
    outer = r.exterior
    for i in range(nx):
        for j in range(ny):
            cell = r.intersection(sbox(x0 + i * sx, y0 + j * sy, x0 + (i + 1) * sx, y0 + (j + 1) * sy))
            for q in polys(cell):
                if q.area > 12:
                    edge = q.buffer(0.3).intersects(outer)      # touches the street-facing outline
                    cells.append((affinity.rotate(q, -GRID_ROT, origin=c), edge))
    return cells


SIDEWALK = 2.2  # m kept free between kerb and facades (footprints are ML-detected; trimmed so sidewalks are walkable)
sidewalk_band = road_buf.buffer(SIDEWALK, quad_segs=2)


def add_bldg(p, src, key, h, hs, st, kind, split=True):
    if park_poly is not None and park_poly.buffer(-1).contains(p.centroid):
        return
    if p.intersects(sidewalk_band):
        trimmed = [q for q in polys(make_valid(p.difference(sidewalk_band))) if q.area > 10]
        if not trimmed:
            return
        if len(trimmed) > 1 or trimmed[0].area < p.area * 0.999:
            for n, q in enumerate(trimmed):
                add_bldg(q, src, f"{key}~{n}" if len(trimmed) > 1 else key, h, hs, st, kind, split)
            return
    if split and hs == "est" and p.area > 450 and kind in ("yes", "house", "residential", "commercial", "retail", "apartments"):
        for n, (q, edge) in enumerate(subdivide(p)):
            k2 = f"{key}#{n}"
            r = hrand(k2, 1000) / 1000
            stq = max(1, st + (1 if r > 0.82 else -1 if r < 0.3 else 0) - (0 if edge else 1))
            hq = (3.6 + r * 1.0) if stq == 1 else stq * 3.1 + 0.8 + r * 0.5
            ro = ring_out(q, 0.2)
            if ro:
                bld_out.append({**ro, "hgt": dm(hq), "st": stq, "s": src, "hs": "est", "k": kind, "id": k2, "sub": 1})
        return
    ro = ring_out(p, 0.2)
    if ro:
        bld_out.append({**ro, "hgt": dm(h), "st": st, "s": src, "hs": hs, "k": kind, "id": key})


dropped = collections.Counter()
for p, t, wid in osm_b:
    if on_road(p):
        dropped["osm_on_road"] += 1
        continue
    lv = t.get("building:levels")
    h, hs = None, "est"
    try:
        if "height" in t:
            h, hs = float(t["height"].split()[0]), "tag"
        elif lv:
            h, hs = float(lv) * 3.1 + 0.8, "tag"
    except ValueError:
        pass
    if h is None:
        h, st = est_height(p, f"o{wid}")
    st = max(1, round((h - 0.8) / 3.1))
    kind = t.get("building", "yes")
    if kind == "church" or t.get("amenity") == "place_of_worship":
        kind = "church"
    add_bldg(p, "o", f"osm:way/{wid}", h, hs, st, kind)

for p, i in ms_b:
    if osm_tree is not None:
        hit = [j for j in osm_tree.query(p) if osm_b[j][0].intersection(p).area > 0.3 * p.area]
        if hit:
            dropped["ms_dupe_osm"] += 1
            continue
    if on_road(p):
        dropped["ms_on_road"] += 1
        continue
    key = f"msft:032232031/{i}"
    kind = "yes"
    if "cathedral" in lm_polys and lm_polys["cathedral"].buffer(2).contains(p.centroid):
        dropped["ms_in_cathedral"] += 1  # cathedral is a hand-built landmark
        continue
    if "camp" in lm_polys and lm_polys["camp"].buffer(2).contains(p.centroid) and p.area > 300:
        # CAMP: at least 9 storeys per Alcaldía description (research doc); exact height pending
        add_bldg(p, "m", key, 9 * 3.3 + 1, "est", 9, "civic")
        continue
    h, st = est_height(p, key)
    add_bldg(p, "m", key, h, "est", st, kind)
print("buildings out", len(bld_out), dict(dropped))

# ---------------------------------------------------------------- POIs & trees
pois, trees = [], []
for e in els:
    t = e.get("tags", {})
    if e["type"] == "node":
        x, y = P(e["lon"], e["lat"])
        if not AREA.contains(Point(x, y)):
            continue
        if t.get("natural") == "tree":
            trees.append([dm(x), dm(y)])
        elif t.get("name") and (t.get("amenity") or t.get("shop") or t.get("historic") or t.get("tourism")):
            pois.append({"n": t["name"], "t": t.get("amenity") or t.get("shop") or t.get("historic") or t.get("tourism"), "x": dm(x), "y": dm(y)})

# ---------------------------------------------------------------- block surface kind + sidewalk walking rings
bc_tree = STRtree([Polygon([(b["p"][i] / 10, b["p"][i + 1] / 10) for i in range(0, len(b["p"]), 2)]).centroid for b in bld_out])
blocks_out, walks = [], []
for b in blocks:
    ro = ring_out(b, 0.1)
    if not ro:
        continue
    n_b = len(bc_tree.query(b, predicate="contains"))
    ro["g"] = 1 if (n_b == 0 and b.area > 2500) else 0      # empty large block -> grass/lot
    blocks_out.append(ro)
    if n_b > 0 and b.area > 300:
        inner = b.buffer(-1.1, join_style=2)
        for ip in polys(inner):
            if ip.exterior.length > 40:
                walks.append(flat(list(ip.simplify(0.5).exterior.coords)[:-1]))
print("walk rings", len(walks))

out = {
    "meta": {
        "name": "Palmira Centro", "units": "decimetres", "crs": LOC.to_proj4(),
        "origin": {"lat": LAT0, "lon": LON0, "label": "Parque Bolívar"},
        "osm_timestamp": osm["osm3s"]["timestamp_osm_base"],
        "msft_release": "Global ML Building Footprints 2026-02-03, quadkey 032232031",
        "attribution": [
            "Map data © OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)",
            "Building footprints © Microsoft, Global ML Building Footprints, CDLA-Permissive-2.0",
        ],
        "area": [-1500, -1450, 1500, 1500],
    },
    "roads": [{"n": r["name"], "c": r["hw"], "w": round(r["w"], 1), "we": r["w_est"], "o": r["o"], "id": r["id"]} for r in roads],
    "gnodes": gnodes, "edges": edges, "signals": signals,
    "blocks": blocks_out, "walks": walks,
    "ped": ped_out, "parks": parks, "fields": landuse, "landmarks": landmarks,
    "buildings": bld_out, "pois": pois, "trees": trees,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
