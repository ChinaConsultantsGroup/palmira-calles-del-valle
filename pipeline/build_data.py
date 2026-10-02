"""Palmira HTML game: turn raw OSM + Microsoft footprints into compact game data.

Inputs (raw, kept outside the repo): Overpass `out geom` JSON + Microsoft footprints geojsonl.
Env: PALMIRA_SCOPE=centro|ciudad selects input files, play area and output name.
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
import os
SCOPE = os.environ.get("PALMIRA_SCOPE", "centro")
CFG = {
    "centro": dict(osm="osm.json", msft="msft_centro.geojsonl", area=(-1500, -1450, 1500, 1500), out="palmira_centro.json"),
    # whole urban area of Palmira (DANE cabecera ~25 km2 sits inside this box) plus a margin of cane fields
    "ciudad": dict(osm="osm_city.json", msft="msft_city.geojsonl", area=(-4700, -3400, 4900, 5100), out="palmira.json"),
}[SCOPE]
OUT = pathlib.Path(__file__).resolve().parent.parent / "game" / CFG["out"]
LAT0, LON0 = 3.5274, -76.3007
LOC = CRS.from_proj4(f"+proj=tmerc +lat_0={LAT0} +lon_0={LON0} +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
T = Transformer.from_crs(4326, LOC, always_xy=True)
# play area (metres, local): a little inside the downloaded bbox so nothing is cut at the edge
AREA = box(*CFG["area"])

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


osm = json.load(open(RAW / CFG["osm"]))
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
# official signalised crossings (Alcaldía de Palmira, datos.gov.co 6hkg-wtby) — much more complete than OSM
OFFICIAL = RAW / "official"
signal_el = []
if (OFFICIAL / "semaforos.geojson").exists():
    sem = json.load(open(OFFICIAL / "semaforos.geojson"))
    gn_tree = STRtree([Point(x / 10, y / 10) for x, y in gnodes])
    n_off = 0
    for f in sem["features"]:
        x, y = P(*f["geometry"]["coordinates"][:2])
        if f["properties"].get("feature") == "crossing":
            if f["properties"].get("position_check", "ok") != "ok":
                continue
            i = gn_tree.nearest(Point(x, y))
            if math.dist((gnodes[i][0] / 10, gnodes[i][1] / 10), (x, y)) < 30:
                signals.append(int(i)); n_off += 1
        elif AREA.contains(Point(x, y)):
            signal_el.append([f["properties"].get("kind", ""), dm(x), dm(y)])
    print("official signal crossings snapped", n_off, "elements", len(signal_el))
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
        clip = LineString(r["pts"]).intersection(AREA)
        for part in (clip.geoms if hasattr(clip, "geoms") else [clip]):
            if part.geom_type == "LineString" and part.length > 3:
                ped_out.append({"w": round(r["w"], 1), "t": r["hw"][0], "p": flat(part.coords), "n": r["name"]})

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
for i, line in enumerate(open(RAW / CFG["msft"])):
    g = json.loads(line)
    pts = [P(x, y) for x, y in g["geometry"]["coordinates"][0]]
    p = make_valid(Polygon(pts))
    for pp in polys(p):
        if pp.area > 6 and AREA.contains(pp.centroid):
            ms_b.append((pp, i))
print("osm bldgs", len(osm_b), "msft bldgs", len(ms_b))

# Alcaldía de Palmira cadastre 2024 (LC_Construccion_ON): real footprints and number of floors. Wins over OSM/Microsoft.
cat_b, manz = [], []
if (OFFICIAL / "catastro_construcciones.geojson").exists():
    cj = json.load(open(OFFICIAL / "catastro_construcciones.geojson"))
    for f in cj["features"]:
        g = f["geometry"]; pr = f["properties"]
        rings = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for rg in rings:
            pts = [P(x, y) for x, y in rg[0]]
            if len(pts) < 4:
                continue
            for pp in polys(make_valid(Polygon(pts))):
                if pp.area > 4 and AREA.contains(pp.centroid):
                    cat_b.append((pp, pr))
    del cj
    for f in json.load(open(OFFICIAL / "catastro_manzanas.geojson"))["features"]:
        g = f["geometry"]; rings = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for rg in rings:
            manz.extend(polys(make_valid(Polygon([P(x, y) for x, y in rg[0]]))))
cat_tree = STRtree([p for p, _ in cat_b]) if cat_b else None
manz_tree = STRtree(manz) if manz else None
print("catastro parts", len(cat_b), "manzanas", len(manz))

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


def add_bldg(p, src, key, h, hs, st, kind, split=True, band=None):
    band = sidewalk_band if band is None else band
    if park_poly is not None and park_poly.buffer(-1).contains(p.centroid):
        return
    if p.intersects(band):
        trimmed = [q for q in polys(make_valid(p.difference(band))) if q.area > (4 if src == 'c' else 10)]
        if not trimmed:
            return
        if len(trimmed) > 1 or trimmed[0].area < p.area * 0.999:
            for n, q in enumerate(trimmed):
                add_bldg(q, src, f"{key}~{n}" if len(trimmed) > 1 else key, h, hs, st, kind, split, band)
            return
    if split and hs == "est" and p.area > 450 and kind in ("yes", "house", "residential", "commercial", "retail", "apartments"):
        for n, (q, edge) in enumerate(subdivide(p)):
            k2 = f"{key}#{n}"
            r = hrand(k2, 1000) / 1000
            stq = max(1, st + (1 if r > 0.82 else -1 if r < 0.3 else 0) - (0 if edge else 1))
            hq = (3.6 + r * 1.0) if stq == 1 else stq * 3.1 + 0.8 + r * 0.5
            ro = ring_out(q, 0.2)
            if ro:
                bld_out.append({**ro, "hgt": dm(hq), "st": stq, "s": src, "k": kind, **({"id": k2} if src == "o" else {})})
        return
    ro = ring_out(p, 0.2)
    if ro:
        bld_out.append({**ro, "hgt": dm(h), "st": st, "s": src, "k": kind, **({"hs": hs} if hs != "est" else {}), **({"id": key} if src == "o" else {})})


dropped = collections.Counter()
USO_KIND = {"Habitacional": "house", "Comercial": "commercial", "Industrial": "industrial", "Institucional": "public", "Educativo": "school",
            "Religioso": "church", "Cultural": "public", "Salubridad": "public", "Uso_Publico": "public", "Recreacional": "public", "Agricola": "farm"}
carriage_band = road_buf.buffer(0.9, quad_segs=2)   # cadastre outlines are surveyed: only trim what overlaps our (estimated-width) carriageway
for n, (p, pr) in enumerate(cat_b):
    fl = max(1, int(pr.get("floors") or 1))
    kind = USO_KIND.get(pr.get("uso"), "yes")
    if pr.get("tipo") == "no_convencional" and fl == 1:
        kind = "shed"
    h = fl * 3.1 + 0.8 if kind != "shed" else 3.0
    add_bldg(p, "c", f"c{n}", h, "cat", fl, kind, split=False, band=carriage_band)
def covered_by_cadastre(p):
    if cat_tree is not None and any(cat_b[j][0].intersection(p).area > 0.25 * p.area for j in cat_tree.query(p)):
        return True
    return manz_tree is not None and any(manz[j].contains(p.centroid) for j in manz_tree.query(p.centroid))
for p, t, wid in osm_b:
    if covered_by_cadastre(p):
        dropped["osm_in_cadastre"] += 1
        continue
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
    if covered_by_cadastre(p):
        dropped["ms_in_cadastre"] += 1
        continue
    if osm_tree is not None:
        hit = [j for j in osm_tree.query(p) if osm_b[j][0].intersection(p).area > 0.3 * p.area]
        if hit:
            dropped["ms_dupe_osm"] += 1
            continue
    if on_road(p):
        dropped["ms_on_road"] += 1
        continue
    key = f"m{i}"
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
pois, trees, pts_extra = [], [], {}
for e in els:
    t = e.get("tags", {})
    if e["type"] == "node":
        x, y = P(e["lon"], e["lat"])
        if not AREA.contains(Point(x, y)):
            continue
        if t.get("natural") == "tree":
            trees.append([dm(x), dm(y)])
        elif t.get("highway") in ("bus_stop", "street_lamp", "give_way", "stop"):
            pts_extra.setdefault(t["highway"], []).append([dm(x), dm(y)])
        elif t.get("name") and (t.get("amenity") or t.get("shop") or t.get("historic") or t.get("tourism")):
            pois.append({"n": t["name"], "t": t.get("amenity") or t.get("shop") or t.get("historic") or t.get("tourism"), "x": dm(x), "y": dm(y)})

# ---------------------------------------------------------------- places (barrio names), waterways, railways
places, water, rail = [], [], []
WATER_W = {"river": 14, "canal": 5, "stream": 4, "ditch": 1.6, "drain": 1.6}
for e in els:
    t = e.get("tags", {})
    if e["type"] == "node" and t.get("place") in ("neighbourhood", "suburb", "quarter") and t.get("name"):
        x, y = P(e["lon"], e["lat"]); places.append({"n": t["name"], "x": dm(x), "y": dm(y)})
    if e["type"] != "way" or "geometry" not in e:
        continue
    pts = [P(g["lon"], g["lat"]) for g in e["geometry"]]
    if t.get("landuse") == "residential" and t.get("name") and pts[0] == pts[-1] and len(pts) > 3:
        try:
            poly = make_valid(Polygon(pts)).intersection(AREA)
        except Exception:
            continue
        for q in polys(poly):
            ro = ring_out(q, 1.0)
            if ro and q.area > 5000:
                places.append({"n": t["name"], **ro})
    if t.get("waterway") in WATER_W:
        line = LineString(pts).intersection(AREA)
        for part in (line.geoms if hasattr(line, "geoms") else [line]):
            if part.geom_type == "LineString" and part.length > 5:
                water.append({"n": t.get("name", ""), "t": t["waterway"], "w": float(t.get("width", WATER_W[t["waterway"]]) or WATER_W[t["waterway"]]) if str(t.get("width", "")).replace(".", "", 1).isdigit() else WATER_W[t["waterway"]], "p": flat(part.simplify(0.5).coords)})
    if t.get("railway") in ("rail", "disused", "abandoned"):
        line = LineString(pts).intersection(AREA)
        for part in (line.geoms if hasattr(line, "geoms") else [line]):
            if part.geom_type == "LineString" and part.length > 5:
                rail.append({"s": t["railway"][0], "p": flat(part.simplify(0.3).coords)})
print("places", len(places), "water", len(water), "rail", len(rail))

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
        inner = b.buffer(-0.75, join_style=2)
        for ip in polys(inner):
            if ip.exterior.length > 40:
                walks.append(flat(list(ip.simplify(0.5).exterior.coords)[:-1]))
print("walk rings", len(walks))

BKINDS = ['yes'] + sorted({b['k'] for b in bld_out} - {'yes'})
out = {
    "meta": {
        "name": "Palmira Centro", "units": "decimetres", "crs": LOC.to_proj4(),
        "origin": {"lat": LAT0, "lon": LON0, "label": "Parque Bolívar"},
        "osm_timestamp": osm["osm3s"]["timestamp_osm_base"],
        "msft_release": "Global ML Building Footprints 2026-02-03, quadkeys 032232031/032232033",
        "attribution": [
            "Map data © OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)",
            "Building footprints © Microsoft, Global ML Building Footprints, CDLA-Permissive-2.0",
            "Construcciones y número de pisos: Alcaldía de Palmira, Base Catastral 2024 (datos.gov.co); semaforización: Alcaldía de Palmira (datos.gov.co) — CC BY-SA 4.0 según publicación, licencia pendiente de confirmación",
        ],
        "area": list(CFG["area"]),
    },
    "roads": [{"n": r["name"], "c": r["hw"], "w": round(r["w"], 1), "we": r["w_est"], "o": r["o"], "id": r["id"]} for r in roads],
    "gnodes": gnodes, "edges": edges, "signals": signals,
    "blocks": blocks_out, "walks": walks,
    "ped": ped_out, "parks": parks, "fields": landuse, "landmarks": landmarks,
    "bkinds": BKINDS, "buildings": [[b["p"], b["hgt"], b["st"], b["s"], BKINDS.index(b["k"]) if b["k"] in BKINDS else 0] + ([b["id"]] if "id" in b else []) for b in bld_out],
    "signal_el": signal_el, "pois": pois, "trees": trees, "points": pts_extra, "places": places, "water": water, "rail": rail,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
