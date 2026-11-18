#!/usr/bin/env python3
"""Feasibility check: PGA scores per player and per round (ESPN) + weather (Open-Meteo).

Standard library only. Run:  python3 feasibility_check.py
Raw responses are saved to feasibility_out/ for inspection.
"""
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "feasibility_out"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (mlops-student-feasibility-check)"}
RESULTS = {}


def get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"   ! FAILED {url}\n     -> {e}")
        return None


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=1, ensure_ascii=False))


def find_keys(obj, needles, path="", hits=None, limit=12):
    """Recursively looks for keys whose name contains one of the given words."""
    hits = [] if hits is None else hits
    if len(hits) >= limit:
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if any(n in k.lower() for n in needles) and not isinstance(v, (dict, list)):
                hits.append((p, v))
            find_keys(v, needles, p, hits, limit)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2]):
            find_keys(v, needles, f"{path}[{i}]", hits, limit)
    return hits


def show(title, hits):
    print(f"   {title}:")
    if not hits:
        print("     (nothing found)")
    for p, v in hits:
        print(f"     {p} = {str(v)[:80]}")


def check_espn(date, label):
    print(f"\n=== ESPN: {label} (date {date}) ===")
    sb = get(f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={date}")
    if not sb:
        RESULTS[f"espn_{label}"] = False
        return
    save(f"espn_scoreboard_{date}.json", sb)
    events = sb.get("events", [])
    print(f"   {len(events)} event(s): {[e.get('name') for e in events]}")
    cal = (sb.get("leagues") or [{}])[0].get("calendar") or []
    print(f"   season calendar: {len(cal)} entries (e.g. {str(cal[:1])[:120]})")
    if not events:
        RESULTS[f"espn_{label}"] = False
        return
    ev = events[0]
    comps = (ev.get("competitions") or [{}])[0].get("competitors") or []
    print(f"   {len(comps)} players in the field")
    rounds_ok = False
    for c in comps[:3]:
        name = (c.get("athlete") or {}).get("displayName")
        ls = [(x.get("period"), x.get("value"), x.get("displayValue")) for x in c.get("linescores") or []]
        print(f"   - {name} | total={c.get('score')} | rounds (period, strokes, vs par) = {ls}")
        rounds_ok = rounds_ok or len(ls) >= 2
    RESULTS[f"espn_{label}"] = rounds_ok
    show("tee times?", find_keys(ev, ["tee", "starttime"]))
    show("course / venue?", find_keys(ev, ["venue", "course", "city", "state", "par", "yard", "latitude", "longitude"]))

    lb = get(f"https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?event={ev.get('id')}")
    if lb:
        save(f"espn_leaderboard_{ev.get('id')}.json", lb)
        show("leaderboard: tee times?", find_keys(lb, ["tee", "starttime"]))
        show("leaderboard: course?", find_keys(lb, ["course", "par", "yard", "latitude", "longitude"]))


def check_meteo():
    print("\n=== Open-Meteo: Augusta National, April 11, 2024 (Masters, round 1) ===")
    params = ("latitude=33.503&longitude=-82.020&start_date=2024-04-11&end_date=2024-04-11"
              "&hourly=wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m"
              "&timezone=America%2FNew_York")
    for label, base in [("observed (ERA5 archive)", "https://archive-api.open-meteo.com/v1/archive"),
                        ("archived forecast", "https://historical-forecast-api.open-meteo.com/v1/forecast")]:
        d = get(f"{base}?{params}")
        ok = bool(d and d.get("hourly", {}).get("time"))
        RESULTS[f"meteo_{label}"] = ok
        if ok:
            save(f"meteo_{label.split()[0]}.json", d)
            h = d["hourly"]
            print(f"   {label} - wind km/h from 7am to 7pm: {h['wind_speed_10m'][7:20]}")


if __name__ == "__main__":
    check_espn("20240411", "Masters 2024")
    check_espn("20160407", "Masters 2016 (historical depth)")
    check_espn("20230309", "Players 2023 (regular tournament)")
    check_meteo()
    print("\n=== SUMMARY ===")
    for k, v in RESULTS.items():
        print(f"   {'OK ' if v else 'NO '}  {k}")
    print(f"\nRaw files: {OUT}")
