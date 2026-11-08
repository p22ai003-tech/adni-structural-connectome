"""Second mid-run split (FREEZE.md, 5 Nov evening): after every machine's hand-out is stopped, give the sessions not yet started to all machines
(the main server, the copies already running, and new copies) so that all finish together. Each machine's running sessions keep their slots until
their expected end (session time per acquisition group, measured on the main server at full load, minus time already run); the waiting sessions
go longest first to whichever slot frees first. Prints counts only, no identifiers.
Usage: python -I rv4_resplit.py <state dir> <out dir> <machine>=<slots>@<minutes from now>[*<slowdown>] ...
  <state dir>/<machine>.txt holds lines "run <seconds elapsed> <unit>" and "wait 0 <unit>" (a new copy has no file).
Writes <out dir>/<machine>.txt (its waiting sessions, in start order) and <out dir>/pull_<machine>.txt (running + waiting, what to copy back)."""
import glob, heapq, json, os, statistics as st, sys, time
import pandas as pd

R4 = "/data/derivatives/recipe_v4"; B = f"{R4}/cohort"
state, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
units = pd.read_csv(f"{R4}/lists/cohort_units.csv", dtype=str).set_index("unit")
grp = lambda u: f"{units.at[u, 'phase']}/{units.at[u, 'vendor']}"
since = time.mktime(time.strptime("2026-11-04 07:00", "%Y-%m-%d %H:%M"))      # 12:30 IST 4 Nov: main server at full load
dur = {}
for f in glob.glob(f"{B}/qc/C_noact_*.json"):
    n = json.load(open(f))
    if n["status"] != "ok" or os.path.getmtime(f) < since: continue
    a = json.load(open(f"{B}/qc/C_{n['unit']}.json")); dur.setdefault(grp(n["unit"]), []).append((a["seconds_total_this_attempt"] + n["seconds_total_this_attempt"]) / 3600)
m_all = st.mean([x for v in dur.values() for x in v])
hours = lambda u: st.mean(dur[grp(u)]) if len(dur.get(grp(u), [])) >= 3 else m_all
heap, running, waiting, slow = [], {}, [], {}
for spec in specs:
    name, rest = spec.split("="); slots, rest = rest.split("@"); delay, _, f = rest.partition("*")
    slots, delay, slow[name] = int(slots), float(delay) / 60, float(f or 1)
    running[name] = []; p = f"{state}/{name}.txt"
    if os.path.exists(p):
        for line in open(p):
            tag, el, u = line.split()
            if tag == "run": running[name].append(u); heapq.heappush(heap, (max(hours(u) * slow[name] - int(el) / 3600, 0.2), name))
            else: waiting.append(u)
    for _ in range(slots - len(running[name])): heapq.heappush(heap, (delay, name))
assert len(waiting) == len(set(waiting)), "a session is waiting on two machines"
assign = {n: [] for n in running}; end = {n: 0 for n in running}
for t, n in heap: end[n] = max(end[n], t)
for u in sorted(waiting, key=hours, reverse=True):
    t, n = heapq.heappop(heap); e = t + hours(u) * slow[n]; heapq.heappush(heap, (e, n)); assign[n].append(u); end[n] = max(end[n], e)
os.makedirs(out, exist_ok=True)
for n in assign:
    open(f"{out}/{n}.txt", "w").write("".join(f"{u}\n" for u in assign[n]))
    open(f"{out}/pull_{n}.txt", "w").write("".join(f"{u}\n" for u in running[n] + assign[n]))
print(f"waiting {len(waiting)} -> " + ", ".join(f"{n} {len(assign[n])} (running {len(running[n])}, done in ~{end[n]:.1f} h)" for n in assign))
assert sum(map(len, assign.values())) == len(waiting)
