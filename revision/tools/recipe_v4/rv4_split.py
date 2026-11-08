"""Mid-run split of the recipe v4 cohort across the main server and its copies (FREEZE.md, 5 Nov). Run on the main server AFTER its driver and xargs are
stopped, so nothing new starts while the lists are made. A session counts as started if it has a QC file, a work folder, or a running worker; started
sessions stay with the main server. The rest, in the driver's own order (acquisition strata in turn), go to the machines in proportion to their
remaining load: the main server's running sessions count as half a session each (on average half done). Prints counts only, no identifiers.
Usage: python -I rv4_split.py <driver jobs file> <out dir> <copies>"""
import os, subprocess, sys

B = "/data/derivatives/recipe_v4/cohort"
jobs, out, ncopy = sys.argv[1], sys.argv[2], int(sys.argv[3])
order = [u.strip() for u in open(jobs) if u.strip()]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
running = {a.split()[-1] for a in ps if "rv4_cohort.sh worker " in a and not a.startswith("grep")}
started = {u for u in order if os.path.exists(f"{B}/qc/C_{u}.json") or os.path.isdir(f"{B}/work/C/{u}")} | running
todo = [u for u in order if u not in started]
W = len(todo) + len(running) / 2
names = ["main"] + [f"clone{k}" for k in range(1, ncopy + 1)]
target = {n: W / len(names) for n in names}
target["main"] = max(target["main"] - len(running) / 2 + 2, 1)      # +2: the copies start about 15 min later (tools warming)
lists = {n: [] for n in names}
for u in todo:
    n = min(names, key=lambda n: ((len(lists[n]) + 1) / target[n], names.index(n)))
    lists[n].append(u)
os.makedirs(out, exist_ok=True)
for n, us in lists.items():
    with open(f"{out}/{n}.txt", "w") as f:
        f.write("".join(f"{u}\n" for u in us))
print(f"in order {len(order)}; started {len(started)} (running now {len(running)}); not started {len(todo)} -> " + ", ".join(f"{n} {len(us)}" for n, us in lists.items()))
assert sum(len(v) for v in lists.values()) == len(todo) and len(set().union(*map(set, lists.values()))) == len(todo)
