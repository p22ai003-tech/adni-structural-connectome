import streamlit as st, json, os, time

st.set_page_config(page_title="Connectome Monitor", page_icon="🧠", layout="wide")
STATUS = '/data/derivatives/qc/sc_matrix_qc/connectome_status.json'

try:
    s = json.load(open(STATUS))
except Exception as e:
    st.error(f"Status file not available yet: {e}")
    time.sleep(10)
    st.rerun()

o = s['overall']
st.title("🧠 Connectome Pipeline — Live Monitor")
st.caption(f"updated {s['ts']} UTC · auto-refresh 20s · source: canonical status file")

# ---- alerts ----
if s.get('alerts'):
    st.error("⚠️  ALERTS:  " + "   |   ".join(s['alerts']))
else:
    st.success("✓ All systems nominal — no alerts")

# ---- overall ----
pct = o['done'] / o['total'] if o['total'] else 0
st.subheader(f"CONNECTOMES COMPLETE:  {o['done']} / {o['total']}   ({pct*100:.0f}%)")
st.progress(pct)
c = st.columns(3)
c[0].metric("Done (≥0.6)", f"{o['done']} / {o['total']}")
c[1].metric("Mean density", o['mean_d'])
c[2].metric("Median density", o['median_d'])

# ---- per cohort ----
st.subheader("By cohort")
for g, v in s['cohort'].items():
    p = v['done'] / v['total'] if v['total'] else 0
    st.write(f"**{g}** — {v['done']}/{v['total']}  ({p*100:.0f}%)")
    st.progress(p)

# ---- S3 + regen ----
st.subheader("S3 self-contained backup & track regeneration")
c = st.columns(2)
sp = s.get('s3_pushed', 0); s3t = s.get('s3_total', 1)
c[0].write(f"**S3 backup** (.tck + sift + tsf + CSVs):  {sp}/{s3t}")
c[0].progress(min(sp / s3t, 1.0) if s3t else 0)
rg = s['runs'].get('regen216', {})
c[1].write(f"**Regen tracks rebuilt**:  {rg.get('done',0)}/216")
c[1].progress(min(rg.get('done', 0) / 216, 1.0))

# ---- compute lanes ----
st.subheader("Compute lanes")
for k, v in s['runs'].items():
    dot = "🟢" if v.get('alive') else "🔴"
    st.write(f"{dot}  **{k}** — done {v.get('done',0)}, workers {v.get('alive',0)}")

# ---- background services ----
st.subheader("Background services")
loops = s.get('loops', {})
cols = st.columns(max(len(loops), 1))
for i, (k, v) in enumerate(loops.items()):
    cols[i].metric(k, "UP" if v else "DOWN")

# ---- host ----
h = s['host']
st.subheader("Host")
c = st.columns(4)
c[0].metric("Load", f"{h['load_ratio']}×  /{h['ncpu']}")
c[1].metric("RAM free", f"{h['ram_free']}/{h['ram_total']} G")
c[2].metric("Disk free", f"{h['disk_free_g']} G")
c[3].metric("tckgen", h['tckgen'])

time.sleep(20)
st.rerun()
