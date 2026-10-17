"""HealthPulse - Streamlit UI. Everything is processed in memory; nothing is stored."""
import pandas as pd
import streamlit as st

import healthpulse as hp

st.set_page_config(page_title="HealthPulse - your Apple Health trends", page_icon="💓", layout="wide")

st.title("💓 HealthPulse")
st.caption("Turn your Apple Health export into plain-English trends. Private by design: "
           "your file is processed in memory in this session and never stored. No AI services are called.")
st.warning("**Not medical advice.** HealthPulse only describes how your own numbers changed. "
           "It cannot diagnose anything or tell you what a change means. "
           "For any health concern, talk to a qualified professional.", icon="⚠️")

with st.expander("How to get your export (30 seconds)"):
    st.markdown(
        "1. iPhone **Health** app → your profile picture (top right) → **Export All Health Data**.\n"
        "2. Unzip the file and upload just **export.xml** (it can be huge; the zip works too).\n"
        "3. Streamlit Cloud's default upload limit is **200 MB**. If your export.xml is larger, "
        "run this app locally (see the README) - there is no size limit there.\n\n"
        "You can also upload a daily CSV with a `date` column and any of: "
        "`steps, resting_hr, sleep_hours, bed_min, wake_min, active_kcal, weight_kg`.")

use_demo = st.checkbox("Use demo data", help="Synthetic data with planted trends - not a real person.")
demo_xml = hp.generate_demo()
st.download_button("Download small demo export.xml", demo_xml, "demo_export.xml", "application/xml")
upload = None if use_demo else st.file_uploader("Upload export.xml, export.zip or a daily CSV", type=["xml", "zip", "csv"])


@st.cache_data(show_spinner="Reading your file (streaming)...", max_entries=2)
def load_upload(data: bytes, name: str) -> pd.DataFrame:
    import io
    return hp.load_any(io.BytesIO(data), name)


daily = None
try:
    if use_demo:
        daily = hp.load_any(demo_xml.encode(), "demo.xml")
    elif upload is not None:
        daily = load_upload(upload.getvalue(), upload.name)
except Exception as e:  # noqa: BLE001 - show a friendly message instead of a traceback
    st.error(f"Could not read that file: {e}")

if daily is None:
    st.info("Upload your export or tick **Use demo data** to see how it works.")
    st.stop()
if daily.dropna(how="all").empty:
    st.error("No supported records found (steps, resting heart rate, sleep, active energy, weight).")
    st.stop()

end = daily.index.max().date()
st.success(f"Loaded {daily.dropna(how='all').shape[0]} days of data, {daily.index.min().date()} → {end}. "
           f"'Last 30 days' means the 30 days ending {end}.")

st.header("What changed")
notes = hp.findings(daily)
if notes:
    for n in notes:
        st.markdown(f"- {n['text']}")
    st.caption("These are descriptive comparisons of your own data. Day-to-day variation is normal, "
               "and a change on its own says nothing about your health.")
else:
    st.write("No notable changes against your previous 30 days or longer-term average.")

table = hp.summary_table(daily)
if not table.empty:
    st.dataframe(table, hide_index=True)

st.header("Trends")
period = st.radio("Group by", ["Weekly", "Monthly"], horizontal=True)
freq = "W" if period == "Weekly" else "M"
charts = [("steps", "Daily steps (average)"), ("resting_hr", "Resting heart rate (bpm)"),
          ("sleep_hours", "Sleep duration (hours)"), ("active_kcal", "Active energy (kcal/day)"),
          ("weight_kg", "Body weight (kg)")]
cols = st.columns(2)
shown = 0
for key, title in charts:
    s = hp.aggregate(daily, key, freq)
    if len(s) < 2:
        continue
    with cols[shown % 2]:
        st.subheader(title)
        st.line_chart(s)
    shown += 1

bv = hp.bedtime_variability(daily).dropna()
if len(bv):
    with cols[shown % 2]:
        st.subheader("Bedtime variability (min, rolling 14 nights)")
        st.caption("Lower = more regular bedtimes.")
        st.line_chart(bv.resample("W" if freq == "W" else "MS").mean())

st.divider()
st.caption("HealthPulse is an informational tool, not a medical device. Not medical advice. "
           "Your data stays in your browser session and is discarded when you close the tab.")
