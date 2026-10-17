# HealthPulse

Privacy-first Streamlit app that turns an **Apple Health export** into plain-English wellbeing trends:
steps, resting heart rate, sleep duration and bedtime consistency, active energy and weight - last 30 days
vs the previous 30 vs your own long-term baseline.

**Not medical advice.** HealthPulse is descriptive only: it never diagnoses, infers conditions or recommends treatment.

- Streams the XML (`xml.etree.iterparse`), keeping only a few record types, so very large exports do not exhaust memory.
- Processed in memory only; nothing is stored; no AI/API calls.
- Accepts `export.xml`, the `export.zip` from Apple, or a daily CSV (`date` + any of `steps, resting_hr, sleep_hours, bed_min, wake_min, active_kcal, weight_kg`).
- Includes a synthetic demo (`demo_export.xml`, regenerate with `healthpulse.generate_demo()`).

## Run locally (no upload size limit)
```
pip install -r requirements.txt
streamlit run app.py
pytest
```
Streamlit Community Cloud limits uploads to 200 MB: upload just `export.xml` from the zip, or run locally.

## Notes
Times use the local clock in the export (UTC offset ignored). "Last 30 days" ends at the last day in your data.
Multiple sources (iPhone + Watch) for steps/energy: the busiest single source per day is used.
