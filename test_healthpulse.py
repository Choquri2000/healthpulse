import io
import zipfile
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import healthpulse as hp

DEMO = hp.generate_demo()


def daily():
    return hp.load_any(DEMO.encode(), "x.xml")


def test_parser_on_demo_xml():
    parsed = hp.parse_health_xml(DEMO.encode())
    assert len(parsed["records"]["steps"]) == 180 * 3
    assert len(parsed["records"]["resting_hr"]) == 180
    assert len(parsed["workouts"]) == 45
    d = hp.build_daily(parsed)
    assert len(d) == 180 and d["steps"].notna().all()
    assert 6.5 < d["sleep_hours"].mean() < 8.5


def test_zip_and_saved_demo_file():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("apple_health_export/export.xml", DEMO)
        z.writestr("apple_health_export/export_cda.xml", "<x/>")
    assert len(hp.load_any(buf, "export.zip")) == 180
    saved = Path(__file__).with_name("demo_export.xml")
    assert saved.exists() and len(hp.load_any(saved.read_bytes(), "demo_export.xml")) == 180


def test_overlapping_sources_not_double_counted():
    xml = ('<HealthData>'
           '<Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" startDate="2025-01-01 10:00:00 +0000" endDate="2025-01-01 11:00:00 +0000" value="1000"/>'
           '<Record type="HKQuantityTypeIdentifierStepCount" sourceName="Watch" startDate="2025-01-01 10:00:00 +0000" endDate="2025-01-01 11:00:00 +0000" value="900"/>'
           '</HealthData>')
    assert hp.load_any(xml.encode(), "a.xml")["steps"].iloc[0] == 1000


def test_planted_trends_detected():
    d = daily()
    kinds = {f["metric"]: f["text"] for f in hp.findings(d)}
    c = hp.compare(d, "resting_hr")
    assert 3 < c["current"] - c["previous"] < 7
    assert "above" in kinds["resting_hr"]
    assert "below" in kinds["steps"]
    assert "bed_var" in kinds  # bedtimes became less regular
    assert "weight_kg" not in kinds or "below" in kinds["weight_kg"]


def test_flat_data_no_findings():
    idx = pd.date_range("2025-01-01", periods=200)
    d = pd.DataFrame({m: 1.0 for m in hp.METRICS}, index=idx)
    d["steps"] = 8000.0
    d["resting_hr"] = 60.0
    assert hp.findings(d) == []


def test_no_medical_language():
    text = " ".join(f["text"].lower() for f in hp.findings(daily()))
    for bad in ("diagnos", "disease", "condition", "you should", "see a doctor", "healthy", "unhealthy"):
        assert bad not in text


def test_csv():
    csv = "date,steps,resting_hr\n" + "\n".join(f"2025-01-{i:02d},{5000+i},60" for i in range(1, 29))
    d = hp.load_any(io.StringIO(csv), "x.csv")
    assert len(d) == 28 and d["steps"].iloc[0] == 5001


def test_app_end_to_end_demo():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    assert "Not medical advice" in at.warning[0].value
    at.checkbox[0].check().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "resting heart rate" in md.lower()
    assert len(at.subheader) >= 4
