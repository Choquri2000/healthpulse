"""HealthPulse engine: stream-parse an Apple Health export, build daily series,
compute descriptive trends. Descriptive only - never diagnoses or advises."""
from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
# Uploads are untrusted: defusedxml refuses entity declarations (billion laughs) and external entities (XXE).
from defusedxml.ElementTree import iterparse

TYPES = {
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_kcal",
    "HKQuantityTypeIdentifierBodyMass": "weight_kg",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
}
ASLEEP_SUFFIXES = ("Asleep", "AsleepCore", "AsleepDeep", "AsleepREM", "AsleepUnspecified")
METRICS = ["steps", "resting_hr", "sleep_hours", "bed_min", "wake_min", "active_kcal", "weight_kg"]

# label, unit, absolute change worth mentioning, relative change worth mentioning, number format
INFO = {
    "steps": ("Daily steps", "steps", 0, 0.10, "{:.0f}"),
    "resting_hr": ("Resting heart rate", "bpm", 3, 0, "{:.1f}"),
    "sleep_hours": ("Sleep duration", "h", 0.5, 0, "{:.1f}"),
    "bed_min": ("Bedtime", "min", 30, 0, "{:.0f}"),
    "wake_min": ("Wake time", "min", 30, 0, "{:.0f}"),
    "bed_var": ("Bedtime variability", "min", 20, 0, "{:.0f}"),
    "active_kcal": ("Active energy", "kcal", 0, 0.10, "{:.0f}"),
    "weight_kg": ("Body weight", "kg", 1.0, 0, "{:.1f}"),
}


# ----------------------------------------------------------------- parsing
def _ts(s: str) -> datetime:
    """'2024-01-05 08:30:00 +0100' -> naive local datetime (offset ignored)."""
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def _open_xml(source):
    """Return a binary file-like for the XML inside a path/bytes/file/zip."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, str):
        source = open(source, "rb")
    source.seek(0)
    head = source.read(4)
    source.seek(0)
    if head[:2] == b"PK":
        zf = zipfile.ZipFile(source)
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        pick = [n for n in names if n.split("/")[-1].lower() == "export.xml"]
        if not pick:
            raise ValueError("No export.xml found inside the zip.")
        return zf.open(pick[0])
    return source


def parse_health_xml(source) -> dict:
    """Stream the export; keep only the record types we need.
    Returns {'records': {metric: DataFrame}, 'workouts': DataFrame}."""
    f = _open_xml(source)
    raw = {k: [] for k in TYPES.values()}
    workouts = []
    root = None
    n = 0
    for ev, el in iterparse(f, events=("start", "end")):
        if ev == "start":
            if root is None:
                root = el
            continue
        tag = el.tag
        if tag == "Record":
            m = TYPES.get(el.get("type"))
            if m:
                try:
                    start, end = _ts(el.get("startDate")), _ts(el.get("endDate"))
                    if m == "sleep":
                        raw[m].append((start, end, el.get("value", "")))
                    else:
                        v = float(el.get("value"))
                        if m == "weight_kg" and el.get("unit") == "lb":
                            v *= 0.45359237
                        raw[m].append((start, end, v, el.get("sourceName", "")))
                except (TypeError, ValueError):
                    pass
        elif tag == "Workout":
            try:
                dur = float(el.get("duration", 0))
                unit = el.get("durationUnit", "min")
                dur = dur / 60 if unit == "s" else dur * 60 if unit == "hr" else dur
                workouts.append((_ts(el.get("startDate")),
                                 el.get("workoutActivityType", "").replace("HKWorkoutActivityType", ""), dur))
            except (TypeError, ValueError):
                pass
        if tag in ("Record", "Workout", "Correlation", "ActivitySummary"):
            el.clear()
            n += 1
            if root is not None and n % 5000 == 0:
                root.clear()  # drop already-processed children so memory stays flat
    cols = ["start", "end", "value", "source"]
    records = {k: pd.DataFrame(v, columns=cols) for k, v in raw.items() if k != "sleep"}
    records["sleep"] = pd.DataFrame(raw["sleep"], columns=["start", "end", "value"])
    wk = pd.DataFrame(workouts, columns=["start", "type", "minutes"])
    return {"records": records, "workouts": wk}


def _sleep_nights(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster asleep segments (gap > 2h splits sessions); keep the longest
    session per wake date. Returns index=date with sleep_hours, bed_min, wake_min."""
    empty = pd.DataFrame(columns=["sleep_hours", "bed_min", "wake_min"])
    if df.empty:
        return empty
    d = df[df["value"].str.endswith(ASLEEP_SUFFIXES)].sort_values("start")
    if d.empty:  # older exports: only InBed
        d = df[df["value"].str.endswith("InBed")].sort_values("start")
    sessions, cur = [], None
    for s, e, _ in d.itertuples(index=False):
        if cur and s - cur["end"] <= timedelta(hours=2):
            cur["asleep"] += max(timedelta(0), e - max(s, cur["end"]))  # overlapping sources
            cur["end"] = max(cur["end"], e)
        else:
            if cur:
                sessions.append(cur)
            cur = {"start": s, "end": e, "asleep": e - s}
    if cur:
        sessions.append(cur)
    best = {}
    for ss in sessions:
        day = ss["end"].date()
        if day not in best or ss["asleep"] > best[day]["asleep"]:
            best[day] = ss
    rows = {}
    for day, ss in best.items():
        h = ss["asleep"].total_seconds() / 3600
        if h < 1:  # naps / fragments are not a night
            continue
        bed = ss["start"].hour * 60 + ss["start"].minute
        bed = (bed - 18 * 60) % 1440 + 18 * 60  # 22:00 -> 1320, 01:00 -> 1500 (no midnight wrap)
        rows[day] = (h, bed, ss["end"].hour * 60 + ss["end"].minute)
    if not rows:
        return empty
    return pd.DataFrame.from_dict(rows, orient="index", columns=["sleep_hours", "bed_min", "wake_min"]).sort_index()


def build_daily(parsed: dict) -> pd.DataFrame:
    """One row per calendar day with the metrics we track."""
    rec = parsed["records"]
    parts = []
    for m, how in (("steps", "sum"), ("active_kcal", "sum"), ("resting_hr", "mean"), ("weight_kg", "mean")):
        df = rec.get(m)
        if df is None or df.empty:
            continue
        df = df.assign(day=df["start"].dt.date)
        if how == "sum":
            # iPhone + Watch both record steps: take the single busiest source per day
            per = df.groupby(["day", "source"])["value"].sum().groupby("day").max()
        else:
            per = df.groupby("day")["value"].mean()
        parts.append(per.rename(m))
    sleep = _sleep_nights(rec.get("sleep", pd.DataFrame()))
    if len(sleep):
        parts.extend([sleep[c] for c in sleep.columns])
    if not parts:
        return pd.DataFrame(columns=METRICS)
    daily = pd.concat(parts, axis=1).sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    for m in METRICS:
        if m not in daily:
            daily[m] = np.nan
    return daily[METRICS]


def load_csv(source) -> pd.DataFrame:
    """Daily CSV: a 'date' column plus any of the METRICS names (case-insensitive)."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df:
        raise ValueError("CSV needs a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    use = [m for m in METRICS if m in df]
    if not use:
        raise ValueError("CSV has no recognised columns: " + ", ".join(METRICS))
    out = df[use].apply(pd.to_numeric, errors="coerce")
    out = out.groupby(out.index.normalize()).mean()
    out = out.reindex(pd.date_range(out.index.min(), out.index.max()))
    for m in METRICS:
        if m not in out:
            out[m] = np.nan
    return out[METRICS]


def load_any(source, name: str = "") -> pd.DataFrame:
    """Dispatch on file name. Returns the daily DataFrame."""
    if name.lower().endswith(".csv"):
        return load_csv(source)
    return build_daily(parse_health_xml(source))


# ----------------------------------------------------------------- trends
def aggregate(daily: pd.DataFrame, metric: str, freq: str = "W") -> pd.Series:
    """Weekly ('W') or monthly ('M') mean of a daily metric."""
    freq = {"M": "MS"}.get(freq, freq)
    return daily[metric].resample(freq).mean().dropna()


def bedtime_variability(daily: pd.DataFrame, window: int = 14) -> pd.Series:
    """Rolling std (minutes) of bedtime - a simple sleep-time consistency measure."""
    return daily["bed_min"].rolling(window, min_periods=max(5, window // 2)).std()


def _mean(s: pd.Series, min_n: int):
    s = s.dropna()
    return float(s.mean()) if len(s) >= min_n else None


def compare(daily: pd.DataFrame, metric: str, window: int = 30) -> dict | None:
    """Last `window` days vs the previous `window` days and vs the earlier
    long-term baseline (up to 365 days before the last window)."""
    if daily.empty:
        return None
    series = bedtime_variability(daily, 14) if metric == "bed_var" else daily[metric]
    end = daily.index.max()
    last = series[end - timedelta(days=window - 1): end]
    prev = series[end - timedelta(days=2 * window - 1): end - timedelta(days=window)]
    base = series[end - timedelta(days=window + 364): end - timedelta(days=window)]
    min_n = max(5, window // 4)
    cur = _mean(last, min_n)
    if cur is None:
        return None
    return {"metric": metric, "current": cur, "previous": _mean(prev, min_n),
            "baseline": _mean(base, max(10, window // 2)), "window": window, "end": end.date()}


def _describe(m: str, c: dict) -> list[str]:
    label, unit, t_abs, t_pct, fmt = INFO[m]
    msgs = []
    for key, ref in (("previous", f"the previous {c['window']} days"), ("baseline", "your longer-term average")):
        other = c[key]
        if other is None:
            continue
        diff = c["current"] - other
        notable = bool(t_abs) and abs(diff) >= t_abs
        if t_pct and other:
            notable = notable or abs(diff) / abs(other) >= t_pct
        if not notable:
            continue
        if m in ("bed_min", "wake_min"):
            when = "later" if diff > 0 else "earlier"
            msgs.append(f"Your average {label.lower()} over the last {c['window']} days is "
                        f"{abs(diff):.0f} minutes {when} than {ref}.")
            continue
        direction = "above" if diff > 0 else "below"
        if t_pct:
            amt = f"{abs(diff) / abs(other) * 100:.0f}% ({fmt.format(abs(diff))} {unit})"
        else:
            amt = f"{fmt.format(abs(diff))} {unit}"
        msgs.append(f"Your {label.lower()} over the last {c['window']} days "
                    f"({fmt.format(c['current'])} {unit}) is {amt} {direction} {ref}.")
    return msgs


def findings(daily: pd.DataFrame, window: int = 30) -> list[dict]:
    """Plain-English, descriptive-only notes about notable changes."""
    out = []
    for m in ["steps", "resting_hr", "sleep_hours", "bed_min", "wake_min", "bed_var", "active_kcal", "weight_kg"]:
        c = compare(daily, m, window)
        if c:
            out += [{"metric": m, "text": t} for t in _describe(m, c)]
    return out


def summary_table(daily: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    rows = []
    for m in ["steps", "resting_hr", "sleep_hours", "bed_var", "active_kcal", "weight_kg"]:
        c = compare(daily, m, window)
        if c:
            rows.append({"Metric": f"{INFO[m][0]} ({INFO[m][1]})", f"Last {window} days": c["current"],
                         f"Previous {window} days": c["previous"], "Long-term baseline": c["baseline"]})
    return pd.DataFrame(rows).round(1)


# ----------------------------------------------------------------- demo data
def generate_demo(end: date = date(2025, 6, 30), days: int = 180, seed: int = 7) -> str:
    """Synthetic Apple-Health-style export.xml with planted trends in the last 30 days:
    resting HR up ~5 bpm, steps down ~25%, bedtimes later and less regular, weight slowly down.
    Entirely made up - no real person's data."""
    rng = np.random.default_rng(seed)
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<HealthData locale="en_US">']
    fmt = "%Y-%m-%d %H:%M:%S +0000"

    def rec(typ, s, e, v, unit, src="Demo Watch"):
        out.append(f'<Record type="{typ}" sourceName="{src}" unit="{unit}" '
                   f'startDate="{s.strftime(fmt)}" endDate="{e.strftime(fmt)}" value="{v}"/>')

    start_day = end - timedelta(days=days - 1)
    for i in range(days):
        d = start_day + timedelta(days=i)
        base = datetime(d.year, d.month, d.day)
        recent = i >= days - 30
        ramp = (i - (days - 30)) / 30 if recent else 0.0
        steps = rng.normal(8200, 900) * (1 - 0.35 * ramp)
        for part, frac in enumerate((0.3, 0.5, 0.2)):
            s = base + timedelta(hours=8 + part * 5)
            rec("HKQuantityTypeIdentifierStepCount", s, s + timedelta(hours=1), int(steps * frac), "count")
        rhr = rng.normal(60, 1.2) + 5 * min(1.0, ramp * 3)
        rec("HKQuantityTypeIdentifierRestingHeartRate", base + timedelta(hours=12),
            base + timedelta(hours=12), round(rhr, 1), "count/min")
        kcal = rng.normal(480, 60) * (1 - 0.2 * ramp)
        rec("HKQuantityTypeIdentifierActiveEnergyBurned", base + timedelta(hours=18),
            base + timedelta(hours=19), round(kcal, 1), "kcal")
        bed_h = 23.0 + rng.normal(0, 0.25) + (0.9 * ramp + rng.normal(0, 0.6) if recent else 0)
        bed = base - timedelta(days=1) + timedelta(hours=bed_h)
        wake = bed + timedelta(hours=rng.normal(7.6, 0.4) - 0.6 * ramp)
        mid = bed + (wake - bed) / 2
        rec("HKCategoryTypeIdentifierSleepAnalysis", bed, mid, "HKCategoryValueSleepAnalysisAsleepCore", "")
        rec("HKCategoryTypeIdentifierSleepAnalysis", mid, wake, "HKCategoryValueSleepAnalysisAsleepDeep", "")
        if i % 3 == 0:
            w = 82.0 - 0.012 * i + rng.normal(0, 0.2)
            rec("HKQuantityTypeIdentifierBodyMass", base + timedelta(hours=7),
                base + timedelta(hours=7), round(w, 1), "kg", "Demo Scale")
        if i % 4 == 0:
            s = base + timedelta(hours=17)
            out.append(f'<Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="{int(rng.normal(35, 5))}" '
                       f'durationUnit="min" sourceName="Demo Watch" startDate="{s.strftime(fmt)}" '
                       f'endDate="{(s + timedelta(minutes=35)).strftime(fmt)}"/>')
    out.append("</HealthData>")
    return "\n".join(out)
