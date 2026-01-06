import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def add_time_features(df):
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # 15-minute slots: 0..95
    df["slot"] = df["hour"] * 4 + (df["minute"] // 15)

    # Monday=0, Sunday=6
    df["weekday"] = df["timestamp"].dt.weekday

    return df

def compute_baseline(df, q=0.7):
    baseline = {}

    grouped = df.groupby(["weekday", "slot"])

    for (weekday, slot), group in grouped:
        baseline[(weekday, slot)] = group["power"].quantile(q)

    return baseline

def apply_baseline(df, baseline):
    df = df.copy()
    df["baseline_power"] = 0.0

    for i in range(len(df)):
        key = (df.at[i, "weekday"], df.at[i, "slot"])
        df.at[i, "baseline_power"] = baseline[key]

    return df

def compute_residual(df):
    df = df.copy()
    df["residual"] = df["power"] - df["baseline_power"]
    return df

def median_absolute_deviation(series):
    median = np.median(series)
    deviations = np.abs(series - median)
    return np.median(deviations)

def compute_threshold(df, k=3.0):
    r = df["residual"]
    r = r[r != 0]

    if len(r) == 0:
        return 0.0

    mad = median_absolute_deviation(r)
    return k * mad
"""
def compute_threshold(df, k=3.0):
    mad = median_absolute_deviation(df["residual"])
    threshold = k * mad
    return threshold
"""    
def detect_dr_flag(df, threshold):
    df = df.copy()
    df["dr_flag"] = 0

    for i in range(len(df)):
        r = df.at[i, "residual"]

        if r < -threshold:
            df.at[i, "dr_flag"] = -1
        elif r > threshold:
            df.at[i, "dr_flag"] = 1
        else:
            df.at[i, "dr_flag"] = 0

    return df

def compute_dr_capacity(df):
    df = df.copy()
    df["dr_capacity_kw"] = 0.0

    for i in range(len(df)):
        if df.at[i, "dr_flag"] != 0:
            df.at[i, "dr_capacity_kw"] = abs(df.at[i, "residual"])
        else:
            df.at[i, "dr_capacity_kw"] = 0.0

    return df

def group_events(df):
    events = []
    current_event = None

    for i in range(len(df)):
        flag = df.at[i, "dr_flag"]
        time = df.at[i, "timestamp"]

        if flag != 0 and current_event is None:
            current_event = {
                "start": time,
                "end": time,
                "flag": flag,
                "energy_kwh": abs(df.at[i, "residual"]) * 0.25
            }

        elif flag != 0 and current_event is not None:
            current_event["end"] = time
            current_event["energy_kwh"] += df.at[i, "dr_capacity_kw"] * 0.25

        elif flag == 0 and current_event is not None:
            events.append(current_event)
            current_event = None

    if current_event is not None:
        events.append(current_event)

    return events


def get_candidate_features(df):
    excluded = {
        "timestamp",
        "power",
        "baseline_power",
        "residual",
        "dr_flag",
        "dr_capacity_kw",
        "hour",
        "minute",
        "slot",
        "weekday"
    }

    candidates = []

    for col in df.columns:
        if col in excluded:
            continue

        dtype = df[col].dtype

        if dtype in ["float64", "int64", "bool"]:
            candidates.append(col)

    return candidates

def normalise_feature(series):
    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        return None  # useless feature

    return (series - mean) / std

def feature_correlation(residual, feature):
    if feature.isnull().any():
        return 0.0

    return np.corrcoef(residual, feature)[0, 1]

def select_useful_features(df, candidates, min_corr=0.2, max_features=3):
    useful = []

    for col in candidates:
        norm_feature = normalise_feature(df[col])
        if norm_feature is None:
            continue

        corr = feature_correlation(df["residual"], norm_feature)

        if abs(corr) >= min_corr:
            useful.append((col, corr))

    # keep strongest correlations
    useful.sort(key=lambda x: abs(x[1]), reverse=True)

    return useful[:max_features]

def correct_residual(df, selected_features):
    df = df.copy()

    for col, corr in selected_features:
        norm_feature = normalise_feature(df[col])

        # simple linear correction
        df["residual"] = df["residual"] - corr * norm_feature

    return df

def detect_dr(df):
    # Time features
    df = add_time_features(df)

    # Baseline
    baseline = compute_baseline(df, q=0.7)
    df = apply_baseline(df, baseline)

    # Residual
    df = compute_residual(df)

    # Extra feature handling
    candidates = get_candidate_features(df)
    selected = select_useful_features(df, candidates)
    df = correct_residual(df, selected)

    # DR detection
    threshold = compute_threshold(df)
    print(threshold)
    df = detect_dr_flag(df, threshold)
    df = compute_dr_capacity(df)
    print(df["residual"].describe())

    # Events
    events = group_events(df)

    return df, events, selected

def filter_short_events(events, min_steps=2):
    filtered = []

    for e in events:
        duration_steps = int(
            (e["end"] - e["start"]).total_seconds() / (15 * 60)
        ) + 1

        if duration_steps >= min_steps:
            filtered.append(e)

    return filtered

    
if __name__ == "__main__":

    df = pd.read_csv("training_data_a.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_out, events, selected_features = detect_dr(df)
    events = filter_short_events(events, min_steps=2)

    print("Selected features:", selected_features)
    print("Number of events:", len(events))

    # Print first 10 events
    for i, event in enumerate(events[:30]):
        print(f"Event {i+1}: {event}")

    # Plot full timeline
    plt.figure(figsize=(16, 6))

    plt.plot(df_out["timestamp"][:5000], df_out["power"][:5000], label="power", alpha=0.7)
    plt.plot(df_out["timestamp"][:5000], df_out["baseline_power"][:5000], label="baseline", linewidth=2)
    """
    # Highlight DR events
    for event in events:
        plt.axvspan(
            event["start"],
            event["end"],
            color="red" if event["flag"] == -1 else "green",
            alpha=0.15
        )"""

    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Demand Response Detection – Full Timeline")
    plt.legend()
    plt.tight_layout()
    plt.show()

