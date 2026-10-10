from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DATA_FILE = Path("data/campaigns.csv")
OUTPUT_DIR = Path("outputs")
WINDOW = 28
MIN_HISTORY = 14
MAD_THRESHOLD = 3.5
COOLDOWN_DAYS = 3
MIN_CLICKS = 20
MIN_BASELINE_CONVERSIONS = 3
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    required = {"date", "campaign_id", "spend", "impressions", "clicks", "conversions", "is_incident"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.duplicated(["campaign_id", "date"]).any():
        raise ValueError("Duplicate campaign-date rows found.")
    for col in ["spend", "impressions", "clicks", "conversions"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")
    return df.sort_values(["campaign_id", "date"]).reset_index(drop=True)

def add_baselines(df):
    df = df.copy()
    df["conversion_rate"] = np.where(df.clicks > 0, df.conversions / df.clicks, np.nan)
    df["log_spend"] = np.log1p(df.spend)
    metrics = ["spend", "log_spend", "clicks", "conversions", "conversion_rate"]
    for metric in metrics:
        # Exclude today's value: only use data available before today.
        prior = df.groupby("campaign_id")[metric].shift(1)
        df[f"{metric}_baseline"] = prior.groupby(df["campaign_id"]).transform(
            lambda s: s.rolling(WINDOW, min_periods=MIN_HISTORY).median()
        )
        deviation = (prior - df[f"{metric}_baseline"]).abs()
        df[f"{metric}_mad"] = deviation.groupby(df["campaign_id"]).transform(
            lambda s: s.rolling(WINDOW, min_periods=MIN_HISTORY).median()
        )
        floor = 0.01 if metric == "log_spend" else (0.001 if metric == "conversion_rate" else 1.0)
        scale = (1.4826 * df[f"{metric}_mad"]).clip(lower=floor)
        df[f"{metric}_score"] = (df[metric] - df[f"{metric}_baseline"]) / scale
    return df

def generate_candidates(df):
    alerts = []
    for _, r in df.iterrows():
        c, d = r.campaign_id, r.date
        cb, sb, rb = r.conversions_baseline, r.spend_baseline, r.conversion_rate_baseline

        # Tracking failure: enough clicks and historically nontrivial conversions, now zero.
        if pd.notna(cb) and pd.notna(sb) and r.clicks >= MIN_CLICKS and cb >= MIN_BASELINE_CONVERSIONS and r.conversions == 0:
            alerts.append({
                "date": d, "campaign_id": c, "alert_type": "conversion_tracking",
                "severity": "critical", "metric": "conversions",
                "actual": float(r.conversions), "baseline": float(cb),
                "explanation": f"{c}: conversions fell to 0 from a prior 28-day median of {cb:.1f}, while clicks were {int(r.clicks)}. Tracking or the conversion funnel may be broken.",
                "diagnostic_step": "Verify conversion tag firing, event delivery, attribution, and the checkout funnel."
            })

        # Spend deviation uses log-space robust score to reduce scale effects.
        ss = r.log_spend_score
        if pd.notna(ss) and abs(ss) >= MAD_THRESHOLD:
            direction = "increased" if r.spend > sb else "decreased"
            alerts.append({
                "date": d, "campaign_id": c, "alert_type": "spend_anomaly",
                "severity": "high" if abs(ss) >= 5 else "medium", "metric": "spend",
                "actual": float(r.spend), "baseline": float(sb),
                "explanation": f"{c}: spend {direction} to {r.spend:.2f}, versus prior 28-day median {sb:.2f}; robust log-spend score {ss:.2f}.",
                "diagnostic_step": "Check budget caps, bid strategy, pacing, campaign status, and delivery."
            })

        # Delivery interruption: spend and delivery metrics all collapse.
        if pd.notna(sb) and sb > 0 and r.spend == 0 and r.impressions == 0 and r.clicks == 0:
            alerts.append({
                "date": d, "campaign_id": c, "alert_type": "delivery_interruption",
                "severity": "critical", "metric": "spend",
                "actual": 0.0, "baseline": float(sb),
                "explanation": f"{c}: spend, impressions, and clicks are all zero, versus prior median spend {sb:.2f}. This suggests a delivery interruption; confirm campaign status and budget.",
                "diagnostic_step": "Check whether the campaign is active, budget is exhausted, ads are disapproved, or delivery is paused."
            })

        # Conversion-rate drop: guard against low traffic and zero conversions already caught above.
        rs = r.conversion_rate_score
        if pd.notna(rs) and pd.notna(rb) and r.clicks >= MIN_CLICKS and rb > 0 and rs <= -MAD_THRESHOLD and r.conversions > 0:
            alerts.append({
                "date": d, "campaign_id": c, "alert_type": "conversion_rate_drop",
                "severity": "high", "metric": "conversion_rate",
                "actual": float(r.conversion_rate), "baseline": float(rb),
                "explanation": f"{c}: conversion rate fell to {r.conversion_rate:.2%} from prior median {rb:.2%}.",
                "diagnostic_step": "Inspect landing page, checkout funnel, traffic quality, and tracking."
            })
    return pd.DataFrame(alerts)

def suppress_duplicates(candidates):
    if candidates.empty:
        return candidates.copy()
    candidates = candidates.sort_values(["campaign_id", "alert_type", "date"])
    kept, last_sent = [], {}
    for idx, r in candidates.iterrows():
        key = (r.campaign_id, r.alert_type)
        prior = last_sent.get(key)
        if prior is None or (r.date - prior).days > COOLDOWN_DAYS:
            kept.append(idx)
            last_sent[key] = r.date
    return candidates.loc[kept].sort_values(["date", "campaign_id"]).reset_index(drop=True)

def evaluate_rows(df, alerts):
    # Row-level evaluation for a first baseline. Incident-level scoring should also
    # be reported because a multi-day incident is one incident, not many.
    truth = df["is_incident"].astype(int).to_numpy()
    alert_days = set(zip(alerts.campaign_id, pd.to_datetime(alerts.date))) if not alerts.empty else set()
    predicted = np.array([(c, d) in alert_days for c, d in zip(df.campaign_id, df.date)], dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "true_positives_rows": int(tp), "false_positives_rows": int(fp),
        "false_negatives_rows": int(fn), "true_negatives_rows": int(tn),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "note": "Row-level scores are preliminary; also evaluate each seeded incident as a single event."
    }

def make_chart(df, alerts, campaign_id):
    sub = df[df.campaign_id == campaign_id]
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub.date, sub.spend, label="Actual spend")
    ax.plot(sub.date, sub.spend_baseline, label="Prior rolling median")
    if not alerts.empty:
        a = alerts[(alerts.campaign_id == campaign_id) & (alerts.alert_type == "spend_anomaly")]
        if not a.empty:
            ax.scatter(a.date, a.actual, label="Spend alerts")
    ax.set_title(f"{campaign_id}: daily spend and baseline")
    ax.set_xlabel("Date"); ax.set_ylabel("Spend"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"{campaign_id}_spend.png"); plt.close(fig)

def main():
    df = add_baselines(load_data())
    candidates = generate_candidates(df)
    alerts = suppress_duplicates(candidates)
    df.to_csv(OUTPUT_DIR / "scored_metrics.csv", index=False)
    alerts.to_csv(OUTPUT_DIR / "alerts.csv", index=False)
    report = {
        "data_type": "synthetic demonstration data",
        "rows_processed": int(len(df)),
        "campaigns_processed": int(df.campaign_id.nunique()),
        "date_start": str(df.date.min().date()),
        "date_end": str(df.date.max().date()),
        "candidate_alerts_before_suppression": int(len(candidates)),
        "total_alerts_after_suppression": int(len(alerts)),
        "alerts_by_type": alerts.alert_type.value_counts().to_dict() if not alerts.empty else {},
        "alerts_by_severity": alerts.severity.value_counts().to_dict() if not alerts.empty else {},
        "row_level_evaluation": evaluate_rows(df, alerts),
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    make_chart(df, alerts, "C003")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
          
