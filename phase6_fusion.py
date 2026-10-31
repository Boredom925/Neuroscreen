"""Phase 6: train and run the multimodal XGBoost fusion model.

This module deliberately does not infer an architecture from a PyTorch checkpoint.
It consumes the *measured outputs* from Phases 1-5.  This keeps the fusion model
reproducible and lets a child have any number of audio/video recordings: numeric
features from repeated recordings are aggregated into one row per child.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGETS = {
    "social_communication": "y_social_communication",
    "attention_regulation": "y_attention_regulation",
    "speech_language": "y_speech_language",
    "repetitive_behaviour": "y_repetitive_behaviour",
}

# Maxima from the Phase-5 M-CHAT-R axis weights.  They normalize its contribution
# to 0-1 before blending it with the corresponding XGBoost probability.
QUESTIONNAIRE_AXIS_MAX = {"sc": 12.0, "ar": 5.0, "sl": 3.0, "rb": 2.0}
QUESTIONNAIRE_FOR_TARGET = {
    "social_communication": "sc",
    "attention_regulation": "ar",
    "speech_language": "sl",
    "repetitive_behaviour": "rb",
}
ID_COLUMNS = {"child_id", "file", "processed_at", "mchatr_band", "mchatr_followup", "language"}


def _numeric_frame(frame: pd.DataFrame, prefix: str, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Return numeric columns, prefixed so phase column names cannot collide."""
    exclude = set(exclude) | ID_COLUMNS
    out = frame.drop(columns=[c for c in exclude if c in frame], errors="ignore").copy()
    out = out.select_dtypes(include=[np.number, "bool"])
    return out.rename(columns={c: f"{prefix}{c}" for c in out.columns})


def aggregate_records(frame: pd.DataFrame, child_id: str, prefix: str) -> pd.DataFrame:
    """Aggregate zero, one, or many recordings to a single feature row.

    For each numeric feature, mean, standard deviation (0 for one record), and
    maximum are retained.  The recording count is an explicit reliability cue.
    """
    if frame is None or frame.empty:
        return pd.DataFrame({"child_id": [child_id], f"{prefix}recording_count": [0]})
    numeric = _numeric_frame(frame, prefix)
    row = {"child_id": child_id, f"{prefix}recording_count": int(len(frame))}
    for column in numeric:
        values = pd.to_numeric(numeric[column], errors="coerce")
        row[f"{column}__mean"] = values.mean()
        row[f"{column}__std"] = values.std(ddof=0)
        row[f"{column}__max"] = values.max()
    return pd.DataFrame([row])


def load_phase_csv(path: str | Path | None) -> pd.DataFrame:
    """Load a phase CSV, returning an empty table when that modality is absent."""
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_fusion_row(
    child_id: str,
    *,
    phase1_csv: str | Path | None = None,
    phase2_csv: str | Path | None = None,
    phase3_csv: str | Path | None = None,
    audio_csv: str | Path | None = None,
    questionnaire_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Combine phase outputs into exactly one row for one child.

    Phase 1-3 CSVs may contain one row per video/frame/window. `audio_csv` may
    contain one row per .mp3/.wav/.mp4. Multiple media files are supported by
    aggregation, never by making multiple predictions for the same child.
    """
    parts = [pd.DataFrame({"child_id": [child_id]})]
    for prefix, source in (("p1_", phase1_csv), ("p2_", phase2_csv), ("p3_", phase3_csv), ("audio_", audio_csv)):
        data = load_phase_csv(source)
        if not data.empty and "child_id" in data:
            data = data[data["child_id"].astype(str) == str(child_id)]
        parts.append(aggregate_records(data, child_id, prefix))

    q = load_phase_csv(questionnaire_csv)
    if not q.empty:
        if "child_id" not in q:
            raise ValueError("questionnaire CSV must include child_id")
        q = q[q["child_id"].astype(str) == str(child_id)]
        if len(q) != 1:
            raise ValueError(f"Expected exactly one questionnaire row for {child_id!r}; found {len(q)}")
        q = pd.concat([q[["child_id"]].reset_index(drop=True), _numeric_frame(q, "q_").reset_index(drop=True)], axis=1)
        parts.append(q)

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="child_id", how="left", validate="one_to_one")
    return merged


def _feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {"child_id", *TARGETS.values()}
    return [c for c in data.select_dtypes(include=[np.number, "bool"]).columns if c not in excluded]


def _finite(value: object, default: float = 0.0) -> float:
    """Convert absent/NaN CSV values to a safe numeric default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def train_models(training_csv: str | Path, output_dir: str | Path, random_state: int = 42) -> dict:
    """Train four independently labelled, missing-value-tolerant XGBoost models."""
    data = pd.read_csv(training_csv)
    required = {"child_id", *TARGETS.values()}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Training CSV is missing: {sorted(missing)}")
    features = _feature_columns(data)
    if not features:
        raise ValueError("No numeric fusion features found")

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = {"n_children": int(len(data)), "features": features, "targets": {}}
    for risk_name, label in TARGETS.items():
        subset = data.dropna(subset=[label]).copy()
        y = subset[label].astype(int)
        if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
            raise ValueError(f"{label} must contain both binary classes (0 and 1)")
        x_train, x_test, y_train, y_test = train_test_split(
            subset[features], y, test_size=0.2, random_state=random_state, stratify=y
        )
        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=random_state, n_jobs=1,
        )
        model.fit(x_train, y_train)
        joblib.dump(model, out / f"{risk_name}_xgb.joblib")
        metrics["targets"][risk_name] = {
            "n_labelled": int(len(subset)),
            "held_out_accuracy": float(model.score(x_test, y_test)),
        }
    (out / "feature_schema.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict(fusion_row: pd.DataFrame, model_dir: str | Path, threshold: float = 0.50) -> pd.DataFrame:
    """Return four screened risk scores and an explicit M-CHAT-R referral flag."""
    if len(fusion_row) != 1:
        raise ValueError("predict expects exactly one child fusion row")
    model_dir = Path(model_dir)
    features = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    x = fusion_row.reindex(columns=features, fill_value=np.nan)
    output = {"child_id": fusion_row.iloc[0].get("child_id", "unknown")}
    for risk_name in TARGETS:
        model = joblib.load(model_dir / f"{risk_name}_xgb.joblib")
        model_probability = float(model.predict_proba(x)[:, 1][0])
        axis = QUESTIONNAIRE_FOR_TARGET[risk_name]
        q_value = _finite(fusion_row.iloc[0].get(f"q_q_axis_{axis}", 0.0))
        questionnaire_probability = float(np.clip(q_value / QUESTIONNAIRE_AXIS_MAX[axis], 0, 1))
        # The Phase-5 recommendation: clinically validated questionnaire carries 40%.
        final = 0.60 * model_probability + 0.40 * questionnaire_probability
        output[f"{risk_name}_risk"] = round(final, 4)
        output[f"{risk_name}_screen_positive"] = int(final >= threshold)
    total = _finite(fusion_row.iloc[0].get("q_mchatr_total", 0.0))
    override = int(bool(_finite(fusion_row.iloc[0].get("q_mchatr_refer_override", 0))) or total >= 8)
    output["mchatr_total"] = total
    output["mchatr_refer_override"] = override
    output["referral_recommended"] = int(override or any(v == 1 for k, v in output.items() if k.endswith("_screen_positive")))
    output["notice"] = "Screening decision-support only; not a clinical diagnosis. Review with a qualified clinician."
    return pd.DataFrame([output])
