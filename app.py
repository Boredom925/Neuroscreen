"""NeuroScreen college-project dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from feature_extractors import aggregate_child, audio_features, load_models, prototype_domain_scores, video_features

ROOT = Path(__file__).parent
st.set_page_config(page_title="NeuroScreen", page_icon="🧠", layout="wide")
st.title("NeuroScreen - Phase 1-5 Feature Dashboard")
st.warning("College-project prototype only. These outputs are not a diagnosis and must not guide clinical decisions.")

@st.cache_resource(show_spinner="Loading Phase 1-3 models...")
def get_models():
    return load_models(ROOT)

mode = st.radio("How are these recordings grouped?", ["One child", "Multiple children"], horizontal=True)
uploads = st.file_uploader("Upload videos/audio", type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"], accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more files to begin. Videos run Phases 1-3 and contain audio for Phase 4; audio files run Phase 4.")
    st.stop()

assignments = {}
if mode == "One child":
    child_id = st.text_input("Child ID (use a de-identified ID)", "child_001")
    assignments = {item.name: child_id for item in uploads}
else:
    st.caption("Assign each uploaded file to a de-identified child ID.")
    columns = st.columns(2)
    for index, item in enumerate(uploads):
        assignments[item.name] = columns[index % 2].text_input(item.name, value=item.name.split("_")[0], key=f"child_{index}")

st.subheader("Questionnaire")
use_questionnaire = st.checkbox("Include questionnaire scores", value=False)
st.caption("Optional: enter the normalized domain scores (0 to 1) calculated by your Phase 5 notebook. The questionnaire has a 60% weight in the prototype fusion. If all values are 0, all four prototype indicators are 0.")
questionnaire_rows = []
for child in sorted(set(assignments.values())):
    cols = st.columns(4)
    questionnaire_rows.append({"child_id": child,
        "q_axis_sc": cols[0].number_input(f"{child}: social", 0., 1., 0. if use_questionnaire else 0., .05, disabled=not use_questionnaire),
        "q_axis_ar": cols[1].number_input(f"{child}: attention", 0., 1., 0. if use_questionnaire else 0., .05, disabled=not use_questionnaire),
        "q_axis_sl": cols[2].number_input(f"{child}: speech", 0., 1., 0. if use_questionnaire else 0., .05, disabled=not use_questionnaire),
        "q_axis_rb": cols[3].number_input(f"{child}: repetitive", 0., 1., 0. if use_questionnaire else 0., .05, disabled=not use_questionnaire),
    })

if st.button("Extract Phase 1-5 features and create dashboard", type="primary"):
    try:
        models = get_models(); file_rows = defaultdict(list)
        progress = st.progress(0, text="Preparing files")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index, item in enumerate(uploads):
                path = directory / item.name; path.write_bytes(item.getvalue())
                suffix = path.suffix.lower(); row = {"file": item.name}
                if suffix in {".mp4", ".mov", ".mkv", ".avi"}:
                    row.update(video_features(path, models))
                    row.update(audio_features(path))
                else:
                    row.update(audio_features(path))
                file_rows[assignments[item.name]].append(row)
                progress.progress((index + 1) / len(uploads), text=f"Processed {item.name}")
        features = pd.DataFrame([aggregate_child(rows, child) for child, rows in file_rows.items()])
        questionnaire = pd.DataFrame(questionnaire_rows)
        final = features.merge(questionnaire, on="child_id", how="left")
        risks = pd.DataFrame([{**{"child_id": record["child_id"]}, **prototype_domain_scores(record, record)} for record in final.to_dict("records")])
        for column in [name for name in risks.columns if name != "child_id"]:
            level = column.replace("Risk", "Risk Level")
            risks[level] = risks[column].map(lambda score: "High" if score >= .67 else "Mid" if score >= .34 else "Low")
        st.session_state["fusion_features"] = final
        st.session_state["prototype_risks"] = risks
        st.session_state["dashboard_ready"] = True
    except Exception as error:
        st.exception(error)
        st.info("The full Phase 4 stack is resource intensive. For a Streamlit Cloud demo, use short clips and the Whisper `tiny` model configured in feature_extractors.py.")

if st.session_state.get("dashboard_ready"):
    final = st.session_state["fusion_features"]
    risks = st.session_state["prototype_risks"]
    st.success("Feature extraction complete.")
    st.subheader("Final prototype indicators")
    st.caption("These are transparent, non-XGBoost demo composites. Replace them with trained Phase 6 predictions only after you obtain labelled child-level data.")
    st.dataframe(risks, use_container_width=True)
    left, right = st.columns(2)
    left.download_button("Download Phase 1-5 fusion features CSV", final.to_csv(index=False).encode(), "fusion_features.csv", "text/csv", key="download_fusion")
    right.download_button("Download prototype indicator CSV", risks.to_csv(index=False).encode(), "prototype_risk_indicators.csv", "text/csv", key="download_indicators")
    with st.expander("Show extracted feature table"):
        st.dataframe(final, use_container_width=True)
