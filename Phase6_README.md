# Phase 6 - Multimodal Fusion (XGBoost)

`Phase6_Multimodal_Fusion_XGBoost.ipynb` trains and runs the final four-output
fusion model described in the report. It supports many `.mp3`, `.wav`, or `.mp4`
recordings per child: first run the Phase 4 batch section over those files, then
pass its resulting `audio_features.csv` to `build_fusion_row`. Repeated recordings
are reduced to per-feature mean, standard deviation, and maximum plus a count.

## Required bridge outputs

The existing Phase 4 and Phase 5 notebooks produce the audio and questionnaire
CSV files. For raw video, Phases 1-3 must each export a numeric CSV with a
`child_id` column (one row per frame/window/video is fine). Suggested columns:

- Phase 1: `emotion_happy`, `emotion_sad`, `emotion_angry`, `emotion_fear`,
  `emotion_neutral`, `emotion_surprise`, `emotion_disgust`
- Phase 2: `face_risk_probability`, `eye_contact`, `face_attention`, `face_focus`
- Phase 3: `arm_flapping_probability`, `head_banging_probability`, `spinning_probability`

The three supplied `.pth` files are weight checkpoints, not runnable pipelines;
their model definitions, transforms and label mapping are not present in this
folder. Do not guess these details from the filenames. Add the corresponding
Phase 1-3 inference scripts, then export the bridge CSVs above.

## Training data required

XGBoost cannot be validly trained from the three checkpoints or from one child.
Create a de-identified `fusion_training.csv`, one row per child, with the fusion
features and four clinician-annotated binary labels:

`y_social_communication`, `y_attention_regulation`, `y_speech_language`, and
`y_repetitive_behaviour`.

The notebook refuses to train if any label lacks both classes. Report held-out
metrics, calibration, demographic subgroup performance, and external validation
before relying on the model in a real clinical workflow.
