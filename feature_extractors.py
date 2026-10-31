"""Inference adapters for the supplied NeuroScreen Phase 1-4 artifacts.

These functions use the architectures and preprocessing from the notebooks in
this project. They return features only; they do not diagnose autism.
"""
from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
BEHAVIOURS = ["arm_flapping", "head_banging", "spinning"]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PoseLSTM(nn.Module):
    """Architecture used by behavior_group_best.pth in Phase3_Retrain.ipynb."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(132, 128, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.fc(sequence[:, -1, :])


def _state_dict(payload):
    return payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload


def load_models(root: str | Path, device: str = "cpu"):
    """Load the three provided checkpoints from the project root."""
    import timm
    from torchvision import models

    root = Path(root); device = torch.device(device)
    p1 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    p1.load_state_dict(_state_dict(torch.load(root / "emotion_controlled_finetune_best.pth", map_location=device)))
    p2 = models.resnet50(weights=None); p2.fc = nn.Linear(p2.fc.in_features, 2)
    p2.load_state_dict(_state_dict(torch.load(root / "autism_face_best.pth", map_location=device)))
    p3 = PoseLSTM()
    p3.load_state_dict(_state_dict(torch.load(root / "behavior_group_best.pth", map_location=device)))
    for model in (p1, p2, p3): model.to(device).eval()
    return p1, p2, p3, device


def video_features(video_path: str | Path, models_bundle, max_frames: int = 30) -> dict:
    """Sample evenly spaced frames and return one Phase 1-3 feature record."""
    p1, p2, p3, device = models_bundle
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release(); raise ValueError("Video has no readable frames")
    indices = np.linspace(0, total - 1, max_frames).astype(int)
    images, landmarks = [], []

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = _pose_asset()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.IMAGE
    )
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame = capture.read()
            if not ok: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(IMAGE_TRANSFORM(Image.fromarray(rgb)))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.pose_landmarks:
                landmark = [value for point in result.pose_landmarks[0] for value in (point.x, point.y, point.z, point.visibility)]
                landmarks.append(np.asarray(landmark, dtype=np.float32))
            else:
                landmarks.append(np.zeros(132, dtype=np.float32))
    capture.release()
    if not images: raise ValueError("No usable frames were decoded")
    with torch.no_grad():
        batch = torch.stack(images).to(device)
        emotion = F.softmax(p1(batch), dim=1).mean(dim=0).cpu().numpy()
        face = F.softmax(p2(batch), dim=1).mean(dim=0).cpu().numpy()
        padded = np.zeros((30, 132), dtype=np.float32)
        padded[:min(30, len(landmarks))] = landmarks[:30]
        behaviour = F.softmax(p3(torch.from_numpy(padded).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    row = {f"p1_emotion_{name}": float(score) for name, score in zip(EMOTIONS, emotion)}
    row.update({"p2_autistic_probability": float(face[0]), "p2_non_autistic_probability": float(face[1]),
                "p3_pose_detection_rate": float(np.any(padded != 0, axis=1).mean())})
    row.update({f"p3_{name}_probability": float(score) for name, score in zip(BEHAVIOURS, behaviour)})
    return row


def _pose_asset() -> Path:
    cache = Path(tempfile.gettempdir()) / "pose_landmarker_lite.task"
    if not cache.exists():
        import urllib.request
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", cache)
    return cache


def audio_features(media_path: str | Path, whisper_model: str = "tiny") -> dict:
    """Phase 4-compatible audio biomarkers, adapted from Phase4_Audio_Analysis.ipynb."""
    import opensmile
    import whisper
    from silero_vad import get_speech_timestamps, load_silero_vad

    media_path = Path(media_path)
    wav_path = Path(tempfile.gettempdir()) / f"neuroscreen_{media_path.stem}.wav"
    command = ["ffmpeg", "-y", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(wav_path)]
    complete = subprocess.run(command, capture_output=True, text=True)
    if complete.returncode: raise RuntimeError("ffmpeg could not extract audio")
    data, sample_rate = sf.read(wav_path, dtype="float32")
    duration = len(data) / sample_rate
    vad = load_silero_vad(); segments = get_speech_timestamps(torch.from_numpy(data), vad, sampling_rate=16000, threshold=0.5,
        min_speech_duration_ms=150, min_silence_duration_ms=300)
    spans = [(item["start"] / 16000, item["end"] / 16000) for item in segments]
    speech_time = sum(end - start for start, end in spans)
    pauses = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    pauses = [pause for pause in pauses if pause >= .3]
    transcription = _whisper(whisper_model).transcribe(str(wav_path), fp16=False)["text"].strip()
    word_count = len(transcription.split())
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    values = smile.process_file(str(wav_path)).iloc[0]
    f0 = float(values["F0semitoneFrom27.5Hz_sma3nz_amean"])
    return {
        "audio_duration_s": round(duration, 2), "audio_speech_rate_wpm": round(word_count / max(speech_time, duration, .01) * 60, 1),
        "audio_pause_count": len(pauses), "audio_avg_pause_duration_s": float(np.mean(pauses)) if pauses else 0.,
        "audio_response_latency_s": max(0., spans[0][0]) if spans else np.nan,
        "audio_pitch_mean_hz": 27.5 * 2 ** (f0 / 12),
        "audio_energy_loudness_mean": float(values["loudness_sma3_amean"]),
        "audio_speech_ratio": speech_time / duration if duration else 0.,
    }


@lru_cache(maxsize=1)
def _whisper(model_name: str):
    """Load Whisper once per Streamlit process, not once per uploaded file."""
    import whisper
    return whisper.load_model(model_name)


def aggregate_child(rows: list[dict], child_id: str) -> dict:
    """Turn all files for one child into a single mean/max/std feature row."""
    data = pd.DataFrame(rows); output = {"child_id": child_id, "recording_count": len(data)}
    for column in data.select_dtypes(include="number"):
        output[f"{column}__mean"] = float(data[column].mean())
        output[f"{column}__max"] = float(data[column].max())
        output[f"{column}__std"] = float(data[column].std(ddof=0))
    return output


def prototype_domain_scores(features: dict, questionnaire: dict | None = None) -> dict:
    """Transparent college-demo scores; replace with Phase 6 XGBoost after training."""
    def feature(name): return float(features.get(f"{name}__mean", 0.) or 0.)
    q = questionnaire or {}
    score = {
        "Social Communication Risk": .45 * feature("p2_autistic_probability") + .25 * (1 - feature("p1_emotion_happy")) + .30 * float(q.get("q_axis_sc", 0)),
        "Attention Regulation Risk": .40 * feature("p2_autistic_probability") + .30 * (1 - feature("p3_pose_detection_rate")) + .30 * float(q.get("q_axis_ar", 0)),
        "Speech & Language Risk": .45 * min(1., feature("audio_pause_count") / 10) + .25 * min(1., feature("audio_response_latency_s") / 5) + .30 * float(q.get("q_axis_sl", 0)),
        "Repetitive Behaviour Risk": .25 * feature("p3_arm_flapping_probability") + .25 * feature("p3_head_banging_probability") + .20 * feature("p3_spinning_probability") + .30 * float(q.get("q_axis_rb", 0)),
    }
    return {name: round(float(np.clip(value, 0, 1)), 3) for name, value in score.items()}
