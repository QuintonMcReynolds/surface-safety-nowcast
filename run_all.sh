#!/bin/zsh
# Full pipeline, in order. Everything runs on CPU (~4 h on an M4, most of it Whisper).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export ASR_DEVICE=${ASR_DEVICE:-cpu} HF_HUB_DISABLE_PROGRESS_BARS=1

# 1. data: three ADS-B Exchange sample days at KIAD (12:00-18:00 UTC), OSM airport layout, ATC speech
for day in 2025-08-01 2025-09-01 2025-10-01; do $PY src/fetch_adsbx.py $day 12 18 KIAD; done
$PY src/fetch_osm.py
$PY src/fetch_speech.py

# 2. tracks + hindsight labels
$PY src/tracks.py

# 3. models (independent; can run in parallel)
$PY src/activity.py     # state estimation: CT-HMM / GBM / hybrid, calibration under degradation
$PY src/eta.py          # time to runway entry / takeoff roll, conformal + adaptive conformal intervals
$PY src/incursion.py    # hold-short violation alerts: lead time vs false alerts
$PY src/fusion.py       # ADS-B + TIS-B + simulated camera: EKF + JPDA, clock offset, input loss

# 4. speech: zero-shot Whisper, LoRA fine-tune, fine-tuned eval, then what it means for alerts
$PY src/asr.py eval openai/whisper-small whisper-small
$PY src/asr.py train openai/whisper-small models/whisper-small-atc-lora 500
$PY src/asr.py eval models/whisper-small-atc-lora whisper-small-atc-lora
$PY src/voice_gate.py

# 5. figures
$PY src/report.py
