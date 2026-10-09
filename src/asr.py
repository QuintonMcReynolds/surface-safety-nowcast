"""Whisper on ATC audio: zero-shot evaluation, LoRA fine-tuning, latency.

  python src/asr.py eval  openai/whisper-small           # zero-shot
  python src/asr.py train openai/whisper-small models/whisper-small-lora
  python src/asr.py eval  models/whisper-small-lora      # fine-tuned
Train: UWB-ATCC train split (Czech approach/tower, ~10.5 h). Test: UWB-ATCC test (held-out
utterances, same domain) and the ATCO2 1 h set (other airports; never seen in training)."""
import io, json, random, sys, time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent))
from atc_text import normalise, parse

ROOT = Path(__file__).resolve().parents[1]
import os
DEV = os.environ.get("ASR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
SPEECH = ROOT / "data/speech"


def load(split, n=None, seed=0):
    files = sorted((SPEECH / ("atco2_corpus_1h" if split == "atco2" else "uwb_atcc") / "data").glob(
        "test-*.parquet" if split in ("atco2", "uwb_test") else "train-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df.duration > 0.6) & (df.duration < 29)].reset_index(drop=True)
    if n and n < len(df):
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return df


def audio(row):
    a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000:  # UWB-ATCC is 8 kHz radio audio
        from scipy.signal import resample_poly
        g = np.gcd(16000, sr); a = resample_poly(a, 16000 // g, sr // g).astype("float32")
    return a


def get_model(name):
    base = name
    cfg = Path(name) / "adapter_config.json"
    if cfg.exists():
        base = json.load(open(cfg))["base_model_name_or_path"]
    proc = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    if cfg.exists():
        model = PeftModel.from_pretrained(model, name).merge_and_unload()
    model.generation_config.forced_decoder_ids = None
    return proc, model.to(DEV).eval()


@torch.no_grad()
def transcribe(proc, model, wavs):
    feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
    ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=128, num_beams=1)
    return proc.batch_decode(ids, skip_special_tokens=True)


SLOTS = ["callsign", "runway", "hold_short", "cross", "line_up", "cleared_takeoff", "cleared_land", "via"]


def evaluate(name, tag, uwb_n=600):
    proc, model = get_model(name)
    res = {}
    for split, n in [("uwb_test", uwb_n), ("atco2", None)]:
        df = load(split, n)
        hyps = []
        for i in range(0, len(df), 16):
            hyps += transcribe(proc, model, [audio(r) for _, r in df.iloc[i:i + 16].iterrows()])
        ref = [normalise(t) for t in df.text]; hyp = [normalise(h) for h in hyps]
        keep = [i for i, r in enumerate(ref) if r]
        ref = [ref[i] for i in keep]; hyp = [hyp[i] or "<empty>" for i in keep]
        # intent agreement: parse the human transcript vs parse the ASR output
        pr, ph = [parse(r) for r in ref], [parse(h) for h in hyp]
        slot = {}
        for k in SLOTS:
            has = [i for i in range(len(pr)) if pr[i][k] not in (None, False, [])]
            if has:
                slot[k] = {"n": len(has), "recall": float(np.mean([pr[i][k] == ph[i][k] for i in has]))}
        false_cmd = float(np.mean([any(ph[i][k] is True and pr[i][k] is False for k in
                                       ["hold_short", "cross", "line_up", "cleared_takeoff", "cleared_land"])
                                   for i in range(len(pr))]))
        res[split] = {"n": len(ref), "hours": float(df.duration.sum() / 3600),
                      "wer": jiwer.wer(ref, hyp), "slots": slot, "spurious_command_rate": false_cmd}
        pd.DataFrame({"ref": ref, "hyp": hyp}).to_csv(ROOT / f"reports/asr_{tag}_{split}.csv", index=False)
        print(tag, split, f"WER {res[split]['wer']:.3f}", {k: round(v['recall'], 3) for k, v in slot.items()}, flush=True)
    # latency: one transmission at a time, as in live operation
    df = load("atco2", 80, seed=1)
    lat = []
    for _, r in df.iterrows():
        a = audio(r); transcribe(proc, model, [a]) if not lat else None
        t = time.perf_counter(); transcribe(proc, model, [a])
        if DEV == "mps": torch.mps.synchronize()
        lat.append((time.perf_counter() - t, r["duration"]))
    lat = np.array(lat)
    res["latency"] = {"p50_s": float(np.median(lat[:, 0])), "p95_s": float(np.percentile(lat[:, 0], 95)),
                      "median_audio_s": float(np.median(lat[:, 1])), "device": DEV}
    print(tag, "latency", res["latency"], flush=True)
    json.dump(res, open(ROOT / f"reports/asr_{tag}.json", "w"), indent=1)


def train(base, out, epochs=1, bs=4, lr=1e-3, r=32, max_steps=None):
    proc = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()
    model.to(DEV).train()
    df = load("train")
    df = df[df.text.str.split().str.len() >= 2].reset_index(drop=True)
    prefix = proc.tokenizer("", add_special_tokens=True).input_ids[:-1]  # <|sot|><|en|><|transcribe|><|notimestamps|>
    proc.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * (len(df) // bs) if max_steps is None else max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)
    order = [i for _ in range(epochs) for i in random.Random(0).sample(range(len(df)), len(df))]
    t0, step = time.time(), 0
    for step in range(steps):
        rows = df.iloc[order[step * bs:(step + 1) * bs]]
        feats = proc.feature_extractor([audio(r) for _, r in rows.iterrows()], sampling_rate=16000,
                                       return_tensors="pt").input_features.to(DEV)
        lab = proc.tokenizer([t for t in rows.text], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
        dec_in = labels[:, :-1].clone().masked_fill(labels[:, :-1] == -100, proc.tokenizer.pad_token_id)
        tgt = labels[:, 1:]
        loss = model(input_features=feats, decoder_input_ids=dec_in.to(DEV), labels=None).logits
        loss = torch.nn.functional.cross_entropy(loss.reshape(-1, loss.shape[-1]), tgt.to(DEV).reshape(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if DEV == "mps" and step % 25 == 0:
            torch.mps.empty_cache()
        if step % 50 == 0 or step < 3:
            el = time.time() - t0
            print(f"step {step}/{steps} loss {loss.item():.3f}  {el/60:.1f} min, eta {el/(step+1)*(steps-step-1)/60:.0f} min", flush=True)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    json.dump({"base": base, "steps": steps, "batch": bs, "lr": lr, "rank": r, "train_utts": len(df),
               "train_hours": float(df.duration.sum() / 3600), "minutes": (time.time() - t0) / 60},
              open(Path(out) / "train_meta.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "eval":
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else Path(sys.argv[2]).name)
    else:
        train(sys.argv[2], sys.argv[3], max_steps=int(sys.argv[4]) if len(sys.argv) > 4 else None)
