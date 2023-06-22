import whisper
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
from lang.utils import get_iso_693_3
import pickle as pkl
import torch

OUTPUT_DIR = "fleurs_whisper_results-medium"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fleurs_langID = load_dataset("google/fleurs", "all", split="test") # to download all data

model = whisper.load_model("medium", device='cuda')
pred = []
true = []

decode = {i: get_iso_693_3(fleurs_langID.features["lang_id"].names[i].split('_')[0]) for i in range(102)}

out_pred = open(os.path.join(OUTPUT_DIR, "whisper_pred.txt"), 'w')
out_true = open(os.path.join(OUTPUT_DIR, "whisper_true.txt"), 'w')

with torch.no_grad():
    for i in tqdm(range(fleurs_langID.num_rows)):
        audio = fleurs_langID[i]['audio']['array'].astype(np.float32)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        pred_id = max(probs, key=probs.get)
        pred.append(get_iso_693_3(pred_id))
        print(get_iso_693_3(pred_id), file=out_pred, flush=True)

        true_label_id = fleurs_langID[i]['lang_id']
        true.append(decode[true_label_id])
        print(decode[true_label_id], file=out_true, flush=True)

pkl.dump(pred, open(os.path.join(OUTPUT_DIR, "whisper_pred.pkl"), "wb"))
pkl.dump(true, open(os.path.join(OUTPUT_DIR, "whisper_true.pkl"), "wb"))