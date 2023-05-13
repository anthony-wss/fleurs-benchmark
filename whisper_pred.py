import whisper
import numpy as np

from datasets import load_dataset
import dataclasses, iso639
from tqdm import tqdm
import os

OUTPUT_DIR = "./whisper_results"

def get_iso_693_3(lang_id):
    language = iso639.Language.match(lang_id)
    return dataclasses.asdict(language)['part3']

fleurs_langID = load_dataset("google/fleurs", "all", split="test") # to download all data

model = whisper.load_model("base")
preds = []
true_label = []

decode = {i: fleurs_langID.features["lang_id"].names[i].split('_')[0] for i in range(102)}

for i in tqdm(range(fleurs_langID.num_rows)):
    audio = fleurs_langID[i]['audio']['array'].astype(np.float32)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    pred_id = max(probs, key=probs.get)
    preds.append(get_iso_693_3(pred_id))
    true_label_id = fleurs_langID[i]['lang_id']
    true_label.append(get_iso_693_3(decode[true_label_id]))

import pickle as pkl

pkl.dump(preds, open(os.path.join(OUTPUT_DIR, "whisper_prediction.pkl"), "wb"))
pkl.dump(preds, open(os.path.join(OUTPUT_DIR, "whisper_true_labels.pkl"), "wb"))

from sklearn.metrics import classification_report

class_names = [get_iso_693_3(decode[s]) for s in range(102)]
print(classification_report(true_label, preds, labels=list(range(len(class_names))), target_names=class_names, zero_division=0), file=open(os.path.join(OUTPUT_DIR, 'whisper_report.txt'), "w"))