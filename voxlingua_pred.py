import whisper
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
from lang.utils import get_iso_693_3
import pickle as pkl
import torch
from speechbrain.pretrained import EncoderClassifier

OUTPUT_DIR = "./voxlingua_results"
device = 'cuda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

fleurs_langID = load_dataset("google/fleurs", "all", split="test") # to download all data

pred = []
true = []

decode = {i: get_iso_693_3(fleurs_langID.features["lang_id"].names[i].split('_')[0]) for i in range(102)}
language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")

with torch.no_grad():
    for i in tqdm(range(100)): # fleurs_langID.num_rows
        audio = torch.tensor(fleurs_langID[i]['audio']['array']).to(device)
        pred_id = language_id.classify_batch(audio)[3][0]
        del audio
        pred_lang = get_iso_693_3(pred_id)
        pred.append(pred_lang)

        true_label_id = fleurs_langID[i]['lang_id']
        true.append(decode[true_label_id])

pkl.dump(pred, open(os.path.join(OUTPUT_DIR, "voxlingua_pred.pkl"), "wb"))
pkl.dump(true, open(os.path.join(OUTPUT_DIR, "voxlingua_true.pkl"), "wb"))
