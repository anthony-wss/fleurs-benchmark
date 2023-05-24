import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
import dataclasses, iso639
from tqdm import tqdm
import os

OUTPUT_DIR = "./voxlingua_results"

device = 'cuda'
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch

def get_iso_693_3(lang_id):
    if lang_id == 'jw':
        lang_id = 'jv'
    if lang_id == 'iw':
        return 'iw111'
    language = iso639.Language.match(lang_id)
    return dataclasses.asdict(language)['part3']

fleurs_langID = load_dataset("google/fleurs", "all", split="test") # to download all data

import torchaudio
from speechbrain.pretrained import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")
# Download Thai language sample from Omniglot and cvert to suitable form
# signal = language_id.load_audio("https://omniglot.com/soundfiles/udhr/udhr_th.mp3")
# prediction =  language_id.classify_batch(signal)

num_samples = fleurs_langID.num_rows
# num_samples = 100

# audios = []
# for i in tqdm(range(num_samples)):
#     audios.append(torch.tensor(fleurs_langID[i]['audio']['array']))
decode = {i: fleurs_langID.features["lang_id"].names[i].split('_')[0] for i in range(102)}
true_label = [get_iso_693_3(decode[fleurs_langID[i]['lang_id']]) for i in range(num_samples)]

# test_loader = DataLoader(audios, batch_size=4, shuffle=True, collate_fn=collate_fn_padd)
print('preprocess done.', num_samples)

preds = []

output_file = open(os.path.join(OUTPUT_DIR, "voxlingua_pred.txt"), "a")

with torch.no_grad():
    for i in tqdm(range(41582, num_samples)):
        audio = torch.tensor(fleurs_langID[i]['audio']['array']).to(device)
        pred_id = language_id.classify_batch(audio)[3][0]
        del audio
        pred_lang = get_iso_693_3(pred_id)
        preds.append(pred_lang)
        print(i, pred_lang, file=output_file)

import pickle as pkl

pkl.dump(preds, open(os.path.join(OUTPUT_DIR, "voxlingua_prediction.pkl"), "wb"))
pkl.dump(true_label, open(os.path.join(OUTPUT_DIR, "voxlingua_true_labels.pkl"), "wb"))

from sklearn.metrics import classification_report
print(classification_report(true_label, preds, zero_division=0), file=open(os.path.join(OUTPUT_DIR, 'voxlingua_report.txt'), "w"))