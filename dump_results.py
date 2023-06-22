import pickle as pkl
from sklearn.metrics import classification_report, accuracy_score
from lang.utils import get_iso_693_3
from lang.fleurs import FLEURS_LANGS

for dataset in ['voxlingua', 'whisper']:
    pred = pkl.load(open(f'fleurs_{dataset}_results/{dataset}_prediction.pkl', 'rb'))
    true = pkl.load(open(f'fleurs_{dataset}_results/{dataset}_true_labels.pkl', 'rb'))

    pred = [get_iso_693_3(l) for l in pred]
    true = [get_iso_693_3(l) for l in true]

    with open(f'fleurs_{dataset}_results/{dataset}_report.txt', 'w') as f:
        print(accuracy_score(true, pred, labels=list(FLEURS_LANGS.keys()), zero_division=0, digits=4), file=f)
    
    print(dataset, 'done')