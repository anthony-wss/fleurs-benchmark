import pickle as pkl
from sklearn.metrics import accuracy_score
from lang.utils import get_iso_693_3
from lang.fleurs import FLEURS_LANGS

dataset = 'whisper'
pred = pkl.load(open(f'fleurs_{dataset}_results-medium/{dataset}_pred.pkl', 'rb'))
true = pkl.load(open(f'fleurs_{dataset}_results-medium/{dataset}_true.pkl', 'rb'))

CJK = ['yue', 'ja', 'ko', 'cmn']
CMN = ['ar', 'az', 'he', 'kk', 'ky', 'mn', 'ps', 'fa', 'ckb', 'tg', 'tr', 'uz']
EE = ['hy', 'be', 'bg', 'cs', 'et', 'ka', 'lv', 'lt', 'mk', 'pl', 'ro', 'ru', 'sr', 'sk', 'sl', 'uk']
SA = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sd', 'ta', 'te', 'ur']
SEA = ['my', 'ceb', 'tl', 'id', 'jv', 'km', 'lo', 'ms', 'mi', 'th', 'vi']
SSA = ['af', 'am', 'ff', 'lg', 'ha', 'ig', 'kam', 'ln', 'luo', 'nso', 'ny', 'om', 'sn', 'so', 'sw', 'umb', 'wo', 'xh', 'yo', 'zu']
WE = ['ast', 'bs', 'ca', 'hr', 'da', 'nl', 'en', 'fi', 'fr', 'gl', 'de', 'el', 'hu', 'is', 'ga', 'it', 'kea', 'lb', 'mt', 'no', 'oc', 'pt', 'es', 'sv', 'cy']
CJK = [get_iso_693_3(l) for l in CJK]
CMN = [get_iso_693_3(l) for l in CMN]
EE = [get_iso_693_3(l) for l in EE]
SA = [get_iso_693_3(l) for l in SA]
SEA = [get_iso_693_3(l) for l in SEA]
SSA = [get_iso_693_3(l) for l in SSA]
WE = [get_iso_693_3(l) for l in WE]

def filter(pred, true, labels):
    ret = []
    for i in range(len(pred)):
        if true[i] in labels:
            ret.append(i)
    return ret

total_len = 0

for group in [WE, EE, CMN, SSA, SA, SEA, CJK]:
    ids = filter(pred, true, group)
    total_len += len(ids)
    print(group, accuracy_score([pred[i] for i in ids], [true[i] for i in ids]))
