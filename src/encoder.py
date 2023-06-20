import codecs
import pickle
from json import dumps, loads

import pandas as pd

from sklearn.preprocessing import LabelEncoder


def encode_scaler(scaler):
    chans = scaler[0].index.tolist()
    means = scaler[0].values.tolist()
    stds = scaler[1].values.tolist()
    encoded = f'{chans}*{means}*{stds}'
    # encoded = encoded.replace(' ', '')
    return encoded


def encode_label(label_encoder: LabelEncoder):
    labels = label_encoder.classes_
    labels = getattr(labels, "tolist", lambda: labels)()
    labels_str = dumps(list(labels))
    
    codes = list(label_encoder.transform(labels))
    codes = getattr(codes, "tolist", lambda: codes)()
    encoded = f'{labels_str}*{codes}'
    return encoded


def decode_scaler(encoded):
    chans, means, stds = encoded.split('*')
    scaler = [pd.Series(data=[float(mean) for mean in means[1:-1].split(', ')],
                        index=[int(chan) for chan in chans[1:-1].split(', ')]),
              pd.Series(data=[float(std) for std in stds[1:-1].split(', ')],
                        index=[int(chan) for chan in chans[1:-1].split(', ')])
              ]
    return scaler

def decode_label(encoded):
    labels, codes = encoded.split('*')
    labels = loads(labels)
    
    le = LabelEncoder()
    le.fit_transform(labels)
    
    return le


def encode_pickle(obj):
    encoded = codecs.encode(pickle.dumps(obj, 4), "base64").decode()
    return encoded


def decode_pickle(encoded):
    obj = pickle.loads(codecs.decode(encoded.encode(), "base64"))
    return obj
