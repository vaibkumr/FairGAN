import gensim
import gzip
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pickle
import we

def clean_vocab(words, E):
    for w in words:
        if w not in E.words:
            words.remove(w)
    return words

def normalize(v):
    return v / np.linalg.norm(v)

def get_max(E, g, words):
    df = pd.DataFrame(data={"word": list(words)})
    b = {}
    for word in tqdm(words):
        b[word] = get_gender_logit(word, E, g)
    df['gender_score'] = df.word.map(b)
    df.sort_values(by='gender_score', inplace=True)
    return df

def gender_subspace(E):
        pairs = [
              ("woman", "man"),
              ("her", "his"),
              ("she", "he"),
              ("aunt", "uncle"),
              ("niece", "nephew"),
              ("daughters", "sons"),
              ("mother", "father"),
              ("daughter", "son"),
              ("granddaughter", "grandson"),
              ("girl", "boy"),
              ("stepdaughter", "stepson"),
              ("mom", "dad"), ]
        difs = []
        for f, m in pairs:
            difs.append(E.v(f)- E.v(m))
        difs = np.array(difs)
        #PCA
        difs = np.cov(np.array(difs).T)
        evals, evecs = np.linalg.eig(difs)
        return normalize(np.real(evecs[:, np.argmax(evals)]))

def gender_subspace_simple(E):
    return normalize(E.v('she') - E.v('he'))

# def load_words():
#     words = []
#     with open('w2v_gnews_small.vocab', 'r') as handle:
#         for line in handle.readlines():
#             words.append(line.strip())
#     return words
# words = load_words()



def get_gender_logit(word, E, g):
    return E.v(word).dot(g)

def get_gender(word, E, g):
    return 1 if E.v(word).dot(g) < 0 else 0 #1 is Male, 0 is female

def get_rgender(word, E, g):
    return 0 if E.v(word).dot(g) < 0 else 1 #reversed gender

def label_to_gender(label):
    return 'M' if label else 'F'
