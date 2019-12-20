import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def get_gender_logit(word, E, g):
    """
    Return the dot product of a word and gender direction (g)
    """
    return E.v(word).dot(g)

def get_gender(word, E, g):
    """
    self explanatory
    """
    return 1 if E.v(word).dot(g) < 0 else 0 #1 is Male, 0 is female

def get_rgender(word, E, g):
    """
    return the reverse gender
    """
    return 0 if E.v(word).dot(g) < 0 else 1 #reversed gender

class gender_set(Dataset):
    """
    Dataset class for training GAN
    """
    def __init__(self, words, E, g, device):
        """
        @param words is list of words to debias
        @E is the embedding
        @g is gender direction
        @device is either cuda of cpu
        """
        self.words = words
        self.embs = torch.Tensor([E.v(w) for w in words]).to(device)
        self.gender = torch.Tensor([get_gender(w, E, g) for w in words]).to(device)
        self.rgender = torch.Tensor([get_rgender(w, E, g) for w in words]).to(device)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        """
        We want to return the word, its embedding, its gender and its reverse gender
        """
        return self.words[idx], self.embs[idx], self.gender[idx], self.rgender[idx]
