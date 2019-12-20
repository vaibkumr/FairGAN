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
from utils import *
from dataloader import *
from net import *

#Reproducibility
SEED = 42 #The answer to everything
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_new_emb(word, netG):
    return netG(z[0].unsqueeze(dim=0), torch.Tensor(E.v(word)).unsqueeze(dim=0).to(device))

def test_generator(netG, g, device):
    with torch.no_grad():
        word_list = "nanny engineer warrior boss midwife".split()
        for word in word_list:
            orig = torch.Tensor(E.v(word)).to(device)
            new = get_new_emb(word, netG).squeeze()
            similarity = new.dot(orig)
            bias_bef = orig.dot(torch.Tensor(g).to(device))
            bias_aft = new.dot(torch.Tensor(g).to(device))
            print(f"Word: {word} Similarity: {similarity:.5f} | Bias: {bias_bef:.5f} / {bias_aft:.5f}")


# Hyperparameters and other
EMB = 'data/glove' #path to embedding in vector-vocab format. See we.py to know how this format works.
words_list_file = "data/toy_debias_list.pkl" #pickle file list of words to debias (small list from bulakbasi et. al.)
bs = 128
nz = 128
lrd = 1e-3
lrg = 1e-2
beta1 = 0.5
epochs = 1000
PATH_D = "models/D.pth"
PATH_G = "models/G.pth"

E = we.WordEmbedding(EMB) #Load embedding
g = gender_subspace_simple(E) #Get gender direction

with open(words_list_file, 'rb') as handle:
    words = pickle.load(handle) #Load word list to debias

words = clean_vocab(words, E) #Clean word list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init dataset and dataloader
train_dataset = gender_set(words, E, g, device)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)


netG = Generator(nz).to(device)
netD = SexistDiscriminator().to(device)

criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))

z = torch.randn(bs, nz, device=device) #Fixed noise

G_losses = []
G_Glosses = []
G_Mlosses = []
D_losses = []
D_Flosses = []
D_Rlosses = []

for epoch in tqdm(range(epochs)):
    for i, (word, emb, gender, rgender) in enumerate(train_loader):
        #######################
        # Update netD
        #######################
        netD.zero_grad()
        outputR = netD(emb)
        errD_real = criterion1(outputR, gender) #train discriminator to classify real data as Male or Female
        errD_real.backward()

#         noise = z[:emb.shape[0], :] #Fixed noise
        noise = torch.randn(emb.shape[0], nz, device=device) #Random noise

        fake = netG(noise, emb)
        outputF = netD(fake.detach()) #Detach, we dont want D to change G's weights
        errD_fake = criterion1(outputF, gender) #train discriminator to classify generated data as Male or Female
        errD_fake.backward(retain_graph=True)
        errD = errD_real + errD_fake

        optimizerD.step()

        ########################
        # Update netG
        ########################
        netG.zero_grad()

        M = torch.abs(torch.bmm(fake.unsqueeze(1), emb.unsqueeze(2))).mean()
        G = torch.abs(torch.matmul(fake, torch.Tensor(g).to(device))).mean()
        errG_meaning = 15*G - 5*M #We want to maximize M and minimize G
        errG_meaning.backward(retain_graph=True)

        errG_gender = criterion1(outputF, rgender) #fooling discriminator = how well has it predicted the opposite label (rgender) = how bad has it predicted
        errG_gender.backward()
        errG = errG_gender + errG_meaning

        optimizerG.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch: [{epoch+1} / {epochs}]")
        print(f"errD_real: {errD_real:.3f} | errD_fake: {errD_fake:.3f} | errG_gender: {errG_gender:.3f} | errG_meaning: {errG_meaning:.5f}")
        test_generator(netG, g, device)

    D_losses.append(errD.item())
    D_Flosses.append(errD_fake.item())
    D_Rlosses.append(errD_real.item())

    G_losses.append(errG.item())
    G_Glosses.append(errG_gender.item())
    G_Mlosses.append(errG_meaning.item())


if PATH_G:
    torch.save(netG.state_dict(), PATH_G)

if PATH_D:
    torch.save(netD.state_dict(), PATH_D)
