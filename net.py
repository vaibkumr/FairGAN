import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator wants to create new embeddings such that it can fool the
    sexist Discriminator.
    """
    def __init__(self, nz, nd=300):
        super().__init__()
        self.fc1 = nn.Linear(nz, 300)
        self.act = nn.LeakyReLU()
        self.model = nn.Sequential(
            nn.Linear(600, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, nd),
        )

    def forward(self, z, c):
        """
        @param z is noise
        @param c is the current embedding, generator uses this information
        """
        x = self.act(self.fc1(z))
        x = torch.cat([x, c], dim=1)
        x = self.model(x)
        return x / torch.norm(x) #return normalized embedding

class SexistDiscriminator(nn.Module):
    """
    The SexistDiscriminator takes in the embedding and classifies it as either
    male or female. It's goal is to become the most sexist thing on earth.
    """
    def __init__(self, nd=300):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nd, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sig(x).squeeze()
