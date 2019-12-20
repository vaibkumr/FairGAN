# About
Word embeddings like GloVe are gender biased. There has been many approaches to mitigate it. In this project we mitigate gender bias through a GAN based approach.
- Discriminator: or `SexistDiscriminator` as we call it, is neural net with the goal of becoming the most sexist thing on earth. It wants to predict the gender of a gender-neutral word (like nurse or receptionist) just by its word embedding.
- Generator: Generates new embeddings for a word such that it fools the discriminator as well as is close to the original embedding of the word (to preserve semantic meaning).

# Disclaimer
- This is not the code for the paper titled "FairGAN: Fairness-aware Generative Adversarial Networks" or any other paper.
- There is no paper associated with the code here, this is just a fun little project I did in my third year of undergraduate.
- We got this idea back in 2018 and wanted to write a paper until we found this amazing paper "Mitigating Unwanted Biases with Adversarial Learning" by Zhang et. al. which is a better and more complex version of my work here.
- This is 100% our original idea, work and code.

# How
- Just skim through `Results and Debiasing.ipynb` to see how well this method performs in debiasing direct bias (bolukbasi et. al.)
- Run `FairGAN.py` to create, train and save generator and discriminator models.

## Generator
- Input: Random noise and original word embedding
- Output: New word embedding

## Discriminator
- Input: Word embedding
- Output: Gender (1 for male, 0 for female, we assume that gender is binary)

## Note
With some more innovation and work, this project has the potential of becoming a research paper. I will try to work more on this as soon as I can find time.


![](https://i.imgur.com/OsyhIY5.jpg)

-- Yuzuru, Koe no Katachi
Don't be a sexist!
