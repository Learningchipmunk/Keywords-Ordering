"""
Written by Jean-Charles Layoun, 24/09/2022

This file holds every script we used to compute the metrics of Information theory.
"""

import numpy as np
from sklearn.metrics import mutual_info_score 


def normalize_distrib(distrib):
    """Function that normalizes distribution with respect to the L1 norm"""
    return np.array(distrib)/np.linalg.norm(distrib, ord=1) if(np.linalg.norm(distrib, ord=1) != 0) else distrib

def entropy(distrib):
    """Function that computes the Entropy of a probability distribution p(x):
       -sum over x of p(x) * log(p(x))
    """
    eps = 1e-12
    if np.linalg.norm(distrib, ord=1)==0: return np.nan## We decide that Entropy is Nan when there is no occurences.
    return -np.sum(distrib * np.log(distrib + eps))


def kl_div(x, y):
    """Function that computes the kl divergence of two discrete distributions"""
    idx1 = np.intersect1d(np.where(x>0), np.where(y>0))
    idx2 = np.where(x==0)

    return (x[idx1]*np.log(x[idx1]/y[idx1])).sum() + y[idx2].sum()

def MutualInfoEmbed(kw1_embed, kw2_embed):
    """Function that computes the Mutual Information for two discrete distributions"""
    kw1_embed = np.abs(kw1_embed) / np.linalg.norm(kw1_embed, ord=2)
    kw2_embed = np.abs(kw2_embed) / np.linalg.norm(kw2_embed, ord=2)

    return mutual_info_score(kw1_embed, kw2_embed) 

def ComputeMetricsForKeywordPair(kw1_embed, kw2_embed, verbose=False):
    """Final function that computes the metrics given two keyword embeddings."""
    H1_var = entropy(kw1_embed)
    H2_var = entropy(kw2_embed)
    kl_div1 = kl_div(kw1_embed, kw2_embed).sum()
    kl_div2 = kl_div(kw2_embed, kw1_embed).sum()
    ratio   = kl_div1/kl_div2 if kl_div2 != 0 else np.nan
    # MI_var  = MutualInfoEmbed(kw1_embed, kw2_embed)

    if(verbose):
        print("Entropy for kw1:", H1_var)
        print("Entropy for kw2:", H2_var)
    # else:
    #     clear_output(wait=True)

    return (H1_var, H2_var, kl_div1, kl_div2, ratio)#, MI_var