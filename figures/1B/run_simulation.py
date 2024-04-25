import math
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def simulate(iter_, mu, sigma, xlims, width):
    """
    Description:
        - Estimate how many cells drawn from normal (mu, sigma) bounded by xlims are sahred across adjacent tissue sections
    Parameters:
        - iter_: int, number of cells to simulate
        - mu: float, cell diameter mean (um)
        - sigma: float, cell diameter standard devation (um)
        - xlims: tuple, minimum and maximum bounds of space (um)
        - width: float, slide separation (um)
    Returns:
        - shared: int, number of cells partially contained in section
        - unshared: int, number of cells fully contained in section
    """
    shared = 0
    unshared = 0
    for i in range(iter_):
        r = np.random.normal(mu, sigma)/2
        x = np.random.uniform(low=xlims[0],high=xlims[1])
        min_ = x-r
        max_ = x+r
        if (min_ < width < max_):  # shared case
            shared += 1
        elif (0 < max_ < width): # unshared case
            unshared +=1
        else:
            continue
    return shared, unshared

def get_stats(cell_area):
    area = np.array(cell_area)
    equiv_diam = 2*0.65*(area/math.pi)**0.5
    mu = np.median(equiv_diam)
    sigma = np.std(equiv_diam)
    return mu, sigma

results = {}
cores = ['A10','A5','A6','A8','B10','B1','B3','B4',
         'B6','B7','B9','C10','C1','C4','D11','D2','D8',
         'E10','E3','E5','E6','E7','E8','E9','F11','F1','F2','F7',
         'G11','G1','G3','G6','G7','G9','H10','H1',
         'H2','H3','H4','H6','H7','H8','H9']
for c in tqdm(cores):
    nuc_table = pd.read_csv(f'/home/groups/ChangLab/heussner/tma-integration/data/nuc_tables/{c}_tCyCIF_tumor_nuc.csv')
    nm,ns = get_stats(nuc_table['area'])
    nuc_intact,nuc_edge = simulate(100000,nm,ns,[-20,25],5)
    results[c]=nuc_intact/(nuc_intact+nuc_edge)

with open('coexist/notebooks/figures/1B/simulation_results.pkl','wb') as handle:
    pickle.dump(results,handle)