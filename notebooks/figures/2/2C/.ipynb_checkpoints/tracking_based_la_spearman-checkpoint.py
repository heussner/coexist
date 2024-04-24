import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt
from tifffile import imread
import os
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import seaborn as sns
from utils import ids_to_table, plot_correlations, save_matches, filter_matches, corr_dist, get_correlations
import pickle

def ids_to_table(ids, table):
    match_table = table[table['CellID'].isin(ids)] # get relevant rows
    df1 = match_table.set_index('CellID')
    match_table = df1.reindex(ids) # set new table in correct order
    return match_table
    
save_path = '/home/groups/ChangLab/heussner/coexist/notebooks/figures/2/2C/tma-integration/results/Linear_assignment_radius'
immune_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1']

tumor_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31']

shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']

cores = ['A10','A5','A6','A8','B10','B1','B3','B4',
         'B6','B7','B9','C10','C1','C4','D11','D2','D8',
         'E10','E3','E5','E6','E7','E8','E9','F11','F1','F2','F7',
         'G11','G1','G3','G6','G7','G9','H10','H1',
         'H2','H3','H4','H6','H7','H8','H9']

for core in tqdm(cores):
    # read tCyCIF data
    
    tracking = pd.read_csv(f'/home/groups/ChangLab/heussner/tma-integration/results/Cell_tracking/{core}.csv')
    tracking = tracking[tracking['tumor_id']>0]
    #all cells tables
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/'
    tumor_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    immune_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    cols = ['x','y']
    sdist = pairwise_distances(immune_all[cols].to_numpy(), immune_all[cols].to_numpy())
    np.fill_diagonal(sdist,10000)
    ax0 = np.min(sdist, axis=0)
    radius = np.mean(ax0)+ 3*np.std(ax0)
    #normalize tracked cells based on full population
    immune_all_norm = immune_all[immune_markers].copy()
    immune_all_norm = zscore(immune_all_norm)
    immune_all_norm['CellID'] = immune_all['CellID'].copy()
    immune_tracked_norm = ids_to_table(tracking['immune_id'], immune_all_norm)
    
    # load matrices
    
    immune = immune_tracked_norm[shared_markers].copy()
    tumor = tumor_all[shared_markers].copy()
    
    tumor = zscore(tumor,axis=0)
    #pearson correlation distance

    cdist = 1 - spearmanr(a=immune,
                     b=tumor, 
                     axis=1)[0][0:len(immune),len(immune):len(immune)+len(tumor)]
    #euclidean distance between target cells
    cols = ['x','y']
    tumor_spatial = tumor_all[cols].to_numpy()
    sdist = pairwise_distances(tumor_spatial, tumor_spatial)
    
    matched_indexes = [tumor_all[tumor_all['CellID']==i].index for i in list(tracking['tumor_id'])]
    dist = cdist.copy()
    for i,match in enumerate(matched_indexes):
        # get target indices within radius r from sdist
        row = sdist[match,:].copy()
        indexes = list(np.where(row<=radius))
        # replace values in cdist
        mask = np.ones(dist[i,:].size, dtype=bool)
        mask[indexes] = False
        dist[i,:][mask] = np.inf
    
    rows, cols = linear_sum_assignment(dist)
    scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
    matching = [list(tracking['immune_id']), [tumor_all['CellID'].iloc[c] for c in cols], scores]
    table = pd.DataFrame(data={'immune_id':matching[0],'tumor_id':matching[1],'score':matching[2]})
    table.to_csv(os.path.join(save_path,f'{core}.csv'))
