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
import pickle

def ids_to_table(ids, table):
    match_table = table[table['CellID'].isin(ids)] # get relevant rows
    df1 = match_table.set_index('CellID')
    match_table = df1.reindex(ids) # set new table in correct order
    return match_table

def get_correlations(markers, immune_table, tumor_table):
    correlations = []
    for i,s in enumerate(markers):
        correlations.append(spearmanr(immune_table[s],tumor_table[s])[0])
    return correlations
    
save_path = '/home/groups/ChangLab/heussner/coexist/notebooks/figures/2/2A/Radius_exploration'
immune_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1']
tumor_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31']
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']
cores = ['B3']
for core in tqdm(cores):
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    tumor_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    immune_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    immune_all_sample = immune_all.copy()
    immune = immune_all_sample[shared_markers].copy()
    tumor = tumor_all[shared_markers].copy()
    
    immune = zscore(immune,axis=0)
    tumor = zscore(tumor,axis=0)
    cdist = 1 - spearmanr(a=immune.to_numpy(),
                     b=tumor.to_numpy(), 
                     axis=1)[0][0:len(immune),len(immune):len(immune)+len(tumor)]

    rows, cols = linear_sum_assignment(cdist)
    scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
    matching_table = pd.DataFrame(data={'immune':immune_all_sample['CellID'], 'tumor': [tumor_all['CellID'].iloc[c] for c in cols], 'score':scores})
    imtbl = ids_to_table(matching_table['immune'],immune_all_sample)
    tmtbl = ids_to_table(matching_table['tumor'],tumor_all)
    sla_all = np.mean(get_correlations(shared_markers, imtbl,tmtbl))
    with open(os.path.join(save_path,f'la_all_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_all, handle)
print('Done')