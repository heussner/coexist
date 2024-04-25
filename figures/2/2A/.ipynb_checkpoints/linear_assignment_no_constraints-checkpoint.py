import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from tqdm import tqdm

def ids_to_table(ids, table):
    match_table = table[table['CellID'].isin(ids)] # get relevant rows
    df1 = match_table.set_index('CellID')
    match_table = df1.reindex(ids) # set new table in correct order
    return match_table

def get_correlations(markers, slide_2_table, slide_1_table):
    correlations = []
    for i,s in enumerate(markers):
        correlations.append(spearmanr(slide_2_table[s],slide_1_table[s])[0])
    return correlations
    
save_path = 'coexist/notebooks/figures/2/2A/Radius_exploration'
slide_2_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1'] # CyCIF immune panel
slide_1_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31'] # CyCIF tumor panel
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']
cores = ['B3']
for core in tqdm(cores):
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    slide_2_all_sample = slide_2_all.copy()
    slide_2 = slide_2_all_sample[shared_markers].copy()
    slide_1 = slide_1_all[shared_markers].copy()
    
    slide_2 = zscore(slide_2,axis=0)
    slide_1 = zscore(slide_1,axis=0)
    cdist = 1 - spearmanr(a=slide_2.to_numpy(),
                     b=slide_1.to_numpy(), 
                     axis=1)[0][0:len(slide_2),len(slide_2):len(slide_2)+len(slide_1)]

    rows, cols = linear_sum_assignment(cdist)
    scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
    matching_table = pd.DataFrame(data={'immune':slide_2_all_sample['CellID'], 'tumor': [slide_1_all['CellID'].iloc[c] for c in cols], 'score':scores})
    imtbl = ids_to_table(matching_table['immune'],slide_2_all_sample)
    tmtbl = ids_to_table(matching_table['tumor'],slide_1_all)
    sla_all = np.mean(get_correlations(shared_markers, imtbl,tmtbl))
    with open(os.path.join(save_path,f'la_all_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_all, handle)
print('Done')