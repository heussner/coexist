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

# marker names for immune, tumor, and their overlap
immune_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1']
tumor_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31']
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']

radii = [i for i in range(501)]
### Spatial linear assignment w/ tracking as anchors:
# read tCyCIF data
save_path = '/home/groups/ChangLab/heussner/coexist/figures/2/2A/Radius_exploration/'
cores = ['B3']
for core in tqdm(cores):
    
    tracking = pd.read_csv(f'/home/groups/ChangLab/heussner/coexist/figures/2/2A/Cell_tracking/{core}.csv') # path to tracking results
    tracking = tracking[tracking['tumor_id']>0]
    
    #all cells tables
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/'
    tumor_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    immune_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    #normalize tracked cells based on full population
    immune_all_norm = immune_all[immune_markers].copy()
    immune_all_norm = zscore(immune_all_norm)
    immune_all_norm['CellID'] = immune_all['CellID'].copy()
    immune_tracked_norm = ids_to_table(tracking['immune_id'], immune_all_norm)
    
    # load matrices
    
    immune = immune_tracked_norm[shared_markers].copy()
    tumor = tumor_all[shared_markers].copy()
    
    tumor = zscore(tumor,axis=0)
    sla_track = []

    cdist = 1 - spearmanr(a=immune.to_numpy(),
                     b=tumor.to_numpy(), 
                     axis=1)[0][0:len(immune),len(immune):len(immune)+len(tumor)]
    
    print('Starting spatial linear assignment with tracked anchors')
    for r in tqdm(radii):
        cols = ['x','y']
        tumor_spatial = tumor_all[cols].to_numpy()
        sdist = pairwise_distances(tumor_spatial, tumor_spatial)
        
        matched_indexes = [tumor_all[tumor_all['CellID']==i].index for i in list(tracking['tumor_id'])]
        dist = cdist.copy()
        for i,match in enumerate(matched_indexes):
            # get target indices within radius r from sdist
            row = sdist[match,:].copy()
            indexes = list(np.where(row<=r))
            # replace values in cdist
            mask = np.ones(dist[i,:].size, dtype=bool)
            mask[indexes] = False
            dist[i,:][mask] = np.inf
        
        rows, cols = linear_sum_assignment(dist)
        scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
        matching = [list(tracking['immune_id']), [tumor_all['CellID'].iloc[c] for c in cols], scores]
        imtbl = ids_to_table(matching[0],immune_all)
        tmtbl = ids_to_table(matching[1],tumor_all)
        sla_track.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
    with open(os.path.join(save_path,f'sla_track_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_track, handle)
    print('Done')
    
    ### Spatial linear assignment with random N cells
    print('Starting spatial linear assignment with random anchors')
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    tumor_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    immune_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    
    #normalize tracked cells based on full population
    immune_all_norm = immune_all[immune_markers].copy()
    immune_all_norm = zscore(immune_all_norm)
    immune_all_norm['CellID'] = immune_all['CellID'].copy()
    immune_all_norm['x'] = immune_all['x'].copy()
    immune_all_norm['y'] = immune_all['y'].copy()
    n = len(tracking)
    seeds = []
    for k in range(5):
        sample = list(immune_all_norm['CellID'].sample(n))
        immune_all_sample = ids_to_table(sample, immune_all_norm)
        immune_all_sample['CellID'] = immune_all_sample.index
        # load matrices
        immune = immune_all_sample[shared_markers].copy()
        
        immune = immune_all_sample[shared_markers].copy()
        tumor = tumor_all[shared_markers].copy()
        
        tumor = zscore(tumor,axis=0)
        sla_random = []
        #spearman correlation distance
        cdist = 1 - spearmanr(a=immune.to_numpy(),
                 b=tumor.to_numpy(), 
                 axis=1)[0][0:len(immune),len(immune):len(immune)+len(tumor)]
        for r in tqdm(radii):
            cols = ['x','y']
            tumor_spatial = tumor_all[cols].to_numpy()
            immune_spatial = immune_all_sample[cols].to_numpy()
            sdist = pairwise_distances(immune_spatial, tumor_spatial)
            radius = r
            dist = np.where(sdist<=radius, cdist.copy(), 10000000)
            rows, cols = linear_sum_assignment(dist)
            scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
            matching_table = pd.DataFrame(data={'immune':immune_all_sample['CellID'], 'tumor': [tumor_all['CellID'].iloc[c] for c in cols], 'score':scores})
            imtbl = ids_to_table(matching_table['immune'],immune_all_sample)
            tmtbl = ids_to_table(matching_table['tumor'],tumor_all)
            sla_random.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
        seeds.append(sla_random)
    with open(os.path.join(save_path,f'sla_random_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(seeds, handle)
    print('Done')

    ### Spatial linear assignment w/ all cells
    print('Starting spatial linear assignment with no anchors')
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    tumor_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    immune_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    
    immune_all_sample = immune_all.copy()
    # load matrices
    
    immune = immune_all_sample[shared_markers].copy()
    tumor = tumor_all[shared_markers].copy()
    
    immune = zscore(immune,axis=0)
    tumor = zscore(tumor,axis=0)
    cdist = 1 - spearmanr(a=immune.to_numpy(),
                     b=tumor.to_numpy(), 
                     axis=1)[0][0:len(immune),len(immune):len(immune)+len(tumor)]
    sla_all = []
    for r in tqdm(radii):
        cols = ['x','y']
        tumor_spatial = tumor_all[cols].to_numpy()
        immune_spatial = immune_all_sample[cols].to_numpy()
        sdist = pairwise_distances(immune_spatial, tumor_spatial)
        dist = np.where(sdist<=r, cdist.copy(), 10000000)
        rows, cols = linear_sum_assignment(dist)
        scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
        matching_table = pd.DataFrame(data={'immune':immune_all_sample['CellID'], 'tumor': [tumor_all['CellID'].iloc[c] for c in cols], 'score':scores})
        imtbl = ids_to_table(matching_table['immune'],immune_all_sample)
        tmtbl = ids_to_table(matching_table['tumor'],tumor_all)
        sla_all.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
    with open(os.path.join(save_path,f'sla_all_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_all, handle)
    print('Done')