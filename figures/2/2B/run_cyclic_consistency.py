import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from tqdm import tqdm

# marker names for immune, tumor, and their overlap
slide_2_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1']
slide_1_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31']
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']

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

radii = [i for i in range(200)]
results = {}
cores = ['B3']
for core in cores:
    # read tCyCIF data
    tracked = pd.read_csv(f'coexist/notebooks/figures/2/2A/Cell_tracking_results/{core}.csv')
    tracked = tracked[tracked['tumor_id']>0]
    #all cells tables
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables'
    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))

    slide_2_tracked = ids_to_table(tracked['immune_id'], slide_2_all)
    slide_1_tracked = ids_to_table(tracked['tumor_id'], slide_1_all)
    overlap = []
    correlations = []
    #normalize tracked cells based on full population
    slide_2_all_norm = slide_2_all[slide_2_markers].copy()
    slide_2_all_norm = zscore(slide_2_all_norm,axis=0)
    slide_2_all_norm['CellID'] = slide_2_all['CellID'].copy()
    slide_2_tracked_norm = ids_to_table(tracked['immune_id'], slide_2_all_norm)
    
    slide_2 = slide_2_tracked_norm[shared_markers].copy()
    slide_1 = slide_1_all[shared_markers].copy()
    
    slide_1 = zscore(slide_1,axis=0)
    #spearman correlation distance
    slide_2_cdist = 1 - spearmanr(a=slide_2,
                     b=slide_1, 
                     axis=1)[0][0:len(slide_2),len(slide_2):len(slide_2)+len(slide_1)]

    #normalize tracked cells based on full population
    slide_1_all_norm = slide_1_all[slide_1_markers].copy()
    slide_1_all_norm = zscore(slide_1_all_norm,axis=0)
    slide_1_all_norm['CellID'] = slide_1_all['CellID'].copy()
    slide_1_tracked_norm = ids_to_table(tracked['tumor_id'], slide_1_all_norm)

    # load matrices
    slide_1 = slide_1_tracked_norm[shared_markers].copy()
    slide_2 = slide_2_all[shared_markers].copy()
    slide_2 = zscore(slide_2,axis=0)
    slide_1_cdist = 1 - spearmanr(a=slide_1,
                     b=slide_2, 
                     axis=1)[0][0:len(slide_1),len(slide_1):len(slide_2)+len(slide_1)]

    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))

    slide_2_tracked = ids_to_table(tracked['immune_id'], slide_2_all)
    slide_1_tracked = ids_to_table(tracked['tumor_id'], slide_1_all)
    
    for r in tqdm(radii):
        #normalize tracked cells based on full population
        slide_2_all_norm = slide_2_all[slide_2_markers].copy()
        slide_2_all_norm = zscore(slide_2_all_norm)
        slide_2_all_norm['CellID'] = slide_2_all['CellID'].copy()
        slide_2_tracked_norm = ids_to_table(tracked['immune_id'], slide_2_all_norm)
        
        slide_2 = slide_2_tracked_norm[shared_markers].copy()
        slide_1 = slide_1_all[shared_markers].copy()

        slide_1 = zscore(slide_1,axis=0)
    
        #euclidean distance between target cells
        cols = ['x','y']
        slide_1_spatial = slide_1_all[cols].to_numpy()
        sdist = pairwise_distances(slide_1_spatial, slide_1_spatial)
    
        matched_indexes = [slide_1_all[slide_1_all['CellID']==i].index for i in list(tracked['tumor_id'])]
        dist = slide_2_cdist.copy()
        
        for i,match in enumerate(matched_indexes):
            # get target indices within radius r from sdist
            row = sdist[match,:].copy()
            indexes = list(np.where(row<=r))
            # replace values in cdist
            mask = np.ones(dist[i,:].size, dtype=bool)
            mask[indexes] = False
            dist[i,:][mask] = np.inf
    
        rows, cols = linear_sum_assignment(dist)
        scores = np.array([dist[i, j] for i, j in zip(rows, cols)])
        matching = [rows, cols, scores]
    
        #convert immune index to CellID
        matching_slide_2 = [tracked['immune_id'], [slide_1_all['CellID'].iloc[c] for c in cols], scores]
        
        #normalize tracked cells based on full population
        slide_1_all_norm = slide_1_all[slide_1_markers].copy()
        slide_1_all_norm = zscore(slide_1_all_norm)
        slide_1_all_norm['CellID'] = slide_1_all['CellID'].copy()
        slide_1_tracked_norm = ids_to_table(tracked['tumor_id'], slide_1_all_norm)
    
        # load matrices
        slide_1 = slide_1_tracked_norm[shared_markers].copy()
        slide_2 = slide_2_all[shared_markers].copy()
        
        slide_2 = zscore(slide_2,axis=0)
        
        #euclidean distance between target cells
        cols = ['x','y']
        slide_2_spatial = slide_2_all[cols].to_numpy()
        sdist = pairwise_distances(slide_2_spatial, slide_2_spatial)
    
        matched_indexes = [slide_2_all[slide_2_all['CellID']==i].index for i in list(tracked['immune_id'])]
        dist = slide_1_cdist.copy()
        
        for i,match in enumerate(matched_indexes):
            # get target indices within radius r from sdist
            row = sdist[match,:].copy()
            indexes = list(np.where(row<=r))
            # replace values in cdist
            mask = np.ones(dist[i,:].size, dtype=bool)
            mask[indexes] = False
            dist[i,:][mask] = np.inf
        
        rows, cols = linear_sum_assignment(dist)
        scores = np.array([dist[i, j] for i, j in zip(rows, cols)])
        matching = [rows, cols, scores]

        matching_slide_1 = [[slide_2_all['CellID'].iloc[c] for c in cols], tracked['tumor_id'], scores]
        
        # create sets
        slide_2Set = set((x,y) for x,y in zip(matching_slide_2[0], matching_slide_2[1]))
        slide_1Set = set((x,y) for x,y in zip(matching_slide_1[0], matching_slide_1[1]))
        intersection = slide_2Set.intersection(slide_1Set)
        slide_2Unique = slide_2Set.difference(slide_1Set)
        slide_1Unique = slide_1Set.difference(slide_2Set)
    
        slide_1_both = []
        slide_2_both = []
        for m in intersection:
            slide_1_both.append(m[1])
            slide_2_both.append(m[0])
    
        slide_1_table = ids_to_table(slide_1_both, slide_1_all)
        slide_2_table = ids_to_table(slide_2_both, slide_2_all)
        correlations.append(np.mean(get_correlations(shared_markers,slide_2_table,tm_table)))
        overlap.append(len(intersection)/len(slide_2_tracked))
    results[core]=[overlap,correlations]

with open('cyclic.pkl','wb') as handle:
    pickle.dump(results, handle)
