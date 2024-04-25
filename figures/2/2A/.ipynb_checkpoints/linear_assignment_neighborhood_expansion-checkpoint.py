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

# marker names for slide 1, slide 2, and their overlap
slide_2_markers = ['DNA_1','CD3', 'PDL1', 'GRZB','Ki67', 'PanCK', 'CD45','CD68', 'CD3d', 'CD8a',
             'CD163', 'aSMA', 'CD14','CD4', 'FOXP3', 'PDL1_2','CD11b', 'pRB', 'PD1',
             'LaminABC', 'PDL1_3', 'LAG3','CD20', 'HLA_A', 'MHC_II_DPB1'] # CyCIF immune panel
slide_1_markers = ['DNA_1','CD3', 'pERK', 'Rad51','CCND1', 'Vimentin', 'aSMA','Ecad', 'ER', 'PR',
            'EGFR', 'pRB', 'HER2','Ki67', 'CD45', 'p21','CK14','CK19', 'CK17',
            'LaminABC', 'AR', 'H2Ax','PCNA','PanCK', 'CD31'] # CyCIF tumor panel
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']

radii = [i for i in range(501)]
### Spatial linear assignment w/ tracking as anchors:
# read tCyCIF data
save_path = 'coexist/figures/2/2A/Neighborhood_expansion_results/'
cores = ['B3']
for core in tqdm(cores):
    
    tracking = pd.read_csv(f'coexist/figures/2/2A/Cell_tracking_results/{core}.csv') # path to tracking results
    tracking = tracking[tracking['tumor_id']>0]
    
    #all cells tables
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/'
    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    #normalize tracked cells based on full population
    slide_2_all_norm = slide_2_all[slide_2_markers].copy()
    slide_2_all_norm = zscore(slide_2_all_norm)
    slide_2_all_norm['CellID'] = slide_2_all['CellID'].copy()
    slide_2_tracked_norm = ids_to_table(tracking['immune_id'], slide_2_all_norm)
    
    # load matrices
    slide_2 = slide_2_tracked_norm[shared_markers].copy()
    slide_1 = slide_1_all[shared_markers].copy()
    
    slide_1 = zscore(slide_1,axis=0)
    sla_track = []

    cdist = 1 - spearmanr(a=slide_2.to_numpy(),
                     b=slide_1.to_numpy(), 
                     axis=1)[0][0:len(slide_2),len(slide_2):len(slide_2)+len(slide_1)]
    
    print('Starting spatial linear assignment with tracked anchors')
    for r in tqdm(radii):
        cols = ['x','y']
        slide_1_spatial = slide_1_all[cols].to_numpy()
        sdist = pairwise_distances(slide_1_spatial, slide_1_spatial)
        
        matched_indexes = [slide_1_all[slide_1_all['CellID']==i].index for i in list(tracking['tumor_id'])]
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
        matching = [list(tracking['immune_id']), [slide_1_all['CellID'].iloc[c] for c in cols], scores]
        imtbl = ids_to_table(matching[0],slide_2_all)
        tmtbl = ids_to_table(matching[1],slide_1_all)
        sla_track.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
    with open(os.path.join(save_path,f'sla_track_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_track, handle)
    print('Done')
    
    ### Spatial linear assignment with random N cells
    print('Starting spatial linear assignment with random anchors')
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    
    #normalize tracked cells based on full population
    slide_2_all_norm = slide_2_all[slide_2_markers].copy()
    slide_2_all_norm = zscore(slide_2_all_norm)
    slide_2_all_norm['CellID'] = slide_2_all['CellID'].copy()
    slide_2_all_norm['x'] = slide_2_all['x'].copy()
    slide_2_all_norm['y'] = slide_2_all['y'].copy()
    n = len(tracking)
    seeds = []
    for k in range(5):
        sample = list(slide_2_all_norm['CellID'].sample(n))
        slide_2_all_sample = ids_to_table(sample, slide_2_all_norm)
        slide_2_all_sample['CellID'] = slide_2_all_sample.index
        # load matrices
        slide_2 = slide_2_all_sample[shared_markers].copy()
        
        slide_2 = slide_2_all_sample[shared_markers].copy()
        slide_1 = slide_1_all[shared_markers].copy()
        
        slide_1 = zscore(slide_1,axis=0)
        sla_random = []
        #spearman correlation distance
        cdist = 1 - spearmanr(a=slide_2.to_numpy(),
                 b=slide_1.to_numpy(), 
                 axis=1)[0][0:len(slide_2),len(slide_2):len(slide_2)+len(slide_1)]
        for r in tqdm(radii):
            cols = ['x','y']
            slide_1_spatial = slide_1_all[cols].to_numpy()
            slide_2_spatial = slide_2_all_sample[cols].to_numpy()
            sdist = pairwise_distances(slide_2_spatial, slide_1_spatial)
            radius = r
            dist = np.where(sdist<=radius, cdist.copy(), 10000000)
            rows, cols = linear_sum_assignment(dist)
            scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
            matching_table = pd.DataFrame(data={'immune':slide_2_all_sample['CellID'], 'tumor': [slide_1_all['CellID'].iloc[c] for c in cols], 'score':scores})
            imtbl = ids_to_table(matching_table['immune'],slide_2_all_sample)
            tmtbl = ids_to_table(matching_table['tumor'],slide_1_all)
            sla_random.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
        seeds.append(sla_random)
    with open(os.path.join(save_path,f'sla_random_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(seeds, handle)
    print('Done')

    ### Spatial linear assignment w/ all cells
    print('Starting spatial linear assignment with no anchors')
    path = '/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/' # path to feature tables
    slide_1_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_tumor_both.csv'))
    slide_2_all = pd.read_csv(os.path.join(path,f'{core}_tCyCIF_immune_both.csv'))
    
    slide_2_all_sample = slide_2_all.copy()
    # load matrices
    
    slide_2 = slide_2_all_sample[shared_markers].copy()
    slide_1 = slide_1_all[shared_markers].copy()
    
    slide_2 = zscore(slide_2,axis=0)
    slide_1 = zscore(slide_1,axis=0)
    cdist = 1 - spearmanr(a=slide_2.to_numpy(),
                     b=slide_1.to_numpy(), 
                     axis=1)[0][0:len(slide_2),len(slide_2):len(slide_2)+len(slide_1)]
    sla_all = []
    for r in tqdm(radii):
        cols = ['x','y']
        slide_1_spatial = slide_1_all[cols].to_numpy()
        slide_2_spatial = slide_2_all_sample[cols].to_numpy()
        sdist = pairwise_distances(slide_2_spatial, slide_1_spatial)
        dist = np.where(sdist<=r, cdist.copy(), 10000000)
        rows, cols = linear_sum_assignment(dist)
        scores = np.array([cdist[i, j] for i, j in zip(rows, cols)])
        matching_table = pd.DataFrame(data={'immune':slide_2_all_sample['CellID'], 'tumor': [slide_1_all['CellID'].iloc[c] for c in cols], 'score':scores})
        imtbl = ids_to_table(matching_table['immune'],slide_2_all_sample)
        tmtbl = ids_to_table(matching_table['tumor'],slide_1_all)
        sla_all.append(np.mean(get_correlations(shared_markers, imtbl,tmtbl)))
    with open(os.path.join(save_path,f'sla_all_spearman_{core}.pkl'),'wb') as handle:
        pickle.dump(sla_all, handle)
    print('Done')