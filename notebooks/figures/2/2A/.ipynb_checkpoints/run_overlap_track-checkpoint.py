import numpy as np
from tifffile import imread
import os
from skimage.io import imshow
from skimage.measure import regionprops_table
import pandas as pd
from numpy.linalg import norm
from scipy.stats import zscore, spearmanr
from tqdm import tqdm

def track(t_img, r_img, corr):
    """
    Description:
    -Track target cells to reference cells
    
    Parameters:
    -t_img: uint array, Cell mask for target image
    -r_img: uint array, Cell mask for reference image
    -corr: float array, Target x reference pairwise correlations
    
    Return:
    -cellID: list, target cell IDs tracked to the ordered list of reference cell IDs
    """
    props = ['label','coords','centroid','mean_intensity']
    t_frame = pd.DataFrame(regionprops_table(t_img, t_img, properties=props)) # target regionprops
    r_frame = pd.DataFrame(regionprops_table(r_img, r_img, properties=props)) # reference regionprops
    
    cellID = np.zeros(len(r_frame)) # cell ID map from reference to target
    scores = np.ones(len(r_frame))*2
    for i in range(len(r_frame)): # for each region in reference
    
        coords = r_frame['coords'][i].squeeze()
        x = t_img[tuple(coords.T)] # get target where reference region is
    
        ux, n = np.unique(x, return_counts=True) # get counts of unique pixel values of target in reference region
        area = np.sum(n)
        
        if ux[0] == 0: # remove background pixels from unique list
            ux = ux[1:]
            n = n[1:]
        if (len(n) != 0):
            odist1 = np.zeros(ux.shape)
            cdist1 = np.zeros(ux.shape)
            edist1 = np.zeros(ux.shape)
            
            for j in range(len(ux)): # if there are cells present
                cdist1[j] = corr[i, t_frame[t_frame['label'] == ux[j]].index]
                odist1[j] = 2*(1-n[j]/area)
                edist1[j] = ((r_frame['centroid-0'].iloc[i] - t_frame[t_frame['label'] == ux[j]]['centroid-0'].values[0])**2 + (r_frame['centroid-1'].iloc[i] - t_frame[t_frame['label'] == ux[j]]['centroid-1'].values[0])**2)**.5
            
            if len(cdist1[np.where(odist1==min(odist1))]) > 1:
                if len(ux[cdist1==min(cdist1[np.where(odist1==min(odist1))])])>1:
                    cellID[i] = ux[edist1==min(edist1[cdist1==min(cdist1[np.where(odist1==min(odist1))])])]
                    scores[i] = min(cdist1[np.where(ux==cellID[i])])
                else:
                    cellID[i] = ux[cdist1==min(cdist1[np.where(odist1==min(odist1))])]
                    scores[i] = min(cdist1[np.where(odist1==min(odist1))])
            else:
                cellID[i] = ux[np.where(odist1==min(odist1))]
                scores[i] = cdist1[np.where(odist1==min(odist1))]
        else:
            cellID[i] = 0
    
        #checking cyclic consistency
        if cellID[i] != 0:
            idx = t_frame['coords'][t_frame['label'] == cellID[i]].squeeze() # get target coords in reference cell id
            y = r_img[tuple(idx.T)].copy() # get reference array where target cell is
            uy, m = np.unique(y, return_counts=True)
            area2 = np.sum(m)
            if uy[0] == 0: # remove background pixels from unique list
                uy = uy[1:]
                m = m[1:]
            
            odist2 = np.ones(uy.shape)
            cdist2 = np.ones(uy.shape)
            edist2 = np.zeros(uy.shape)
            for j in range(len(uy)):
                cdist2[j] = corr[r_frame[r_frame['label'] == uy[j]].index,t_frame[t_frame['label'] == cellID[i]].index][0]
                odist2[j] = 2*(1-m[j]/area2)
                edist2[j] =((r_frame[r_frame['label']==uy[j]]['centroid-0'].values[0]-t_frame[t_frame['label']==cellID[i]]['centroid-0'].values[0])**2 + (r_frame[r_frame['label']==uy[j]]['centroid-1'].values[0]-t_frame[t_frame['label']==cellID[i]]['centroid-1'].values[0])**2)**0.5
            
            if len(cdist2[np.where(odist2==min(odist2))]) > 1:
                if len(uy[cdist2==min(cdist2[np.where(odist2==min(odist2))])])>1:
                    if uy[edist2==min(edist2[cdist2==min(cdist2[np.where(odist2==min(odist2))])])] != r_frame["label"][i]:
                        cellID[i] = 0
                        scores[i] = 2
                elif uy[cdist2==min(cdist2[np.where(odist2==min(odist2))])] != r_frame["label"][i]:
                    cellID[i] = 0
                    scores[i] = 2
            if (len(cdist2[np.where(odist2==min(odist2))]) == 1) and (uy[np.where(odist2==min(odist2))] != r_frame["label"][i]):
                cellID[i] = 0
                scores[i] = 2
    return cellID, scores

cores = ['A10','A5','A6','A8','B10','B1','B3','B4',
         'B6','B7','B9','C10','C1','C4','D11','D2','D8',
         'E10','E3','E5','E6','E7','E8','E9','F11','F1','F2','F7',
         'G11','G1','G3','G6','G7','G9','H10','H1',
         'H2','H3','H4','H6','H7','H8','H9']

target_path = '/home/groups/ChangLab/dataset/TMA_cell_track/result-Nuclear/'
reference_path = '/home/groups/OMSAtlas/share/TMA_TNP_masks/TMA1_005/refined_masks/'
shared_markers = ['DNA_1','CD3','aSMA','pRB','PanCK','CD45','Ki67','LaminABC']

for c in tqdm(cores): #iterate cores
    try:
        slide_1_table = pd.read_csv(f'/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/{c}_tCyCIF_tumor_both.csv') # path to feature tables
        slide_2_table = pd.read_csv(f'/home/groups/ChangLab/heussner/tma-integration/data/nuc_cell_tables/{c}_tCyCIF_immune_both.csv')
        dist = 1 - spearmanr(a=zscore(slide_2_table[shared_markers].to_numpy(),axis=0),
                     b=zscore(slide_1_table[shared_markers].to_numpy(),axis=0), 
                     axis=1)[0][0:len(slide_2_table),len(slide_2_table):len(slide_2_table)+len(slide_1_table)]
        slide_1 = imread(os.path.join(target_path,f'TMA_{c}_reg_img_s1.tif')) # path to masks
        slide_2 = imread(os.path.join(reference_path,f'OHSU_TMA1_005-{c}_refinedMask_Nuclear.tiff'))

        # remove cells not in tables
        table_1_ids = list(slide_1_table['CellID'])
        slide_1_ids = np.unique(slide_1)[1:]
        missing = list(set(slide_1_ids) - set(table_1_ids))
        for m in missing:
            slide_1[slide_1 == m] = 0

        table_2_ids = list(slide_2_table['CellID'])
        slide_2_ids = np.unique(slide_2)[1:]
        missing = list(set(slide_2_ids) - set(table_2_ids))
        for m in missing:
            slide_2[slide_2 == m] = 0

        #get cellID maps
        slide_1_map, scores = track(slide_1,slide_2,dist)
        
        table = pd.DataFrame(data={'immune_id':list(slide_2_table['CellID']),'tumor_id':slide_1_map, 'score':scores})
        table.to_csv(f'/home/groups/ChangLab/heussner/coexist/figures/2/2A/Cell_tracking/{c}.csv',index=False)
    except Exception as e: print(f'{c}: ' + str(e))