from skimage.measure import regionprops_table
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np


def match(im1_mask, im2_mask, cost_matrix):
    """
    Description:
        Match cells across adjacent MTIs
    Parameters:
        - im1_mask: uint array, cell segmentation mask for reference image
        - im2_mask: uint array, cell segmentation mask for target image
        - cost_matrix: float array, reference x target pairwise Spearman correlations
    Return:
        - match_table: pd.DataFrame, match cellID pairs with match scores
    """
    props = ['label','coords','centroid','mean_intensity']
    im1_frame = pd.DataFrame(regionprops_table(im1_mask, im1_mask, properties=props)) # reference regionprops
    im2_frame = pd.DataFrame(regionprops_table(im2_mask, im2_mask, properties=props)) # target regionprops
    cellID = np.zeros(len(im1_frame)) # cell ID map from reference to target
    scores = np.ones(len(im1_frame))*2
    for i in range(len(im1_frame)): # for each region in reference
        coords = im1_frame['coords'][i].squeeze()
        x = im2_mask[tuple(coords.T)] # get target where reference region is
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
                cdist1[j] = cost_matrix[i, im2_frame[im2_frame['label'] == ux[j]].index]
                odist1[j] = 2*(1-n[j]/area)
                edist1[j] = ((im1_frame['centroid-0'].iloc[i] - im2_frame[im2_frame['label'] == ux[j]]['centroid-0'].values[0])**2 + (im1_frame['centroid-1'].iloc[i] - im2_frame[im2_frame['label'] == ux[j]]['centroid-1'].values[0])**2)**.5
            if len(cdist1[np.where(cdist1==min(cdist1))]) > 1:
                cellID[i] = ux[edist1==min(edist1[np.where(cdist1==min(cdist1))])]
                scores[i] = cdist1[np.where(ux==cellID[i])]
            else:
                cellID[i] = ux[np.where(cdist1==min(cdist1))]
                scores[i] = cdist1[np.where(ux==cellID[i])]
        else:
            cellID[i] = 0 
        if cellID[i] != 0: # checking cyclic consistency
            idx = im2_frame['coords'][im2_frame['label'] == cellID[i]].squeeze() # get target coords in reference cell id
            y = im1_mask[tuple(idx.T)].copy() # get reference array where target cell is
            uy, m = np.unique(y, return_counts=True)
            area2 = np.sum(m)
            if uy[0] == 0: # remove background pixels from unique list
                uy = uy[1:]
                m = m[1:]
            odist2 = np.ones(uy.shape)
            cdist2 = np.ones(uy.shape)
            edist2 = np.zeros(uy.shape)
            for j in range(len(uy)):
                cdist2[j] = cost_matrix[im1_frame[im1_frame['label'] == uy[j]].index,im2_frame[im2_frame['label'] == cellID[i]].index][0]
                odist2[j] = 2*(1-m[j]/area2)
                edist2[j] =((im1_frame[im1_frame['label']==uy[j]]['centroid-0'].values[0]-im2_frame[im2_frame['label']==cellID[i]]['centroid-0'].values[0])**2 + (im1_frame[im1_frame['label']==uy[j]]['centroid-1'].values[0]-im2_frame[im2_frame['label']==cellID[i]]['centroid-1'].values[0])**2)**0.5
            if len(cdist2[np.where(cdist2==min(cdist2))]) > 1:
                if uy[edist2==min(edist2[np.where(cdist2==min(cdist2))])] != im1_frame["label"][i]:
                    cellID[i] = 0
                    scores[i] = 2
            if (len(cdist2[np.where(cdist2==min(cdist2))]) == 1) and (uy[np.where(cdist2==min(cdist2))] != im1_frame["label"][i]):
                cellID[i] = 0
                scores[i] = 2
    match_table = pd.DataFrame(data={'im1_cellID':list(im1_frame['label']),'im2_cellID':cellID.astype(int),'score':scores})
    match_table = match_table[match_table['im2_cellID']!=0]
    return match_table

def track(im1_mask, im2_mask, cost_matrix=None):
    """
    Description:
        Track cells across adjacent MTIs
    Parameters:
        - im1_mask: uint array, cell segmentation mask for reference image
        - im2_mask: uint array, cell segmentation mask for target image
        - cost_matrix: float array, reference x target pairwise Spearman correlations
    Returns:
        - match_table: pd.DataFrame, match cellID pairs with match scores 
    """
    props = ['label','coords','centroid','mean_intensity']
    im1_frame = pd.DataFrame(regionprops_table(im1_mask, im1_mask, properties=props)) # target regionprops
    im2_frame = pd.DataFrame(regionprops_table(im2_mask, im2_mask, properties=props)) # reference regionprops
    cellID = np.zeros(len(im1_frame)) # cell ID map from reference to target
    for i in range(len(im1_frame)): # for each region in reference
        coords = im1_frame['coords'][i].squeeze()
        x = im2_mask[tuple(coords.T)] # get target where reference region is
        ux, n = np.unique(x, return_counts=True) # get counts of unique pixel values of target in reference region
        area = np.sum(n)
        if ux[0] == 0: # remove background pixels from unique list
            ux = ux[1:]
            n = n[1:]
        if (len(n) != 0):# if the largest cell in target is above pixel threshold
            if len(n[n==max(n)]) > 1:# case where multiple equally-sized target cells in reference region
                id_ = ux[n==max(n)] # cellID list
                dist1 = np.zeros(id_.shape)
                for j in range(len(id_)): # iterate cells
                    xdist = im1_frame["centroid-0"][i]- im2_frame["centroid-0"][im2_frame['label']==id_[j]].squeeze()
                    ydist = im1_frame["centroid-1"][i]- im2_frame["centroid-1"][im2_frame['label']==id_[j]].squeeze()
                    dist1[j] = (xdist**2 + ydist**2)**0.5         
                cellID[i] = id_[np.where(dist1==min(dist1))]
            else:
                cellID[i] = ux[np.where(n==max(n))]
        else:
            cellID[i] = 0
        if cellID[i] != 0: # checking cyclic consistency for uniqueness
            idx = im2_frame['coords'][im2_frame['label'] == cellID[i]].squeeze() # get target coords in reference cell id
            y = im1_mask[tuple(idx.T)].copy() # get reference array where target cell is
            uy, m = np.unique(y, return_counts=True)
            if uy[0] == 0: # remove background pixels from unique list
                uy = uy[1:]
                m = m[1:]
            if (len(m[m==max(m)]) > 1):
                id_ = uy[m==max(m)]
                dist2 = np.ones(id_.shape)*100000
                for j in range(len(id_)):
                    x_dist = im2_frame["centroid-0"][im2_frame['label']==cellID[i]].squeeze() - im1_frame["centroid-0"][im1_frame['label']==id_[j]].squeeze()
                    y_dist = im2_frame["centroid-1"][im2_frame['label']==cellID[i]].squeeze() - im1_frame["centroid-1"][im1_frame['label']==id_[j]].squeeze()
                    dist2[j] = (x_dist**2 + y_dist**2)**0.5
                if uy[np.where(dist2==min(dist2))] != im1_frame["label"][i]:
                    cellID[i] = 0
            elif (len(m[m == max(m)]) == 1) and (uy[m==max(m)] != im1_frame["label"][i]):
                cellID[i] = 0
    if cost_matrix:
        scores  = [cost_matrix[i,im2_frame['label'].iloc[cellID[i]]] for i in range(len(cellID))]
    else:
        scores = np.zeros(len(cellID))
    match_table = pd.DataFrame(data={'im1_cellID':list(im1_frame['label']),'im2_cellID':cellID, 'score':scores})
    match_table = match_table[match_table['im2_cellID']!=0]
    return match_table

def linear_assignment(cellID1, cellID2, cost_matrix, distance_matrix, max_distance):
    """
    Description:
        - Run linear sum assignment to match cells in adjacent MTIs
    Parameters:
        - cellID1: list, cellID list for cell segmentation mask 1
        - cellID2: list, cellID list for cell segmentation mask 2
        - cost_matrix: float array, reference x target pairwise Spearman correlations
        - distance_matrix: float array, reference x target pairwise Euclidean distances
        - max_distance: neighborhood radius to consider match candidates
    Returns:
        - match_table: pd.DataFrame, match cellID pairs with match scores
    """
    if len(cellID1)>len(cellID2):
        cost_matrix = cost_matrix.T
        distance_matrix = distance_matrix.T
        input1 = cellID2
        input2 = cellID1
    else:
        input1 = cellID1
        input2 = cellID2
    cost = np.where(distance_matrix<=max_distance, cost_matrix, 10000000)
    rows, cols = linear_sum_assignment(cost)
    scores = np.array([cost[i, j] for i, j in zip(rows, cols)])
    if input1 == cellID1:
        match_table = pd.DataFrame(data={'im1_cellID':cellID1, 'im2_cellID': [cellID2[c] for c in cols], 'score':scores})
    else:
        match_table = pd.DataFrame(data={'im1_cellID':[cellID1[c] for c in cols], 'im2_cellID': cellID2, 'score':scores})
    return match_table

def ids_to_table(cellIDs, table):
    """
    Description:
        - DataFrame from list of cellIDs and original cell-feature table
    Parameters:
        - cellIDs: uint list, cellIDs
        - table: pd.DataFrame, original cell-feature DataFrame that cellIDs are a subset of
    Return:
        - new_table: pd.DataFrame, ordered cell-feature table of cellIDs
    """
    new_table = table[table['cellID'].isin(cellIDs)] # get relevant rows
    df1 = new_table.set_index('cellID')
    new_table = df1.reindex(cellIDs) # set new table in correct order
    new_table.reset_index(inplace=True)
    return new_table

def simulate_overlap(iter_, mu, sigma, thickness):
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
        - f_shared: float, fraction of section 1 cells shared in section 2
    """
    n_shared = 0
    n_unshared = 0
    xlims = [0 - (mu+3*sigma), thickness + (mu+3*sigma)]
    for i in range(iter_):
        r = np.random.normal(mu, sigma)/2
        x = np.random.uniform(low=xlims[0],high=xlims[1])
        min_ = x-r
        max_ = x+r
        if (min_ < thickness < max_):  # shared case
            n_shared += 1
        elif (0 < max_ < thickness): # unshared case
            n_unshared +=1
        else:
            continue
    f_shared = n_shared/(n_shared+n_unshared)
    return f_shared

def kNN(table):
    return graph

def cluster(table):
    return labels
    
def propagate_labels(graph, table):
    return labels
