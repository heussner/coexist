from skimage.measure import regionprops_table
import pandas as pd

def match_cells(im1_mask, im2_mask, pairwise_distances):
    """
    Description:
    -Match target cells to reference cells
    
    Parameters:
    -im1_mask: uint array, Cell mask for target image
    -im2_mask: uint array, Cell mask for reference image
    -pairwise_distances: float array, Target x reference pairwise correlations
    
    Return:
    -match_table: pd.DataFrame, target cell IDs tracked to the ordered list of reference cell IDs
    """
    props = ['label','coords','centroid','mean_intensity']
    im1_frame = pd.DataFrame(regionprops_table(im1_mask, im1_mask, properties=props)) # target regionprops
    im2_frame = pd.DataFrame(regionprops_table(im2_mask, im2_mask, properties=props)) # reference regionprops
    
    cellID = np.zeros(len(im2_frame)) # cell ID map from reference to target
    scores = np.ones(len(im2_frame))*2
    for i in range(len(im2_frame)): # for each region in reference
    
        coords = im2_frame['coords'][i].squeeze()
        x = im1_mask[tuple(coords.T)] # get target where reference region is
    
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
                cdist1[j] = pairwise_distances[i, t_frame[t_frame['label'] == ux[j]].index]
                odist1[j] = 2*(1-n[j]/area)
                edist1[j] = ((im2_frame['centroid-0'].iloc[i] - im1_frame[im1_frame['label'] == ux[j]]['centroid-0'].values[0])**2 + (im2_frame['centroid-1'].iloc[i] - im1_frame[im1_frame['label'] == ux[j]]['centroid-1'].values[0])**2)**.5
            
            if len(cdist1[np.where(cdist1==min(cdist1))]) > 1:
                cellID[i] = ux[edist1==min(edist1[np.where(cdist1==min(cdist1))])]
                scores[i] = cdist1[np.where(ux==cellID[i])]

            else:
                cellID[i] = ux[np.where(cdist1==min(cdist1))]
                scores[i] = cdist1[np.where(ux==cellID[i])]
        else:
            cellID[i] = 0
    
        # checking cyclic consistency
        if cellID[i] != 0:
            idx = im1_frame['coords'][im1_frame['label'] == cellID[i]].squeeze() # get target coords in reference cell id
            y = im2_mask[tuple(idx.T)].copy() # get reference array where target cell is
            uy, m = np.unique(y, return_counts=True)
            area2 = np.sum(m)
            if uy[0] == 0: # remove background pixels from unique list
                uy = uy[1:]
                m = m[1:]
            
            odist2 = np.ones(uy.shape)
            cdist2 = np.ones(uy.shape)
            edist2 = np.zeros(uy.shape)
            for j in range(len(uy)):
                cdist2[j] = pairwise_distances[im2_frame[im2_frame['label'] == uy[j]].index,im1_frame[im1_frame['label'] == cellID[i]].index][0]
                odist2[j] = 2*(1-m[j]/area2)
                edist2[j] =((im2_frame[im2_frame['label']==uy[j]]['centroid-0'].values[0]-im1_frame[im1_frame['label']==cellID[i]]['centroid-0'].values[0])**2 + (im2_frame[im2_frame['label']==uy[j]]['centroid-1'].values[0]-im1_frame[im1_frame['label']==cellID[i]]['centroid-1'].values[0])**2)**0.5
            
            if len(cdist2[np.where(cdist2==min(cdist2))]) > 1:
                if uy[edist2==min(edist2[np.where(cdist2==min(cdist2))])] != im2_frame["label"][i]:
                    cellID[i] = 0
                    scores[i] = 2

            if (len(cdist2[np.where(cdist2==min(cdist2))]) == 1) and (uy[np.where(cdist2==min(cdist2))] != im2_frame["label"][i]):
                cellID[i] = 0
                scores[i] = 2
    
    match_table = pd.DataFrame(data={'im1_cell':cellID,'im2_cell':list(im2_frame['label']),'score':scores})
    match_table = match_table[match_table['im1_cell']!=0]
    return match_table

def track_cells(im1_mask, im2_mask):
    """
    Description:
    -Track target cells to reference cells
    
    Parameters:
    -im1_mask: uint array, Cell mask for target image
    -im2_mask: uint array, Cell mask for reference image
    
    Return:
    -match_table: list, target cell IDs tracked to the ordered list of reference cell IDs
    """
    props = ['label','coords','centroid','mean_intensity']
    vote_threshold=overlap
    im1_frame = pd.DataFrame(regionprops_table(im1_mask, im1_mask, properties=props)) # target regionprops
    im2_frame = pd.DataFrame(regionprops_table(im2_mask, im2_mask, properties=props)) # reference regionprops

    cellID = np.zeros(len(im2_frame)) # cell ID map from reference to target

    for i in range(len(im2_frame)): # for each region in reference

        coords = im2_frame['coords'][i].squeeze()
        x = im1_mask[tuple(coords.T)] # get target where reference region is

        ux, n = np.unique(x, return_counts=True) # get counts of unique pixel values of target in reference region
        area = np.sum(n)

        if ux[0] == 0: # remove background pixels from unique list
            ux = ux[1:]
            n = n[1:]

        if (len(n) != 0) and (max(n)/area > vote_threshold):#if the largest cell in target is above pixel threshold

            if len(n[n==max(n)]) > 1:# case where multiple equally-sized target cells in reference region
                id_ = ux[n==max(n)] # cellID list
                dist1 = np.zeros(id_.shape)
                for j in range(len(id_)): # iterate cells
                    xdist = im2_frame["centroid-0"][i]- im1_frame["centroid-0"][im1_frame['label']==id_[j]].squeeze()
                    ydist = im2_frame["centroid-1"][i]- im1_frame["centroid-1"][im1_frame['label']==id_[j]].squeeze()
                    dist1[j] = (xdist**2 + ydist**2)**0.5
                    
                cellID[i] = id_[np.where(dist1==min(dist1))]

                # case where cells are equally close to centroid

            else:
                cellID[i] = ux[np.where(n==max(n))]

        else:
            cellID[i] = 0

        #checking cyclic consistency for uniqueness
        if cellID[i] != 0:
            idx = im1_frame['coords'][im1_frame['label'] == cellID[i]].squeeze() # get target coords in reference cell id
            y = im2_mask[tuple(idx.T)].copy() # get reference array where target cell is
            uy, m = np.unique(y, return_counts=True)
            
            if uy[0] == 0: # remove background pixels from unique list
                uy = uy[1:]
                m = m[1:]
            
            if (len(m[m==max(m)]) > 1):
                id_ = uy[m==max(m)]
                dist2 = np.ones(id_.shape)*100000
                for j in range(len(id_)):
                    x_dist = im1_frame["centroid-0"][im1_frame['label']==cellID[i]].squeeze() - im2_frame["centroid-0"][im2_frame['label']==id_[j]].squeeze()
                    y_dist = im1_frame["centroid-1"][im1_frame['label']==cellID[i]].squeeze() - im2_frame["centroid-1"][im2_frame['label']==id_[j]].squeeze()
                    dist2[j] = (x_dist**2 + y_dist**2)**0.5

                if uy[np.where(dist2==min(dist2))] != im2_frame["label"][i]:
                    cellID[i] = 0

            elif (len(m[m == max(m)]) == 1) and (uy[m==max(m)] != im2_frame["label"][i]):
                cellID[i] = 0
    match_table = pd.DataFrame(data={'im1_cell':cellID,'im2_cell':list(im2_frame['label'])})
    match_table = match_table[match_table['im1_cell']!=0]
    return match_table

def ids_to_table(ids, table):
    new_table = table[table['CellID'].isin(ids)] # get relevant rows
    df1 = new_table.set_index('CellID')
    new_table = df1.reindex(ids) # set new table in correct order
    new_table.reset_index(inplace=True)
    return new_table