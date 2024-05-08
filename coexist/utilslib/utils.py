from skimage.measure import regionprops_table
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import os

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

def ids_to_table(cellIDs, table, key):
    """
    Description:
        - DataFrame from list of cellIDs and original cell-feature table
    Parameters:
        - cellIDs: uint list, cellIDs
        - table: pd.DataFrame, original cell-feature DataFrame that cellIDs are a subset of
    Return:
        - new_table: pd.DataFrame, ordered cell-feature table of cellIDs
    """
    new_table = table[table[key].isin(cellIDs)] # get relevant rows
    df1 = new_table.set_index(key)
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

def normalize_marker_signals(panel_A: pd.DataFrame, 
                             panel_B: pd.DataFrame, 
                             panel_A_tracked: pd.DataFrame, 
                             panel_B_tracked: pd.DataFrame, 
                             ignore_cols: list = [],
                             remove_outliers: bool = True,
                             log_transform: bool = True):

    from . import process_data
    """
    the dataframes (panel_A and panel_B) must include both tracked cells (within panel_A_tracked and panel_B_tracked) and untracked cells.
    We will perform marker normalization using the all the cells in each panel, then return all dataframes with normalized marker intensities.
    Additionally, we include the option to remove outlier cells from each panel.
    Dataframes must be passed exactly as how they should be processed (ind)
    """
    print("    Number of cells/feature in first panel before normalization = {}/{}".format(panel_A.shape[0], panel_A.shape[1]))
    init_panel_A_index = panel_A.index
    panel_A = process_data.normalize_dataframe(panel_A.copy(), remove_outliers = remove_outliers, log_transform = log_transform, ignore_cols = ignore_cols)
    print("    Number of tumor cells/features in first panel after normalization = {}/{}".format(panel_A.shape[0], panel_A.shape[1]))

    print("    Number of cells/feature in second panel before normalization = {}/{}".format(panel_B.shape[0], panel_B.shape[1]))
    init_panel_B_index = panel_B.index
    panel_B = process_data.normalize_dataframe(panel_B.copy(), remove_outliers = remove_outliers, log_transform = log_transform, ignore_cols = ignore_cols)
    print("    Number of tumor cells/features in first panel after normalization = {}/{}".format(panel_B.shape[0], panel_B.shape[1]))

    #determine which indeces were removed, if any.
    if remove_outliers:
        rm_from_panel_A = init_panel_A_index.difference(panel_A.index)
        rm_from_panel_B = init_panel_B_index.difference(panel_B.index)
        bad_inds_panel_A = panel_A_tracked.index.isin(rm_from_panel_A)
        bad_inds_panel_B = panel_B_tracked.index.isin(rm_from_panel_B)
        bad_inds = np.logical_or(bad_inds_panel_A, bad_inds_panel_B) #drop any rows that are True.
        print("    First panel tracked # cells / # features before removal of outliers: {}/{}".format(panel_A_tracked.shape[0], panel_A_tracked.shape[1]))
        panel_A_tracked.drop(panel_A_tracked.index[bad_inds].tolist(),inplace=True)
        print("    First panel tracked # cells / # features after removal of outliers: {}/{}".format(panel_A_tracked.shape[0], panel_A_tracked.shape[1]))
        print("    Second panel tracked # cells / # features before removal of outliers: {}/{}".format(panel_B_tracked.shape[0], panel_A_tracked.shape[1]))
        panel_B_tracked.drop(panel_B_tracked.index[bad_inds].tolist(),inplace=True)
        print("    Second panel tracked # cells / # features after removal of outliers: {}/{}".format(panel_B_tracked.shape[0], panel_A_tracked.shape[1]))
    
    #Replace the unnormalized tracked data with the normalized data from each individual panel
    panel_A_tracked = panel_A.loc[panel_A_tracked.index].copy()
    panel_B_tracked = panel_B.loc[panel_B_tracked.index].copy()

    return panel_A, panel_B, panel_A_tracked, panel_B_tracked

def gpu_cluster(dataframe: pd.DataFrame, n_neighbors: int = 100, ignore_cols: list = [], cluster_name: str = "cluster"):
    """
    Description:
        - Cluster cells in dataframe using GPU-accelerated Grapheno
    Parameters:
        - dataframe: pd.DataFrame, dataframe which to perform clustering on
        - n_neighbors: int, grapheno clustering parameter 
        - ignore_cols: list, list of column name strings to exclude during clustering
        - cluster_name: str, string for new column name in dataframe designating cluster label
    Returns:
        - dataframe: pd.DataFrame, the same input dataframe but with additional column labeled as argument cluster_name.
    """
    print("    Importing Grapheno libraries...")
    from . import grapheno_ce
    import cudf

    print("    Beginning clustering...")
    dataframe = cudf.from_pandas(dataframe)
    out_csv_path = os.path.join(os.getcwd(), "All_tracked_temp.csv")
    dataframe.to_csv(out_csv_path, index=False) #write out a temporary csv file, necessary for grapheno.
    #use grapheno to cluster 
    dataframe = grapheno_ce.cluster(out_csv_path, 
                features = [x for x in dataframe.columns.tolist() if x not in ignore_cols],
                n_neighbors=n_neighbors,
                new_column_name = cluster_name,
                overwrite = True, # we do not want it to load any cached parquet files.
                write_out_parquet = False, #default behavior will be to not write out the parquet file after clustering.
                min_size=int(np.round(dataframe.shape[0]*0.01))) #min size should be 1% of the dataset, i.e. the "rare cell" population size
    #convert back to pandas 
    dataframe = dataframe.to_pandas()
    #labels = dataframe[cluster_name]
    #delete the temporary file we saved
    os.remove(out_csv_path)
    return dataframe
    
def propagate_labels(dataframe: pd.DataFrame, tracked_labels: pd.Series, n_neighbors = 10, ignore_cols = []):
    """
    Description:
        - We will propagate the super-plexed labels from cells that were matched across multiple panels to the cells that were not matched across multiple panels using
          a GPU accelerated KNN classifier algorithm.
    Parameters:
        - dataframe: pd.DataFrame, dataframe containing both matched and unmatched cells across panels. Index key should match to tracked_labels.
        - tracked_labels: pd.Series, subset of dataframe corresponding to the labels of cells matched across panels. Will be used to propogate labels to the remainder of the dataframe.  Index key should match to dataframe.
        - n_neighbors: int, grapheno clustering parameter 
        - ignore_cols: list, list of column name strings to exclude during clustering
        - cluster_name: str, string for new column name in dataframe designating cluster label
        NOTE: dataframe and tracked_labels should be indexed by the same key (i.e. model.cellID_key)
    Returns:
        - labels: pd.Series, contains the propogated labels 
    """
    assert dataframe.index.name == tracked_labels.index.name, "dataframe and tracked_labels index names must match! dataframe index has name {}, tracked_labels index has name {}".format(dataframe.index.name,tracked_labels.index.name)

    from cuml.neighbors import KNeighborsClassifier as cuKNeighbors

    keep_cols = [x for x in dataframe.columns.tolist() if x not in ignore_cols]

    #split the dataframe into the tracked and untracked subpopulations
    df1_tracked = dataframe.loc[tracked_labels.index][keep_cols] # is this correct
    df1_untracked = dataframe.loc[~dataframe.index.isin(tracked_labels.index)][keep_cols]
    #fit KNN classifier to the tracked population
    print("    Fitting KNN model to dataframe.")
    model = cuKNeighbors(n_neighbors=n_neighbors)
    model.fit(df1_tracked, tracked_labels)
    #for the untracked population, determine their KNN of the tracked population and propogate the labels.
    print("    predicting")
    y_hat = model.predict(df1_untracked)
    labels = pd.Series(np.concatenate([tracked_labels.values.flatten(), y_hat]), index = np.concatenate([tracked_labels.index.values.flatten(), df1_untracked.index.values.flatten()])).sort_index()
    return labels

def run_UMAP(dataframe, cluster: pd.Series = None, super_cluster: pd.Series = None, ignore_cols: list = []):
    """
    Description:
        - Perform GPU accelerated UMAP embedding of dataframe. Will concatenate (correctly) to the cluster and super cluster labels the UMAP embedding.
    Parameters:
        - dataframe: pd.DataFrame, dataframe used as clustering input
        - cluster: pd.Series containing as index key a column within dataframe (typically model.cellID_key) whose values are the cluster labels.
        - super_cluster: pd.Series containing as index key a column within dataframe (typically model.cellID_key) whose values are the super cluster labels (i.e. the multiplexed labels)
        - ignore_cols: list of strings containing column names to ignore during UMAP embedding
    Returns:
        - umap_coords: pd.DataFrame, dataframe containing x-y coordinates UMAP1, UMAP2 corresponding to UMAP embedding of dataframe
    """
    if cluster is not None:
        assert cluster.index.name in dataframe.columns.values, "'{}' is not a column of the provided dataframe!"
    if super_cluster is not None:
        assert super_cluster.index.name in dataframe.columns.values, "'{}' is not a column of the provided dataframe!"

    import time
    import cuml

    np.random.seed(0) #set a random seed for reproducibility
    start = time.time()
    umap = cuml.UMAP(n_neighbors=20)
    drop_list = [x for x in ignore_cols if x in dataframe.columns.to_list()] #filter out any in ignore_cols that are not actually in the dataframe column names
    umap_embed = umap.fit_transform(dataframe.drop(drop_list,axis=1))#.rename(columns={0:'UMAP1',1:'UMAP2'})
    umap_coords = pd.DataFrame(data = umap_embed, columns = ["UMAP1","UMAP2"])
    if cluster is not None:
        #add cluster labels
        good_inds = dataframe[cluster.index.name].isin(cluster.index.values)
        umap_coords.loc[good_inds.values, "cluster"] = cluster.loc[dataframe.loc[good_inds, cluster.index.name].values].values
    if super_cluster is not None:
        #add super cluster labels
        good_inds = dataframe[super_cluster.index.name].isin(super_cluster.index.values)
        umap_coords.loc[good_inds.values, "super_cluster"] = super_cluster.loc[dataframe.loc[good_inds, super_cluster.index.name].values].values
    end = time.time()
    print("Total UMAP training time = {}".format(end-start))
    return umap_coords


def visualize_UMAP(umap_coords: pd.DataFrame, dataframe: pd.DataFrame, color_by:str = 'cluster', color_by_is_cat:bool = True):
    """
    Description:
        - Generate graphic of a UMAP embedding with cluster data
    Parameters:
        - umap_coords: pd.DataFrame, dataframe output from run_UMAP
        - dataframe: pd.DataFrame, dataframe used as input to run_UMAP
        - color_by: str, column name of data contained in argument dataframe or metadata which will be used to set the color scheme of the plot
        - color_by_is_cat: bool, if color_by argument is categorical data type or not!
    Returns:
        - Holoviews display
    """
    assert color_by in umap_coords.columns.values.tolist() + dataframe.reset_index().columns.values.tolist(), "{} is not a valid column name in either the provided dataframe!".format(color_by)
    
    from scipy import stats
    from holoviews.operation.datashader import datashade
    import holoviews as hv
    from holoviews import opts
    from colorcet import fire, glasbey_hv
    import datashader as ds
    hv.extension('bokeh') # specify which library for plotting, e.g. 'bokeh' or 'matplotlib'
    from holoviews.operation.datashader import dynspread
    dynspread.max_px=2
    dynspread.threshold=1.0

    ####REMOVE OUTLIERS FOR A BETTER PLOT####
    keep_inds = (np.abs(stats.zscore(umap_coords[['UMAP1','UMAP2']])) < 5).all(axis=1)
    umap_coords = umap_coords[keep_inds]
    if color_by not in ['cluster', 'super_cluster']:
        umap_coords = pd.concat([umap_coords,dataframe.reset_index()[keep_inds][color_by]], axis=1)
    
    #########################################

    #########################################

    if color_by_is_cat:
        assert color_by in umap_coords.columns.values.tolist(), '"{}" not in umap_coords columns!'.format(color_by)
        #drop any rows with Nans, which results from removing outliers within the normalization typically.
        umap_coords = umap_coords.dropna(axis=0)

        color_key = ['#%02x%02x%02x' % tuple(int(255*j) for j in glasbey_hv[i]) for i in range(len(umap_coords[color_by].unique()))]
        cluster_dta = umap_coords[['UMAP1','UMAP2',color_by]].groupby(color_by).mean()
        labels = hv.Labels({('UMAP1', 'UMAP2'): cluster_dta.to_numpy(), 'text': cluster_dta.index.tolist()}, ['UMAP1', 'UMAP2'], 'text')
        
        umap_ds = dynspread(datashade(hv.Scatter(umap_coords[['UMAP1', 'UMAP2', color_by]]).opts(size=20),
                            aggregator=ds.count_cat(color_by),
                            color_key=color_key).opts(padding=0.1,
                                                        xaxis=None,
                                                        yaxis=None,
                                                        width=700,
                                                        height=700,
                                                        legend_position='right'))
        return umap_ds*labels

    else:
        #continuous data
        umap_ds = dynspread(datashade(hv.Scatter(umap_coords[['UMAP1', 'UMAP2', color_by]]).opts(size=10),
                    aggregator=ds.mean(color_by),
                    cmap = fire).opts(padding=0.1,
                            xaxis=None,
                            yaxis=None,
                            width=700,
                            height=700))

        return umap_ds

def generate_heatmap(dataframe: pd.DataFrame, cluster: pd.Series = None, super_cluster: pd.Series = None, color_by:str = 'cluster', ignore_cols = []):
    """
    Description:
        - Generate clustermap of a dataframe to visualize marker expression leading to cluster assignments.
    Parameters:
        - dataframe: pd.DataFrame, dataframe (index key of cluster or super_cluster should be a column within dataframe - i.e. model.cellID_key)
        - color_by: str, grouping strategy for clustermap - opts: group by 'cluster' if cluster is provided, 'super_cluster' if super_cluster is provided.
        - ignore_cols: list of strings containing column names within dataframe to ignore during hierarchical clustering in clustermap
        - cluster: pd.Series containing as index key a column within dataframe (typically model.cellID_key) whose values are the cluster labels.
        - super_cluster: pd.Series containing as index key a column within dataframe (typically model.cellID_key) whose values are the super cluster labels (i.e. the multiplexed labels)
        NOTE: cluster and/or super_cluster must be provided as an argument in addition to dataframe.
    Returns:
        Seaborn Clustermap
    """
    #validate arguments
    assert color_by in ['cluster', 'super_cluster'], "argument 'color_by' must be in {}, got {}".format(['cluster', 'super_cluster'], color_by)
    assert (cluster is not None) or (super_cluster is not None), "'cluster' and/or 'super_cluster' must be provided as dataseries. No arguments were given."
    if cluster is not None:
        assert cluster.index.name in dataframe.columns.values, "'{}' is not a column of the provided dataframe!"
    if super_cluster is not None:
        assert super_cluster.index.name in dataframe.columns.values, "'{}' is not a column of the provided dataframe!"
    
    import seaborn as sns 

    if cluster is not None:
        #add cluster labels
        good_inds = dataframe[cluster.index.name].isin(cluster.index.values)
        dataframe = dataframe[good_inds].set_index(cluster.index.name)
        dataframe['cluster'] = cluster
        dataframe = dataframe.reset_index()

    if super_cluster is not None:
        #add super cluster labels
        good_inds = dataframe[super_cluster.index.name].isin(super_cluster.index.values)
        dataframe = dataframe[good_inds].set_index(super_cluster.index.name)
        dataframe['super_cluster'] = super_cluster
        dataframe = dataframe.reset_index()
    
    #we want to remove added columns from being part of the hierarchical clustering within clustermap.
    if 'super_cluster' not in ignore_cols:
        ignore_cols = ignore_cols + ['super_cluster']
    if 'cluster' not in ignore_cols:
        ignore_cols = ignore_cols + ['cluster']

    drop_list = [x for x in ignore_cols if x != color_by]
    drop_list = [x for x in drop_list if x in dataframe.columns.to_list()]

    #create the clustermap
    g = sns.clustermap(dataframe.drop(drop_list, axis=1).groupby(color_by).mean().drop(-1.0,errors='ignore'),
                   z_score=1,
                   cmap='Spectral_r',
                   method='ward',
                   center=0,
                   cbar_kws={'label': 'z-score'},
                   figsize=(12,20))
