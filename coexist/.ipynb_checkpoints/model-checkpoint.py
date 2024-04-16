import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr
from sklearn.metrics import pairwise_distances
from . import utils

class COEXIST:
    """
    Main object
    """
    def __init__(self, 
                 im1_mask, im2_mask, 
                 df1, df2,
                 cellID_key,
                 shared_markers=None,
                 method='correlation', iter=100000, thickness=None, mpp1=None, mpp2=None, max_dist=None, diameter_key=None
    ):
        """
        Description:
            Initialization
        Parameters:
            - im1_mask: int array, MTI 1 cell segmentation mask
            - im2_mask: int array, MTI 2 cell segmentation mask
            - df1: pd.DataFrame, cell-feature DataFrame for MTI 1, should have column for cell labels, 'x','y' for cell centroids if using the 'linear assignment' method, and diameter (pixels) if simulating sharing between adjacent sections
            - df2: pd.DataFrame, cell-feature DataFrame for MTI 2, should have column for cell labels, 'x','y' for cell centroids if using the 'linear assignment' method, and diameter (pixels) if simulating sharing between adjacent sections
            - shared_markers: list, list of shared marker keys for df1/df2
            - cellID_key: str, key for cell labels in df1/df2
            - method: str, matching algorithm: 'correlation', 'tracking', or 'linear assignment'
            - thickness: float, thickness of tissue section (um)
            - iter: int, number of simulation repitition
            - mpp: float, microns per pixel of MTIs
            - max_dist: float, neighborhood size (pixels) if choosing 'linear assignment' method
            - diameter_key: str, key for cell/nucleus diameter in df1/df2
        """
        self.im1_mask = im1_mask
        self.im2_mask = im2_mask
        self.df1 = df1
        self.df2 = df2
        self.cellID_key = cellID_key
        self.shared_markers = shared_markers
        assert method in {'correlation', 'overlap', 'linear assignment'}
        self.method = method
        if self.method != 'tracking':
            assert (self.shared_markers is not None)
            if len(self.shared_markers) < 4:
                print("Too few shared markers. Switching method to 'tracking'")
                self.method = 'tracking'
        self.iter = iter
        self.thickness = thickness
        self.mpp1 = mpp1
        self.mpp2 = mpp2
        self.max_dist = max_dist
        self.diameter_key = diameter_key
    
        self.arr1_shared = None
        self.arr2_shared = None
        self.df1_matched = None
        self.df2_matched = None
        self.cost_matrix = None
        self.df1_count = len(self.df1)
        self.df2_count = len(self.df2)

    def preprocess_data(self):
        """
        Description:
            Get shared cell-feature arrays, standardize DataFrames, compute necessary matrices
        """
        if (self.method == 'linear assignment') | (self.method == 'correlation'):
            arr1_shared = self.df1[self.shared_markers]
            arr2_shared = self.df2[self.shared_markers]
            self.arr1_shared = np.array(zscore(arr1_shared, axis=0))
            self.arr2_shared = np.array(zscore(arr2_shared, axis=0))
            print('Computing cost matrix')
            self.cost_matrix = 1 - spearmanr(self.arr1_shared,self.arr2_shared,axis=1)[0][0:self.df1_count,self.df1_count:self.df1_count+self.df2_count]
        if self.method == 'linear assignment':
            print('Computing distance matrix')
            self.distance_matrix = pairwise_distances(self.df1[['x','y']], self.df2[['x','y']])
        print('Done preprocessing')
        return None

    def graph_data(self):
        """
        Description:
            Make kNN graphs for MTI 1 and 2
        """
        print('Making kNN graphs')
        self.arr1_graph = None
        self.arr2_graph = None
    
    def match_cells(self):
        """
        Description:
            Perform intregration across MTI 1 and 2
        """
        print('Matching cells')
        if self.method == 'overlap':
            match_table = utils.track(self.im1_mask, self.im2_mask, cost_matrix=self.cost_matrix)
        elif self.method == 'correlation':
            match_table = utils.match(self.im1_mask, self.im2_mask, self.cost_matrix)
        elif self.method == 'linear assignment':
            match_table = utils.linear_assignment(list(self.df1[self.cellID_key]), list(self.df2[self.cellID_key]), self.cost_matrix, self.distance_matrix, self.max_dist)
        print('Removing low quality matches')
        match_table = match_table[match_table['score']<1]
        print(f'Matched {len(match_table)} cells of {self.df1_count} slide 1 cells and {self.df2_count} slide 2 cells,  {np.round(len(match_table)*200/(self.df1_count+self.df2_count),1)}% shared.')
        self.match_book = match_table
        self.df1_matched = utils.ids_to_table(match_table['im1_cellID'], self.df1)
        self.df2_matched = utils.ids_to_table(match_table['im2_cellID'], self.df2)
        return None

    def check_correlations(self):
        """
        Description:
            Check shared marker Spearman correlations of the matched cells
        """
        for marker in self.shared_markers:
            corr = spearmanr(self.df1_matched[marker],self.df2_matched[marker])
            print(f'{marker}: {np.round(corr[0],2)}')
        return None

    def cluster_cells(self):
        """
        Description:
            Cluster matched cells and propagate labels to the unmatched cells
        """
        print('Clustering matched cells')
        self.matched_labels = None
        print('Propagating labels to unmatched cells')
        self.arr1_labels = None
        self.arr2_labels = None
        return None

    def estimate_overlap(self):
        """
        Description:
            Estimate the fraction of shared cells across MTI 1 and 2
        """
        print('Estimating adjacent section overlap')
        assert self.diameter_key is not None
        assert self.mpp is not None
        assert self.thickness is not None
        fraction_shared = utils.simulate_overlap(self.iter,
                                                 np.mean(self.df1[self.diameter_key])*self.mpp1,
                                                 np.std(self.df1[self.diameter_key])*self.mpp1,
                                                 self.thickness)
        print(f'Approximately {np.round(fraction_shared*100,1)}% of cells overlap')
        return None
      
