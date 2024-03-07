import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr
from scipy.optimize import linear_sum_assignment
from . import utils



class Match:
  """
  Main object
  """
  def __init__(
    self, im1_mask, im2_mask, shared_markers, cellID_label, arr1, arr2, method='correlation'
  ):
    """
    Initialization for Match object.
  
    :param arr1: First dataset with shared features.
    :type arr1: pd.DataFrame
    :param im1_markers: Second dataset with shared features.
    :type im1_markers: List
    :param arr2: First dataset with shared features.
    :type arr2: pd.DataFrame
    :param im2_markers: Second dataset with shared features.
    :type im2_markers: List
    :param method: 'correlation', 'overlap', or 'linear assignment'.
        controlling how the matching is done.
    :type method: str
    """
    # input
    self.im1_mask = im1_mask
    self.im2_mask = im2_mask
    self.arr1 = shared_arr1
    self.arr2 = shared_arr2
    self.shared_markers = shared_markers
    assert method in {'correlation', 'overlap', 'linear_assignment'}
    self.method = method

    self.im1_cells = None
    self.im2_cells = None
    self.arr1_shared = None
    self.arr2_shared = None
    self.arr1_matched = None
    self.arr2_matched = None

    self.im1_cellIDs = list(self.arr1[self.cellID_label])
    self.im2_cellIDs = list(self.arr2[self.cellID_label])
    
    if self.shared_markers:
      arr1_shared = self.arr1[self.shared_markers]
      arr2_shared = self.arr2[self.shared_markers]
      self.arr1_shared = np.array(zscore(arr1_shared, axis=0))
      self.arr2_shared = np.array(zscore(arr2_shared, axis=0))
      self.pairwise_distances = 1 - spearmanr(self.arr1_shared, self.arr2_shared)[0][]

    def graph(self):
      print('Making kNN graphs')
      self.arr1_graph = None
      self.arr2_graph = None
    
    def match(self):
      print('Matching cells...')
      if self.method == 'correlation':
        match_table = utils.match_cells(self.im1_mask, self.im2_mask, self.pairwise_distances)
      elif self.method == 'overlap':
        match_table = utils.track_cells(self.im1_mask, self.im2_mask)
      print('Removing low quality matches...')
      match_table = match_table[match_table['score']<0]
      print(f'Matched {len(match_table)} cells of {len(self.arr1)} slide 1 cells and {len(self.arr2)} slide 2 cells')
      self.arr1_matched = utils.ids_to_table(match_table['slide_1_cells'], self.arr1)
      self.arr2_matched = utils.ids_to_table(match_table['slide_2_cells'], self.arr2)
      return None

    def cluster(self):
      print('Clustering matched cells...')
      self.matched_labels = None
      print('Propagating labels to unmatched cells...')
      self.arr1_labels = None
      self.arr2_labels = None
      return None
      
