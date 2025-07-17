#!/usr/bin/env python

import os
import sys
import re 
import argparse
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from cuml.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


def get_args():

    parser = argparse.ArgumentParser(add_help=True,
        description='Simulates KNN label propagation on synthetic reduced marker panels')

    parser.add_argument('-f', '--featuretable',
        help='CSV feature table with marker expression, X and Y centroids, cell types', 
        required=True)
    
    parser.add_argument('-o', '--outpath',
        help='Path to output CSV',
        required=True)

    parser.add_argument('--markers',
        help='CSV file with markers to use with column label: marker_name',
        required=True)

    parser.add_argument('--randompanels',
        help='Number of random panels to draw',
        required=False,
        default=1000)

    parser.add_argument('--low',
        help='Lower bound for anchor proportions (inclusive)',
        required=False,
        default=0.1)

    parser.add_argument('--high',
        help='Higher bound for anchor proportions (Exclusive)',
        required=False,
        default=1.0)

    parser.add_argument('--step',
        help='Step for anchor proportions',
        required=False,
        default=0.1)

    parser.add_argument('--subsample',
        help='Value by which to sample the untracked cell predictions for f1 calculation',
        required=False,
        default=5000)  # 40000 used for CRC

    return parser.parse_args()


def subset_panel_prediction(
        tracked,
        untracked,
        tracked_labs,
        untracked_labs,
        panel,
        subsample_predictions=5000):

    '''
    Subset tracked and untracked cell dataframes to a small panel of markers,
    train a KNN classifier on the tracked cells/reduced markers to predict groundtruth cell type labels.
    Predict cell type labels on untracked cells, and compute f1 score against groundtruth labels

    tracked:               dataframe of tracked cells x full marker panel
    untracked:             dataframe of untracked cells x full marker panel
    tracked_labs:          list of groundtruth cell type labels for the tracked cells only
    untracked_labs:        dataframe with CellID as index and groundtruth labels
    panel:                 list of markers to subset full marker panel by
    subsample_predictions: integer of how many random cells to draw from untracked predictions for f1 calculation
    '''

    # subset to panel dataframes for tracked and untracked
    tracked = tracked[panel]
    untracked = untracked[panel]

    # train and fit KNN on tracked cells using groundtruth labels
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(tracked, tracked_labs)

    # Use KNN to predict labels on untracked cells
    untracked['knn_derived_labels'] = list(model.predict(untracked))

    # bring in original labels for comparison
    untracked = pd.merge(untracked, untracked_labs, left_index=True, right_index=True)

    # randomly sample from the untracked cells for final score calculation
    untracked_sample = untracked.sample(n=subsample_predictions, replace=False, random_state=30, axis=0)

    # finally calculate f1 score
    score = f1_score(untracked_sample['cell_type'],untracked_sample['knn_derived_labels'], average='weighted')

    return score


def main():

    # input data
    args = get_args()
    df = pd.read_csv(args.featuretable)
    markers = np.array(pd.read_csv(args.markers)['marker_name'])

    # parameters
    anchor_cell_proportions = np.arange(float(args.low), float(args.high), float(args.step))
    panel_size = len(markers) // 2

    # randomly generate panels
    panels = []
    for i in range(int(args.randompanels)):
        np.random.seed(i)
        panels.append(np.random.choice(markers, panel_size, replace=False))

    # initialize results dict
    results = {
        'panel' : [],
        'proportion' : [],
        'f1_score' : []
    }

    # main loop of anchor proportions and panels
    for a in anchor_cell_proportions:

        print(f'Running with anchor proportion of {round(a, 1)}')

        # randomly choose anchor cells
        anchors = np.random.choice(df['CellID'], size=int(a*len(df)), replace=False)

        # pull out tracked groundtruth labels for training knn
        tracked_df = df[df['CellID'].isin(anchors)]
        tracked_df = tracked_df.set_index('CellID')
        tracked_labs = tracked_df['cell_type']

        # prep the untracked (test) dataframe
        untracked_df = df[~df['CellID'].isin(anchors)]
        untracked_df = untracked_df.set_index('CellID')
        untracked_labs = untracked_df['cell_type']

        for i,p in enumerate(panels):

            score = subset_panel_prediction(
                tracked=tracked_df,
                untracked=untracked_df,
                tracked_labs=tracked_labs,
                untracked_labs=untracked_labs,
                panel=p,
                # 5000 for tonsil, 40000 for crc
                subsample_predictions=int(args.subsample)
            )

            results['panel'].append(i)
            results['proportion'].append(a)
            results['f1_score'].append(score)

    # create a dataframe out of current results
    results_df = pd.DataFrame(results)

    # reset results dict
    results = {
        'panel' : [],
        'proportion' : [],
        'f1_score' : []
    }

    # now just running simulation with all anchor cells
    a = 1.0
    print('Running final all anchors simulation')

    # pull out tracked groundtruth labels for training knn 
    tracked_df = df.copy()
    tracked_df = tracked_df.set_index('CellID')
    tracked_labs = tracked_df['cell_type']

    # prep the untracked (test) dataframe 
    untracked_df = df.copy()
    untracked_df = untracked_df.set_index('CellID')
    untracked_labs = untracked_df['cell_type']

    for i,p in enumerate(panels):

        score = subset_panel_prediction(
            tracked=tracked_df,
            untracked=untracked_df,
            tracked_labs=tracked_labs,
            untracked_labs=untracked_labs,
            panel=p,
            subsample_predictions=40000
        )

        results['panel'].append(i)
        results['proportion'].append(a)
        results['f1_score'].append(score)

    # create a dataframe from 100% anchors run and concat with other results
    all_anchors_df = pd.DataFrame(results)
    final_results = pd.concat([results_df, all_anchors_df], axis=0)

    final_results.to_csv(args.outpath)


if __name__ == "__main__":

    sys.exit(main())