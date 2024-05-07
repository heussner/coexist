import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import minmax_scale, robust_scale 
import matplotlib.pyplot as plt


def normalize_dataframe(data: pd.DataFrame, remove_outliers = True, log_transform = False, ignore_cols = []):
    start = time.time()
    ### Ignore all non-floats (categorical) or arguments provided by ignore_cols when normalizing. 
    ignore_cols = [data.columns[i] for i,x in enumerate(data.iloc[0].values.flatten()) if not is_float(str(x))] + [x for x in ignore_cols if x in data.columns] #the str method is used to deal with the nan's and Nones
    ignore_cols = list(set(ignore_cols)) #the two lists may repeat one of the elements, considering you can pass whatever list in bad_cols
    good_cols = [x for x in data.columns.tolist() if np.all([y not in x for y in ignore_cols])]

    #we'll ignore the ignore_cols, and then place them into the dataframe at the end.
    exclude_list = [x for x in ignore_cols if x in data.columns.to_list()]
    exclude_data = data[exclude_list]
    data.drop(exclude_list, axis=1, inplace=True)
    
    # # Consider log transforming the data, if the distributions appear log-normal.
    if log_transform:
        data = (1.+data).apply(np.log)

    if remove_outliers:
        #we actually just want to remove rows that are outside of the 99.9th percentile. This is what they do in phenograph.
        q_dta = data.quantile(q=0.999,axis=0)
        for (col, val) in zip(q_dta.index.to_list(),list(q_dta.values)):
            #also remove the rows from exlude data, if necessary.
            if exclude_data.shape[1]>0:
                exclude_data = exclude_data.loc[data[col]<=val]
            data=data.loc[data[col]<=val]
        
    # #implement robust scaling 
    data = data.apply(robust_scale)

    # # deskew
    data = (data/5).apply(np.arcsinh) #while this is primarily linear in the range of -1 to 1, beyond 1 and -1 it significantly plateus, similar to a log transform.

    ## scale
    data = data.apply(minmax_scale)

    #place back in the ignored columns!
    if exclude_data.shape[1]>0:
        data[exclude_list] = exclude_data
                                                                
    end = time.time()

    print("    total preprocessing time = {} minutes".format((end-start)/60))

    return data


"""
Some notes:
the returned dataframe has an extra column ('Sample') that was not previously included. We will 
want to exclude it PRIOR to doing UMAP embedding.
Just use this:
df.drop(['Sample'], axis=1) # note that in_place is not there, so it just temporarily does the operation.
"""

def visualize_hists(dta, col_name=None, save_fig=False, figname=None, plot_mean=False, exclude_list = []):
    #exclude_list = ['Sample','Label', 'MinX', 'MinY', 'MaxX', 'MaxY', 'CentroidX', 'CentroidY','To Exclude', 'core']
    exclude_list = [x for x in exclude_list if x in dta.columns.to_list()]
    if figname is None:
        figname = "figure.pdf"
        
    if col_name is not None:
        ax = dta.hist(col_name,bins=100)
        #get means
        if plot_mean:
            for r in range(ax.shape[0]):
                for c in range(ax.shape[1]):
                    ttl = ax[r,c].get_title()
                    if ttl!="":
                        ax[r,c].axvline(dta[ttl].mean(), color='k', linestyle='dashed', linewidth=2)
        if save_fig:
            fig = plt.gcf()
            fig.savefig(figname)
    else:
        ax = dta.drop(exclude_list,axis=1).hist(bins=30, figsize=(45, 30))
        #get means
        if plot_mean:
            for r in range(ax.shape[0]):
                for c in range(ax.shape[1]):
                    ttl = ax[r,c].get_title()
                    if ttl!="":
                        ax[r,c].axvline(dta[ttl].mean(), color='k', linestyle='dashed', linewidth=2)
        if save_fig:
            #ax.get_figure().savefig(figname+'.pdf')
            #ax.figure.savefig(figname, dpi=500)
            fig = plt.gcf()
            fig.savefig(figname)

def is_float(string):
	if string is not None:
		if string.replace(".", "").replace("-","").isnumeric():
			return True
		else:
			return False
	else:
		return False
