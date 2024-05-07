"""
Complete code credit to Erik Burlingame
Code adapted from https://gitlab.com/eburling/grapheno
"""
import os
import time
import cuml
import cudf
import cugraph
import dask_cudf
import cupy as cp
from dask.distributed import Client
from cuml.neighbors import NearestNeighbors as NN
from cuml.dask.neighbors import NearestNeighbors as DaskNN
	
def compute_and_cache_knn_edgelist(input_csv_path, 
								   knn_edgelist_path, 
								   features, 
								   n_neighbors,
								   client=None):
	
	print(f'    Computing and caching {n_neighbors}NN '
		  f'edgelist: {knn_edgelist_path}')
	
	if client:
		chunksize = cugraph.dask.get_chunksize(input_csv_path)
		X = dask_cudf.read_csv(input_csv_path, chunksize=chunksize)
		X = X.loc[:, features].astype('float32')
		model = DaskNN(n_neighbors=n_neighbors+1, client=client)
	else:
		X = cudf.read_csv(input_csv_path)
		X = X.loc[:, features].astype('float32')
		model = NN(n_neighbors=n_neighbors+1)
	
	model.fit(X)
	
	n_vertices = X.shape[0].compute() if client else X.shape[0]
	
	# exclude self index
	knn_edgelist = model.kneighbors(X, return_distance=False).loc[:, 1:]  
	if client: # gather from GPUs and make index a contiguous range
		knn_edgelist = knn_edgelist.compute().reset_index(drop=True)
	knn_edgelist = knn_edgelist.melt(var_name='knn', value_name='dst')
	knn_edgelist = knn_edgelist.reset_index().rename(columns={'index':'src'})
	knn_edgelist = knn_edgelist.loc[:, ['src', 'dst']]
	knn_edgelist['src'] = knn_edgelist['src'] % n_vertices # avoids transpose
	knn_edgelist.to_parquet(knn_edgelist_path, index=False)
	
	
def compute_and_cache_jac_edgelist(knn_edgelist_path, 
								   jac_edgelist_path):
	
	print(f'    Computing and caching jaccard edgelist: {jac_edgelist_path}')
	knn_graph = load_knn_graph(knn_edgelist_path)
	jac_graph = cugraph.jaccard(knn_graph)
	jac_graph.to_parquet(jac_edgelist_path, index=False)
	
	
def load_knn_graph(knn_edgelist_path):
	G = cugraph.Graph()
	knn_edgelist = cudf.read_parquet(knn_edgelist_path)
	G.from_cudf_edgelist(knn_edgelist, source='src', destination='dst')
	return G


def load_jac_graph(jac_edgelist_path):
	G = cugraph.Graph()
	jac_edgelist = cudf.read_parquet(jac_edgelist_path)
	G.from_cudf_edgelist(jac_edgelist, edge_attr='jaccard_coeff')
	return G


def sort_by_size(clusters, min_size):
	"""
	Relabel clustering in order of descending cluster size.
	New labels are consecutive integers beginning at 0
	Clusters that are smaller than min_size are assigned to -1.
	Copied from https://github.com/jacoblevine/PhenoGraph.
	
	Parameters
	----------
	clusters: array
		Either numpy or cupy array of cluster labels.
	min_size: int
		Minimum cluster size.
	Returns
	-------
	relabeled: cupy array
		Array of cluster labels re-labeled by size.
		
	"""
	relabeled = cp.zeros(clusters.shape, dtype=int)
	_, counts = cp.unique(clusters, return_counts=True)
	# sizes = cp.array([cp.sum(clusters == x) for x in cp.unique(clusters)])
	o = cp.argsort(counts)[::-1]
	for i, c in enumerate(o):
		if counts[c] > min_size:
			relabeled[clusters == c] = i
		else:
			relabeled[clusters == c] = -1
	return relabeled


################################################
# NOTES #
# Jaccard similarity and Leiden clustering don't
# have distributed GPU implementations yet,
# but they probably will soon, at which point
# it will be worth loading graphs using
# dask_cudf edgelists. As of RAPIDS 22.08 there
# is a distributed GPU implementation of Louvain
# if you run out of memory on single GPU
# computation of Leiden clustering. Note that
# such changes to RAPIDS will likely require
# reworking this code to accomodate, but should
# not be too much, e.g. change cugraph.jaccard
# to cugraph.dask.jaccard, etc.
################################################
def cluster(input_csv_path,
			features,
			data=None,
			n_neighbors=30,
			new_column_name = 'cluster',
			overwrite = False,
			write_out_parquet = True,
			min_size=10):
	"""
	Added "new_column_name" so that we may further delineate sub-clusters within clusters. 
	Specifically, we could pass 'cluster' or 'subcluster' as an argument.
	Added overwrite, in case we wish to entirely redo the clustering.
	It is important to note that 'features' are the only columns that will be used to cluster the cells.
	"""
	
	tic = time.time()

	knn_edgelist_path = os.path.basename(input_csv_path).rsplit('.', 1)[0]
	knn_edgelist_path = f'{knn_edgelist_path}_{n_neighbors}NN_edgelist.parquet'

	jac_edgelist_path = os.path.basename(knn_edgelist_path).rsplit('.', 1)[0]
	jac_edgelist_path = f'{jac_edgelist_path}_jaccard.parquet'

	subtic = time.time()

	if os.path.exists(jac_edgelist_path) and (overwrite==False):

		print(f'    Loading cached jaccard edgelist into graph: {jac_edgelist_path}')

		jac_graph = load_jac_graph(jac_edgelist_path)

		print(f'    Jaccard graph loaded in {(time.time()-subtic):.2f} seconds...')

	elif os.path.exists(knn_edgelist_path) and (overwrite==False):

		print('    Loading cached kNN edgelist for Jaccard graph '
			  f'computation: {knn_edgelist_path}')

		compute_and_cache_jac_edgelist(knn_edgelist_path, 
									   jac_edgelist_path)

		jac_graph = load_jac_graph(jac_edgelist_path)

		print('    Jaccard graph computed, cached, and reloaded in '
			  f'{(time.time()-subtic):.2f} seconds...')

	else:

		compute_and_cache_knn_edgelist(input_csv_path, 
										knn_edgelist_path, 
										features, 
										n_neighbors)

		print(f'    {n_neighbors}NN edgelist computed and cached in '
			  f'    {(time.time()-subtic):.2f} seconds...')

		subtic = time.time()

		compute_and_cache_jac_edgelist(knn_edgelist_path, 
									   jac_edgelist_path)

		jac_graph = load_jac_graph(jac_edgelist_path)

		print('    Jaccard graph computed, cached, and reloaded in '
			  f'{(time.time()-subtic):.2f} seconds...')

	subtic = time.time()

	print('    Computing Leiden clustering over Jaccard graph...')
	clusters, modularity = cugraph.leiden(jac_graph)
	print(f'    Leiden clustering completed in {(time.time()-subtic):.2f} seconds...')
	print(f'    Clusters detected: {len(clusters.partition.unique())}')
	print(f'    Clusters modularity: {modularity}')
		
	clusters = clusters.sort_values(by='vertex').partition.values
	clusters = sort_by_size(clusters, min_size)
	
	df = cudf.read_csv(input_csv_path)
	df[new_column_name] = clusters
	df.index.name = None #if we loaded from a csv file containing an index column, it autoloads a header name for it. Delete it.

	if write_out_parquet:
		out_parquet_path = input_csv_path.rsplit('.', 1)[0]
		out_parquet_path = f'{out_parquet_path}_{n_neighbors}NN_leiden.parquet'
		print(f'    Writing output dataframe: {out_parquet_path}')
		df.to_parquet(out_parquet_path, index=False)
		
	print(f'    Grapheno completed in {(time.time()-tic):.2f} seconds!')
	
	return df