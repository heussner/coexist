## Code and data for reproducing Figure 3 

- All Jupyter notebooks were executed in `figures/3/envs/scanpy.yml`
- All R scripts were executed in `figures/3/envs/rviz.yml`

### Fig 3A

![](images/Figure_3A.png)

*Description*
Alluvial plot of broad cell types called on tracked cells from all breast cancer and normal breast cores in the tissue microarray 

*Notebooks/scripts*
    1. Figure_3A.R

*Notes*

### Fig 3B

![](images/Figure_3B.png)

*Description*
Alluvial plot of refined cell types called on tracked cells from core E06 

*Notebooks/scripts*
1. `notebooks/E06_leiden_clustering.ipynb`
2. `notebooks/E06_prep_alluvial.ipynb`  
3. `scripts/Figure_3B.R`

*Notes*
- Notebook 1 runs Leiden clustering (via Scanpy) on core E06 to generate refined cell type annotations. For clustering of more than one sample, it is recommended to use Grapheno clustering demonstrated in `examples/clustering_example.ipynb`
- Cell type annotations of Leiden clusters assigned in Notebook 1 are found under `figures/3/annotations` and are used throughout subsequent figures 
- Notebook 2 converts data files with Leiden cluster annotations (Example: `data/E06_immune_indiv_leiden.csv`) into a counts matrix that can be read into `ggalluvial` to create the figure 
- Script 3 takes the counts matrix and generates the alluvial diagram 

### Fig 3C

![](images/Figure_3C.png)

*Description*
Annotated heatmap of refined cell types called on tracked cells from core E06 

*Notebooks/scripts*
- `notebooks/Figure_3C.ipynb`

*Notes*
- Takes output of `notebooks/E06_leiden_clustering.ipynb` to create heatmap. For convenience, intermediate files are provided (Example: `data/E06_immune_indiv_leiden.csv`)

### Fig 3D

![](images/Figure_3D.png)

*Description*

*Notebooks/scripts*

*Notes*

### Fig 3E

![](images/Figure_3E.png)

*Description*

*Notebooks/scripts*

*Notes*

### Fig 3F

![](images/Figure_3F.png)

*Description*

*Notebooks/scripts*

*Notes*
