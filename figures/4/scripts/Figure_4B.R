#### -------
#### Use the rviz.yml environment 
#### -------

library(tidyr)
library(dplyr)
library(ggalluvial)
library(scales) 

# read formatted data 
fh <- 'data/E06_alluvial_input.csv'
df <- read.csv(fh)

# define axis order and reformat axis tick labels 
df$panel <- factor(df$panel, levels = c('Tumor','Combined','Immune'))
tick_labs <- c("Tumor Panel", "Combined Panel", "Immune Panel")

# define a cell type order 
df$cell_type <- factor(
    df$cell_type, 
    levels = c(
        'Epithelial', 
        'Basal Epithelial', 
        'Proliferative Epithelial', 
        'Proliferative Basal Epithelial',
        'Myoepithelial', 
        'Stroma',
        'Endothelial',
        'aSMA+ Stroma', 
        'B-cells', 
        'T-cells', 
        'CD8+ T-cells', 
        'CD4+ T-cells',
        'Regulatory T-cells',
        'Macrophages', 
        'CD163+ Macrophages'
    )
)

# cell type colors 
cell_type_colors <- c(
    '#d60000', 
    '#8c3bff', 
    '#018700',
    '#00acc6', 
    '#97ff00', 
    '#ff7ed1', 
    '#6b004f',
    '#ffa52f', 
    '#00009c',
    '#857067', 
    '#004942', 
    '#4f2a00',
    '#00fdcf', 
    '#bcb6ff',
    '#95b379'   
)

# map cell type names to colors 
names(cell_type_colors) <- levels(df$cell_type)
colScale <- scale_fill_manual(name = "Cell type", values = cell_type_colors)

# create the alluvial plot 
ggplot(df,
    aes(x = panel, 
        stratum = cell_type, 
        alluvium = subject,
        y = frequency,
        fill = cell_type, 
        label = cell_type)) +
    scale_x_discrete(expand = c(.1, .1), labels=tick_labs) +
    geom_flow(alpha = .8) +
    geom_stratum(alpha = .8) +
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab('') + 
    ylab('Cell type frequency') +
    scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
    theme(axis.text.x = element_text(size = 12, color = 'black')) + 
    colScale


ggsave('Figure_4B.png', dpi = 500, width = 8, height = 7)
