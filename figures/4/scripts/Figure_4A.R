#### ---------
#### Use rviz.yml environment
#### ---------

library(tidyr)
library(dplyr)
library(ggalluvial)
library(scales) 

# read data with broad cell types from all BC cores 
fh <- 'data/all_alluvial_input.csv'
df <- read.csv(fh)

# define axis order and rename labels 
df$panel <- factor(df$panel, levels = c('tumor','combined','immune'))
tick_labs <- c("Tumor Panel", "Combined Panel", "Immune Panel")

# cell type order 
df$cell_type <- factor(
    df$cell_type, 
    levels = c(
        'Epithelial', 
        'Epithelial/Immune',
        'Immune',
        'Stroma', 
        'Immune/Stroma',
        'Myoepithelial', 
        'Endothelial'
))

# alluvial plot
ggplot(df,
    aes(x = panel, 
        stratum = cell_type, 
        alluvium = subject,
        y = frequency,
        fill = cell_type, 
        label = cell_type)) +
    scale_x_discrete(expand = c(.1, .1), labels=tick_labs) +
    geom_flow(width = 5/12, alpha = .8) +
    geom_stratum(width = 5/12, alpha = .8) +
    geom_text(stat = "stratum", size = 3) +
    theme_bw() + 
    theme(legend.position = "none") +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab('') + 
    ylab('Cell type frequency') +
    scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
    theme(axis.text.x = element_text(size = 12, color = 'black'))


ggsave('Figure_4A.png', dpi = 500, width = 7, height = 7)