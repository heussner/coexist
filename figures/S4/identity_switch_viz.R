library(tidyr)
library(dplyr)
library(ggalluvial)
library(scales) 
library(gridExtra)

fh <- 'data/E06_alluvial_input.csv'
df <- read.csv(fh)

df$panel <- factor(df$panel, levels = c('Tumor','Combined','Immune'))

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

names(cell_type_colors) <- levels(df$cell_type)
colScale <- scale_fill_manual(name = "Cell type", values = cell_type_colors)

df <- df %>% filter(cell_type %in% c('aSMA+ Stroma','Macrophages'))

### Alluvial plot subset to just aSMA Stroma and Macrophages 
g1 <- ggplot(df,
    aes(x = panel, 
        stratum = cell_type, 
        alluvium = subject,
        y = frequency,
        fill = cell_type, 
        label = cell_type)) +
    scale_x_discrete(expand = c(.1, .1), position = "top") +
    geom_flow(width = 5/12, alpha = .8) +
    geom_stratum(width = 5/12, alpha = .8) +
    # geom_text(stat = "stratum", size = 3, nudge_y = 500) +
    geom_text(stat = "stratum", size = 3) +
    # ggtitle('TNBC Cell type classification of tracked cells across panels') +
    theme_bw() + 
    theme(legend.position = "none") +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab('') + 
    ylab('Cell type frequency') +
    scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
    theme(axis.text.x = element_text(size = 12, color = 'black')) + 
    colScale +
    annotate("text", x = 1.5, y = 100, label = "n = 138 cells")


### Rank marker plots 
rank_fh <- 'data/E06_aSMA_Macrophage_ID_switch_ranks.csv'
rank_df <- read.csv(rank_fh)

# add filler data so that axes don't get squished on the tumor side 
rank_df <- rank_df %>% add_row(panel = 'Tumor', rank = 15)

rank_df$panel <- factor(rank_df$panel, levels = c('Tumor','Combined','Immune'))

# immune <- rank_df %>% filter(panel == 'immune')

g2 <- ggplot(rank_df, aes(x = rank, y = scores, label = names)) +
    geom_text(aes(color = marker_origin),hjust=0, vjust=0, angle = 90, fontface = 'bold') +
    scale_color_manual(values = c('darkgreen','purple')) +
    facet_grid(. ~ panel, scales = "free_x", space = 'free') +
    scale_x_continuous(expand = c(0.05, 0.5)) +
    ylim(-80,100) +
    theme_bw() +
    theme(legend.position = "none")



ggsave('Figure_S4.png', dpi = 500, arrangeGrob(g1,g2), width = 9, height = 7)
