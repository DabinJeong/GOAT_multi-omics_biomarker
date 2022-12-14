unlink(".RData")
.libPaths("/zfs3/users/grahamemma9/grahamemma9/R/x86_64-pc-linux-gnu-library/3.5")

suppressPackageStartupMessages(library(tidyverse))
library(tidyverse,verbose=FALSE)
library(stringr,verbose=FALSE)
args=(commandArgs(TRUE))
names <-as.character(args[[1]])
names_perc_in_hmdb <- as.character(args[[2]])
names_gene_list_node_ids <- as.character(args[[3]])
names_scores <- as.character(args[[4]])
names_original_scores <- as.character(args[[5]])
out <- as.character(args[[6]])
#permutations <- as.numeric(args[[2]])
network_name <- "STRING"

#file that denotes percentage of first degree neighbors annotated in HMDB.  
perc_in_hmdb <- read.csv(names_perc_in_hmdb,stringsAsFactors = FALSE)

#load label propagation scores (scores), candidate gene list (variant_info), gene to node id mapping, pre-propagation label values (original_scores)

scores <- read.csv(names_scores)
colnames(scores) <- c("Node.ID", "Score.ID")

#variant_info <- read.csv(paste0(c("./wes_files/", names, "_candidate_genes.csv"), collapse = ""),
#  												 header = TRUE,
#  												 na.strings = c("None", "-1", "NA"),
#  												 stringsAsFactors = FALSE)

#Isolate exomiser Phenotype scores, make sure only one score per gene (multiple rows due to the fact that Exomiser also outputs variant score)
#final_variant_score_df <- data.frame(gene = variant_info$X.GENE_SYMBOL, 
#                                     Exomiser_phenotype_score = variant_info$EXOMISER_GENE_PHENO_SCORE) %>% 
#  group_by(gene) %>% 
#  filter(Exomiser_phenotype_score == max(Exomiser_phenotype_score)) %>% 
#  unique()

gene_list_node_ids <- read.table(names_gene_list_node_ids,
                                    sep = "\t", 
                                    stringsAsFactors = FALSE, 
                                    header = TRUE)
original_scores <- read.table(names_original_scores)[, c(1, 2)]
names(original_scores) <- c("Node.ID", "Original.Score.ID")

#join label propagation output with node id to gene name mapping
output <- scores %>% 
     left_join(gene_list_node_ids, by = c("Node.ID" = "node_id")) #%>% filter(genes %in% variant_info$X.GENE_SYMBOL)

write.csv(output[,c("genes","Score.ID")],paste0(c("./integration/results/", names, "_", network_name, "_ranked_label_propagation_scores.csv"), collapse = ""), row.names = FALSE)
#further join output file with original scores
#whole_df <- output %>% 
#    select(Node.ID, genes, Score.ID) %>% 
#    left_join(original_scores, by = "Node.ID") %>%
#    #filter(genes %in% variant_info$X.GENE_SYMBOL) %>%  
#    arrange(desc(Score.ID)) %>% mutate(metPropagate_rank = 1:nrow(output)) %>%
#    #left_join(final_variant_score_df, by = c("genes" = "gene")) %>% 
#    left_join(perc_in_hmdb, by = "genes") %>% 
#    mutate( scaled1_metProp = (1-weight)*(Score.ID - min(Score.ID, na.rm = TRUE))/(max(Score.ID, na.rm = TRUE)-min(Score.ID, na.rm = TRUE)),
#    scaled_metProp = (1-weight)*(Score.ID - min(Score.ID, na.rm = TRUE))/(max(Score.ID, na.rm = TRUE)-min(Score.ID, na.rm = TRUE)),
#    scaled2_metProp = (scaled1_metProp - min(scaled1_metProp, na.rm = TRUE))/(max(scaled1_metProp, na.rm = TRUE)-min(scaled1_metProp, na.rm = TRUE)),
#    scaled1_exom_phen = weight*(Exomiser_phenotype_score - min(Exomiser_phenotype_score, na.rm = TRUE))/(max(Exomiser_phenotype_score)-min(Exomiser_phenotype_score, na.rm = TRUE)),
#    scaled2_exom_phen = (scaled1_exom_phen - min(scaled1_exom_phen, na.rm = TRUE))/(max(scaled1_exom_phen, na.rm = TRUE)-min(scaled1_exom_phen, na.rm = TRUE)),
#    scaled_exom_phen = weight*(Exomiser_phenotype_score - min(Exomiser_phenotype_score, na.rm = TRUE))/(max(Exomiser_phenotype_score, na.rm = TRUE)-min(Exomiser_phenotype_score, na.rm = TRUE)),
#    scaled_by_met_active_comb_score = scaled2_metProp+scaled2_exom_phen) %>% 
#    arrange(desc(scaled_by_met_active_comb_score)) %>% 
#    mutate(Exom_and_metPropagate_rank = 1:nrow(output)) %>% 
#    select(Node.ID, genes, Score.ID, Original.Score.ID, metPropagate_rank, Exom_and_metPropagate_rank, Exom_phenotype_score)
# 
#write.csv(whole_df, paste0(c("./results/", names, "_", network_name, "_ranked_label_propagation_scores.csv"), collapse = ""), row.names = FALSE)
