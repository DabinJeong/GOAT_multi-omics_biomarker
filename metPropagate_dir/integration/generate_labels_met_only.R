unlink(".RData")
.libPaths("/zfs3/users/grahamemma9/grahamemma9/R/x86_64-pc-linux-gnu-library/3.5")
#assembles scores based on input metabolomic enrichment file and WES variant list
#setwd("/data/project/asthma/bin_final/metPropagate-master/integration")

args=(commandArgs(TRUE))
names <-as.character(args[[1]])
nwk <- as.character(args[[2]])
nwk_id <- as.character(args[[3]])
out <- as.character(args[[4]])

print(names)
suppressPackageStartupMessages(library(tidyverse))
library(tidyverse,verbose=FALSE)
#scales each gene's -log2(2+ p-value of enrichment) and multiples by largest feature z-score annotated to that gene. 
assemblingAndNormalizing <- function(metabolomics_enrichment_file, genes_in_network){
  df <- data.frame(genes = genes_in_network$genes)
  final_df <- select(left_join(df, met_genes_and_scores, by = "genes"), genes, p.value, MaxZscore)
  final_df <- final_df[complete.cases(final_df),]                                                   
  scaled_final_df <- final_df %>% 
    mutate(met_log_pvalues = -log2(final_df$p.value)) %>% 
    mutate(met_transf_scores = (((met_log_pvalues - min(met_log_pvalues, na.rm = TRUE))/(max(met_log_pvalues, na.rm = TRUE)-min(met_log_pvalues, na.rm = TRUE)))*MaxZscore),
           met_zscore_weighed = ((met_transf_scores - min(met_transf_scores, na.rm = TRUE))/(max(met_transf_scores, na.rm = TRUE)-min(met_transf_scores, na.rm = TRUE))))
  scaled_final_df <- scaled_final_df %>% select(genes, label = met_zscore_weighed)
  scaled_final_df[complete.cases(scaled_final_df),]
}

#creates label files. Replaces NA with 0. Permutes labels if performing more than one permutation. 
joinScaleLabels <- function(distance_scores, gene_list_node_ids, file_name){
  labels <- left_join(gene_list_node_ids, distance_scores, by = "genes")
  labels <- replace_na(labels, replace = list(genes = NA, node_id = NA, label = 0))
  print("labels")
  permuted_scores <- data.frame()
  labels <- cbind(labels, sample(x = labels$label, size = length(labels$label)))
  write.table(labels[,2:ncol(labels)], 
              file = file_name, 
              quote = FALSE, 
              col.names = FALSE,
              row.names = FALSE)
  labels[, 2:ncol(labels)]
}
#load all genes in STRING network 
genes_in_network <- read.table(nwk,
                               sep = "\t", stringsAsFactors = FALSE, header = TRUE)
#load gene to node id mapping file
gene_list_node_ids <- read.table(nwk_id,sep = "\t",
                                 stringsAsFactors = FALSE, header = TRUE)

network_name <- "STRING"

#load gene enrichment scores 
#met_genes_and_scores <- read.csv(paste0(c("../integration/enrichment_files/", names, "_enrichment_file.csv"),collapse = ""), stringsAsFactors = FALSE)
met_genes_and_scores <- read.csv(paste0(c("./enrichment_files/", names, "_enrichment_file.csv"),collapse = ""), stringsAsFactors = FALSE)

#load labels 
distance_scores <- assemblingAndNormalizing(met_genes_and_scores, genes_in_network)
label_df <- joinScaleLabels(distance_scores, gene_list_node_ids,out) 
