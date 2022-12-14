import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--subgroup_label',type=str,required=True)
parser.add_argument('--metabolome',required=True)
parser.add_argument('--metabolome_identifier',required=True)
parser.add_argument('--thr',dest='thr',type=float, required=True)
parser.add_argument('--clinical_data',dest="clinical",required=True)
parser.add_argument('--dir_exec',type=str,required=True)
parser.add_argument('--dir_DAM',type=str,required=True)
parser.add_argument('--dir_out',type=str,required=True)
args = parser.parse_args()

meta_id = pd.read_csv(args.metabolome_identifier, sep="\t", index_col=0)['HMDB ID'].dropna().to_dict()
metabolome = pd.read_csv(args.metabolome, sep="\t", index_col=0).iloc[:,2:]
meta_norm_gene = metabolome.T.apply(lambda x:zscore(x)).T 

thr=args.thr

meta_norm_gene_b = meta_norm_gene.applymap(lambda x:1 if (abs(x)>=thr) else 0)
meta_norm_gene_b.index = meta_norm_gene_b.index.map(lambda x: meta_id[x] if x in meta_id else x)

idx_HMDB = meta_norm_gene_b.index.map(lambda x: True if x[:4]=='HMDB' else False)
meta_norm_gene_b_HMDB = meta_norm_gene_b.loc[idx_HMDB,:]

meta_norm_gene_HMDB = meta_norm_gene.copy().loc[idx_HMDB,:]
meta_norm_gene_HMDB.index = meta_norm_gene_HMDB.index.map(lambda x: meta_id[x] if x in meta_id else x)
meta_norm_gene_HMDB.columns = meta_norm_gene_HMDB.columns.astype(int)

all_clinical_subtypes = pd.read_csv(args.clinical,sep='\t',index_col=0).T

for clinical_feature in [args.subgroup_label]:
    sg = pd.DataFrame(all_clinical_subtypes.loc[clinical_feature,:].dropna().astype(str)).sort_values(by=clinical_feature)
    sg.index=sg.index.astype(int)
    sg.index.rename('SUBJNO',inplace=True)
    sg_dict = sg.reset_index().groupby(clinical_feature)['SUBJNO'].apply(list)

samples_sorted = sg.index.tolist()
num_samples_w_label = len(samples_sorted)

thr = 10

with open('{}/DEMETA_in_{}_1_seed'.format(args.dir_DAM,args.subgroup_label)) as f:
    seed1 = f.read().strip().split('\n')
    
with open('{}/DEMETA_in_{}_2_seed'.format(args.dir_DAM,args.subgroup_label)) as f:
    seed2 = f.read().strip().split('\n')

dict_metabolites_sg = {}
dict_metabolites_sg['1'] = seed1
dict_metabolites_sg['2'] = seed2

metabolite_set = pd.read_csv( args.dir_exec +"/METABOLOMIC_PROCESSING_PIPELINE/HMDB_ver4_gene_metabolite_annotations.csv")

gene_meta_mapper = metabolite_set.groupby('gene_name')['HMDB_ids'].apply(list)

background = meta_norm_gene_b_HMDB.index.tolist()
#background = sum(gene_meta_mapper,[])

df_genes_1 = pd.DataFrame(gene_meta_mapper.map(lambda x:'{}/{}'.format(len(set(x).intersection(dict_metabolites_sg['1'])),len(set(x).union(dict_metabolites_sg['1']))))).rename(columns={'HMDB_ids':'numOverlap_meta'}).sort_values(by='numOverlap_meta',ascending=False)
df_genes_2 = pd.DataFrame(gene_meta_mapper.map(lambda x:'{}/{}'.format(len(set(x).intersection(dict_metabolites_sg['2'])),len(set(x).union(dict_metabolites_sg['2']))))).rename(columns={'HMDB_ids':'numOverlap_meta'}).sort_values(by='numOverlap_meta',ascending=False)

genes_1 = df_genes_1.loc[lambda x:x.numOverlap_meta.apply(lambda x:int(x.split('/')[0]))>0,:].index.tolist()
genes_2 = df_genes_2.loc[lambda x:x.numOverlap_meta.apply(lambda x:int(x.split('/')[0]))>0,:].index.tolist()

from scipy.stats import fisher_exact

def MSEA(sig, background, metabolite_set):
    cat = set(metabolite_set)
    diff_and_in_cat = len(set(sig).intersection(set(metabolite_set)))
    diff_and_not_cat = len(set(sig)) - diff_and_in_cat
    
    not_diff = set(background).difference(set(sig))
    not_diff_in_cat= len(not_diff.intersection(cat))
    not_diff_not_cat = len(not_diff)-diff_and_not_cat 

    return fisher_exact([[diff_and_in_cat,diff_and_not_cat],[not_diff_in_cat,not_diff_not_cat]])[1]

def maxZscore(sig, metabolite_set):
    common_DAMs = set(sig).intersection(set(metabolite_set))
    return maxZscore

for subgroup in sg_dict.keys():
    metabolites_sg = dict_metabolites_sg[subgroup]
    samples_sg = sg_dict[subgroup]
    
    genes_MSEA = pd.DataFrame(gene_meta_mapper.map(lambda x: MSEA(metabolites_sg,background,x))).rename(columns={'HMDB_ids':'p.value'}).sort_values(by="p.value")
    
    from statsmodels.stats.multitest import multipletests
    
    genes_MSEA['FDR'] = multipletests(genes_MSEA['p.value'],method='fdr_bh')[1]
    
    metabolites_sg_filt = list(set(metabolites_sg).intersection(meta_norm_gene_HMDB.index))
    samples_sg_filt = list(set(samples_sg).intersection(meta_norm_gene_HMDB.columns))
    
    zscore_max = meta_norm_gene_HMDB.loc[metabolites_sg_filt, samples_sg_filt].max(axis=1).to_dict()
    
    df_sig_metabolite_li = pd.DataFrame(gene_meta_mapper.apply(lambda x:list(set(x).intersection(metabolites_sg_filt))))
    
    genes_MSEA['MaxZscore'] = df_sig_metabolite_li['HMDB_ids'].apply(lambda x:max(map(lambda x:zscore_max[x], np.array(x))) if len(x) != 0 else 0)
    
    genes_MSEA.reset_index().rename(columns={'gene_name':'genes'}).to_csv('{}/enrichment_files/{}_{}_enrichment_file.csv'.format(args.dir_out, args.subgroup_label, subgroup), index=False, header=True)
