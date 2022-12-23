from statsmodels.stats.multitest import multipletests
import sys
import pandas as pd
import numpy as np
from scipy.stats import ranksums
import argparse

def de_omics(dataset,clinical,out_f,label="PH_EOSI_B_300"):
   # filter out samples whose both  omics data and clinical data are present 
    d123456 = np.intersect1d(dataset.index.astype(int), clinical[label].dropna().index)
    p_label2 = clinical.loc[d123456][label]
    df = dataset.loc[d123456]
    g1 = p_label2==1.0
    g2 = p_label2==2.0
    li_t,li_p = [],[]
    for target in df.columns:
        t,p = ranksums(df.T.loc[target][g1], df.T.loc[target][g2])
        li_t.append(t)
        li_p.append(p)
    
    li_p_adj = multipletests(li_p)
    out = zip(df.columns, li_p_adj[1], li_t)
    
    with open(out_f,'w') as f:
        for target,p,t in out:
            f.write('%s\t%f\t%.5f\n'%(target,p,t))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgroup",dest='subgroup_label',required=True)
    parser.add_argument("--clinical",required=True)
    parser.add_argument("--omics",required=True)
    parser.add_argument("--out","-o",dest='o',required=True)
    args = parser.parse_args()
    
    # clinical data
    clinical = pd.read_csv(args.clinical,index_col=0,sep="\t")
    
    # omics data
    omics = pd.read_csv(args.omics,sep="\t",index_col=0).T
    omics_data = omics.fillna(0)
    omics_data.index = omics_data.index.astype(int)

    de_omics(omics_data, clinical, args.o, label=args.subgroup_label)
