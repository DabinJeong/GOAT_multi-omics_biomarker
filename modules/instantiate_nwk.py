import pandas as pd
import argparse
import numpy as np
import time
import sys
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

def isNum(x):
    try:
        float(x)
        return True
    except:
        return False

def corrCut(nwk,cutoff=None):
    '''weight cutoff, positive sorting'''
    nwk.dropna(subset=['weight'],inplace=True)
    nwk.sort_values(by=['weight'],inplace=True,ascending=False)
    if cutoff!=None: 
        return nwk.loc[lambda x:abs(x.weight)>=cutoff]
    else:
        return nwk

def setMinExp(nwk,exp,expCut):
    '''remove gene whose expression is lower than expCut'''
    filtered_gene = exp[exp.max(axis=1)>1]['Hybridization REF']
    boolean_mask = np.logical_and(nwk['protein1'].isin(filtered_gene),nwk['protein2'].isin(filtered_gene))
    return nwk[boolean_mask]

def expCut(nwk,exp,sample_list,expCut):
    '''remove gene whose mean of group(mutated/not-mutated) expression is lower than expCut'''
    with open(sample_list) as f:
        mutSamples=f.read().strip().split()
    exp['no_mut']=exp.loc[:,~exp.columns.isin(mutSamples)].mean(axis=1)
    exp['mut']=exp.loc[:,exp.columns.isin(mutSamples)].mean(axis=1)
    boolean_mask = np.logical_or(exp['no_mut']>=1,exp['mut']>=1)
    gene_selected = exp[boolean_mask]['Hybridization REF']
    boolean_mask2 = np.logical_and(nwk['protein1'].isin(gene_selected),nwk['protein2'].isin(gene_selected))
    return nwk[boolean_mask2]

def FCcut(nwk,FC_df,FCcut):
    keys = FC_df.iloc[:,0]
    values = FC_df.iloc[:,1]
    dictionary = dict(zip(keys, values))
    dictionary.pop('?','Not Found')
    first_col = np.array([dictionary[i] for i in nwk.iloc[:,0]])
    second_col = np.array([dictionary[i] for i in nwk.iloc[:,1]])
    boolean_mask = np.logical_and(abs(first_col) >= FCcut, abs(second_col) >= FCcut)
    boolean_mask2 = nwk['protein1'].apply(lambda x: dictionary[x])*nwk['protein2'].apply(lambda x: dictionary[x])>0
    bigFC_nwk = nwk[boolean_mask & boolean_mask2]
    return bigFC_nwk

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='python 0_instantiate_nwk.py nwkFile exp -corrCut [] -nThreads [] -o []')
    parser.add_argument('nwk',help='network')
    parser.add_argument('exp',help='exp File')
    parser.add_argument('-corrCut',type=float, required=False,help='corealtion cutoff')
    parser.add_argument('-nThreads',type=int, required=False,default=1)
    parser.add_argument('-o',required=True,help='output')
    args=parser.parse_args()
        
    ####correaltion score combined with string score
    start=time.time()
    exp=pd.read_csv(args.exp,sep='\t',header=0,index_col=0)
    
    #remove duplicates
    data=[]
    with open(args.nwk) as f:
        for line in f.readlines():
            tmp=line.strip().split('\t')
            data.append(sorted(tmp))
    
    df_nwk=pd.DataFrame(data[1:],columns=['node_1','node_2'],dtype=float)
    
    df_nwk.drop_duplicates(subset=['node_1','node_2'],inplace=True)
    df_nwk_filt = df_nwk.loc[lambda x:np.logical_and(x.node_1.isin(exp.index),x.node_2.isin(exp.index))].loc[lambda x:x.node_1 != x.node_2]
    
    #make exp dictionary to calculate weight
    lst_exps=dict() 
    with open(args.exp) as f:
        lines=f.readlines()
    for line in lines:
        s=line.strip().split('\t')
        if not isNum(s[1]):
            continue
        else:
            gene, exps = s[0], list(map(float,s[1:]))
            lst_exps[gene]=exps
    lst_pairs2=zip(df_nwk_filt['node_1'],df_nwk_filt['node_2'])
    
    def myCorr(x):
        g1,g2=sorted(x)
        if np.all(np.array(lst_exps[g1])==lst_exps[g1][0]) or np.all(np.array(lst_exps[g2])==lst_exps[g2][0]):
            val,pval=(0,1)
        else:
            val, pval = pearsonr(lst_exps[g1],lst_exps[g2])
        return (g1,g2,abs(val))
    
    p = Pool(args.nThreads)
    res2=p.imap_unordered(myCorr, lst_pairs2)
    p.close()
    p.join()

    corr_res2=[]
    for g1,g2,val in res2:
        if g1==g2:
            continue
        corr_res2.append([g1,g2,val])

    df_nwk_corr=pd.DataFrame(corr_res2,columns=['node_1','node_2','weight'])
    df_nwk_corrCut=corrCut(df_nwk_corr,args.corrCut)
    
    end=time.time()
    time_elapsed=end-start
    print('tr_tr',df_nwk_corrCut.shape)
    #df_nwk_corrCut.to_csv(args.o,sep='\t',header=False,index=False)
    df_nwk_corrCut.loc[:,['node_1','node_2']].to_csv(args.o,sep='\t',header=False,index=False)
    #print(args.o, 'time_elapsed', time_elapsed)
