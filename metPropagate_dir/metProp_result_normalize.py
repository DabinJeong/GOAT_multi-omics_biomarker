from scipy.stats import zscore
import pandas as pd 
import numpy as np
import sys 
import os.path 


sampleID = sys.argv[1]
inp = sys.argv[2]
zscore_cut = sys.argv[3]
out = sys.argv[4]

df = pd.read_csv(inp)
df['Score.norm'] = zscore(df['Score.ID'])

genes = df.loc[lambda x:(x['Score.norm'])>=float(zscore_cut),'genes']
scores = df.loc[lambda x:(x['Score.norm'])>=float(zscore_cut),'Score.norm'].astype(str)
samples = np.full(len(genes),sampleID)

res = zip(genes,scores,samples)

if os.path.isfile(out)==False:
    with open(out,'a') as f:
        for item in res:
            print('\t'.join(item),file=f)
else:
    open(out,'w').close()
    with open(out,'a') as f:
        for item in res:
            print('\t'.join(item),file=f)

