import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--clin")
parser.add_argument("--label")
parser.add_argument("--fold",type=int)
args = parser.parse_args()

label=args.label

clin_raw = pd.read_csv(args.clin,index_col=0,sep='\t')
dict_clinical = clin_raw.loc[:,args.label].to_dict()

all_samples_clinical = {i for i in dict_clinical.keys() if dict_clinical[i] != None}

clin = clin_raw.loc[list(all_samples_clinical),args.label].reset_index()

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
sss.get_n_splits(clin['SUBJNO'], clin[args.label])

tmp = list(sss.split(clin['SUBJNO'], clin[args.label]))
train_index, test_index = tmp[int(args.fold)-1]
train_samples, test_samples = clin.loc[train_index,'SUBJNO'], clin.loc[test_index,'SUBJNO']

with open("{}_samples_train.txt".format(args.label),'w') as f:
    f.write('\n'.join([str(i) for i in train_samples]))
with open("{}_samples_test.txt".format(args.label),'w') as f:
    f.write('\n'.join([str(i) for i in test_samples]))
    
clin_raw.loc[train_samples].to_csv("Merged_clinical_train.tsv".format(args.label),sep='\t')

