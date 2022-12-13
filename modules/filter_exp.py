import json
import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--transcriptome',dest="inp_transcript",required=True)
parser.add_argument('--clinical',dest='clinical_subgroups',required=True)
parser.add_argument('--label',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()

df_transcript = pd.read_csv(args.inp_transcript, sep='\t', index_col=0)
df_transcript.columns = df_transcript.columns.astype(int)
all_clinical_subtypes = pd.read_csv(args.clinical_subgroups,sep='\t',index_col=0).iloc[6:,:]
sg = args.label

samples_sg = pd.DataFrame(all_clinical_subtypes.loc[:,sg].dropna().astype(str)).reset_index()['SUBJNO'].to_list()
samples = list(set(samples_sg).intersection(set(df_transcript.columns)))

cts_cpm_sg = df_transcript.loc[:,samples]
cts_cpm_sg.to_csv(args.out,sep='\t',index=True,header=True)
