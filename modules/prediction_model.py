import pandas as pd
import numpy as np
import argparse
import json

#### GCN #################
from GCN_transformer import *

import os.path as osp
import os

import random

import networkx as nx
from scipy import sparse

import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import torch_geometric.nn as geom_nn
from torch.utils.data import random_split
from torch_geometric.nn import GATConv, GraphConv, GCNConv

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

###########Dataset generation############
import argparse
import networkx as nx
import numpy as np
from scipy import sparse

import torch
import torch.nn.functional as F
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, LabelEncoder

#======================================================================
def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.set_deterministic(True)

### Dataset generation
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor: necessary for 'edges -> coo' format conversion"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.compat.long))
    return indices

def processing_topology(graph):
    '''
    input: edgeList (source, target)
    output: coo format for GNN    # .tocoo() alone does not directly returns the coo format
    '''
    nodes = sorted(list(graph.nodes()))
    adj_mx = np.array(nx.adjacency_matrix(graph, nodelist=nodes).todense())
    edge_index = sparse_mx_to_torch_sparse_tensor(sparse.csr_matrix(adj_mx).tocoo())
    return nodes, edge_index

class AsthmaDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self):
        return 'data.pt'
    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",'-l',dest='label')
    parser.add_argument("-t")
    parser.add_argument("-m")
    parser.add_argument("-p")
    parser.add_argument("-clin")
    parser.add_argument("-train_samples")
    parser.add_argument("-test_samples")
    parser.add_argument("-featureSelection")
    parser.add_argument("-propOut1")
    parser.add_argument("-propOut2")
    parser.add_argument("-DEG")
    parser.add_argument("-DEP")
    parser.add_argument("-K",type=int)
    parser.add_argument("-exp_name")
    parser.add_argument("-nwk")
    parser.add_argument('-random_seed',type=int,default=1)
    args = parser.parse_args()
  
    torch.cuda.empty_cache()
    random_seed = args.random_seed
    seed_everything(random_seed+41)

    if args.featureSelection=='ourBiomarker':
        # Retrieve biomarker candidates
        our_out1 = pd.read_csv(args.propOut1,sep='\t',names=['node','prop_score'],header=0).sort_values(by='prop_score',ascending=False)
        our_out2 = pd.read_csv(args.propOut2,sep='\t',names=['node','prop_score'],header=0).sort_values(by='prop_score',ascending=False)
        our_out_genes1 = our_out1.sort_values(by='prop_score',ascending=False)['node'].unique()
        our_out_genes2 = our_out2.sort_values(by='prop_score',ascending=False)['node'].unique()
        our_out_genes = list(set.union(set(our_out_genes1[:args.K]),set(our_out_genes2[:args.K])))
        genes = our_out_genes

    if args.featureSelection == 'DEG':
        DEG_raw = pd.read_csv(args.DEG, sep='\t',names=['gene','adj.p-val','stats']).dropna()
        DEGs = DEG_raw.sort_values(by='adj.p-val',ascending=True).loc[lambda x:x['adj.p-val']<0.05,:]['gene'].to_list() 
        genes = DEGs    

    if args.featureSelection == 'DEP':
        DEP_raw = pd.read_csv(args.DEP, sep='\t',names=['gene','adj.p-val','stats']).dropna()
        DEPs = DEP_raw.sort_values(by='adj.p-val',ascending=True).loc[lambda x:x['adj.p-val']<0.05,:]['gene'].to_list() 
        genes = DEPs
    
    # Data preparation
    label=args.label

    transcriptome = pd.read_csv(args.t,sep='\t',index_col=0)
    transcriptome.columns = transcriptome.columns.astype(int)

    methylome= pd.read_csv(args.m,sep='\t',index_col=0)
    methylome.columns = methylome.columns.astype(int)

    proteome = pd.read_csv(args.p,sep='\t',index_col=0)
    proteome.columns = proteome.columns.astype(int)

    clinical_raw = pd.read_csv(args.clin,sep='\t',index_col=0)
    dict_clinical = clinical_raw.reset_index().groupby(label)['SUBJNO'].apply(list).to_dict()
    all_samples_clinical = {v for i in dict_clinical.values() for v in i}
    samples_common_omics = set.intersection(set(transcriptome.columns), set(methylome.columns), set(proteome.columns))
    train_samples = set([int(l.strip()) for l in  open(args.train_samples).readlines()])
    test_samples = set([int(l.strip()) for l in  open(args.test_samples).readlines()])
    dict_clinical_r = clinical_raw.loc[:,label].to_dict()
    all_samples_omics_clinical = samples_common_omics.intersection(all_samples_clinical)

    def clinical_label(x,dict_):
        if x in dict_[1]:
            return 'low'
        elif x in dict_[2]:
            return 'high'
        else:
            return 'None'
   
    # normalize input data
    scaler = MinMaxScaler() 
    
    transcriptome_filt_raw = transcriptome.T.loc[list(set(transcriptome.T.index) & set(all_samples_clinical)),
                                                 transcriptome.T.columns.intersection(genes)]
    methylome_filt_raw = methylome.T.loc[list(set(methylome.T.index) & set(all_samples_clinical)),
                                         methylome.T.columns.intersection(genes)]
    proteome_filt_raw = proteome.T.loc[list(set(proteome.T.index) & set(all_samples_clinical)),
                                       proteome.T.columns.intersection(genes)]
    
    transcriptome_filt = pd.DataFrame(scaler.fit_transform(transcriptome_filt_raw.T).T, 
                                      index=transcriptome_filt_raw.index, 
                                      columns=transcriptome_filt_raw.columns)
    methylome_filt = pd.DataFrame(scaler.fit_transform(methylome_filt_raw.T).T, 
                                  index=methylome_filt_raw.index, 
                                  columns=methylome_filt_raw.columns)
    proteome_filt = pd.DataFrame(scaler.fit_transform(proteome_filt_raw.T).T, 
                                 index=proteome_filt_raw.index, 
                                 columns=proteome_filt_raw.columns)
    
    ########### Dataset generation for GCN #################
    nwk = pd.read_csv(args.nwk,sep='\t',names=['g1','g2'])
    G = nx.from_pandas_edgelist(nwk, source='g1', target='g2') 
    
    subgraph = G.subgraph(genes)

    lcc_nodes = max(nx.connected_components(subgraph), key=len)
    subgraph = subgraph.subgraph(lcc_nodes)
    
    nodes, edge_index = processing_topology(subgraph)
  
    def imputed_per_group(df):
        global train_samples, dict_clinical, genes
        group1 = df.loc[list(set.intersection(set(df.index), set(train_samples), set(dict_clinical[1]))),:].mode().iloc[0,:].median()
        group2 = df.loc[list(set.intersection(set(df.index), set(train_samples), set(dict_clinical[2]))),:].mode().iloc[0,:].median()
        return group1, group2 
   
    def imputed(df):
        global train_samples
        median = df.loc[list(set.intersection(set(df.index), set(train_samples))),:].mode().iloc[0,:].median() 
        return median

    tr_1, tr_2 = imputed_per_group(transcriptome_filt)
    m_1, m_2 = imputed_per_group(methylome_filt)
    p_1, p_2 = imputed_per_group(proteome_filt)

    tr_ = imputed(transcriptome_filt)
    m_ = imputed(methylome_filt)
    p_ = imputed(proteome_filt)

    data_list_train = []
    data_list_test = []

    
    for sample in all_samples_clinical:                                                           
        if sample in train_samples:
            x_tmp = []
            for gene in nodes:
                if (gene in transcriptome_filt.columns) and (sample in transcriptome_filt.index):
                    a = transcriptome_filt.loc[sample,gene]
                else:
                    a = np.full(1,tr_)
                if (gene in methylome_filt.columns) and (sample in methylome_filt.index):
                    b = methylome_filt.loc[sample,gene]
                else:
                    b = np.full(1,m_)
                if (gene in proteome_filt.columns) and (sample in proteome_filt.index):
                    c = proteome_filt.loc[sample,gene]
                else:
                    c = np.full(1,p_)
                all_data = list(np.c_[a,b,c])
                x_tmp.append(all_data)
            x_tmp_tensor = torch.tensor(np.array(x_tmp,dtype=np.float32)).view(-1,3)
            if dict_clinical_r[sample]==1:
                data = Data(x=x_tmp_tensor, y=torch.tensor([0]), edge_index=edge_index)
            if dict_clinical_r[sample]==2:
                data = Data(x=x_tmp_tensor, y=torch.tensor([1]), edge_index=edge_index)
            data_list_train.append(data)
        if sample in test_samples:
            x_tmp = []
            for gene in nodes:
                if (gene in transcriptome_filt.columns) and (sample in transcriptome_filt.index):
                    a = transcriptome_filt.loc[sample,gene]
                else:
                    a = np.full(1,tr_)
                if (gene in methylome_filt.columns) and (sample in methylome_filt.index):
                    b = methylome_filt.loc[sample,gene]
                else:
                    b = np.full(1,m_)
                if (gene in proteome_filt.columns) and (sample in proteome_filt.index):
                    c = proteome_filt.loc[sample,gene]
                else:
                    c = np.full(1,p_)
                all_data = list(np.c_[a,b,c])
                x_tmp.append(all_data)
            x_tmp_tensor = torch.tensor(np.array(x_tmp,dtype=np.float32)).view(-1,3)
            if dict_clinical_r[sample]==1:
                data = Data(x=x_tmp_tensor, y=torch.tensor([0]), edge_index=edge_index)
            if dict_clinical_r[sample]==2:
                data = Data(x=x_tmp_tensor, y=torch.tensor([1]), edge_index=edge_index)          
            data_list_test.append(data)                                                         

    Dataset_name = "Dataset_minmaxSample_{}".format(args.exp_name)
    Asthma_train = AsthmaDataset(Dataset_name, data_list_train)
    Asthma_test = AsthmaDataset(Dataset_name+'.test', data_list_test)


    ### Train model
    VALID_RATIO = 0.8
    
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    n_train_examples = int(len(Asthma_train) * VALID_RATIO)
    n_valid_examples = len(Asthma_train) - n_train_examples

    def stratified_split(dataset):
        global VALID_RATIO,random_seed
        from sklearn.model_selection import train_test_split
        labels=[data.y.item() for data in dataset]
        train_indices, val_indices = train_test_split(list(range(len(labels))),train_size=VALID_RATIO,shuffle=True,stratify=labels,random_state=(random_seed+42))

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        return train_dataset, val_dataset
     
    graph_train_data, graph_valid_data = stratified_split(Asthma_train)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    
    graph_train_loader = torch_geometric.loader.DataLoader(graph_train_data,shuffle=True,batch_size=BATCH_SIZE,worker_init_fn=seed_worker,generator=g,num_workers=0) 
    graph_val_loader = torch_geometric.loader.DataLoader(graph_valid_data,shuffle=True,batch_size=BATCH_SIZE,worker_init_fn=seed_worker,generator=g,num_workers=0)
    graph_test_loader = torch_geometric.loader.DataLoader(Asthma_test,batch_size=1,worker_init_fn=seed_worker,generator=g,num_workers=0)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    args.epochs = 40
    args.test = True
    args.learning_rate =  0.001
    args.batch_size = 10
    args.weight_decay = 0 
    args.dropout_rate =  0.2
    device = args.device

    model, best_performances, test_loss, test_acc = experiment_graph(args, graph_train_loader, graph_val_loader, graph_test_loader)
    
    y_li , true_y_li = [],[]
    for data in graph_test_loader:
        data = data.to(device)
        out, att_idx, att_w = model(data)
        
        y = out.cpu().detach().flatten().tolist()
        true_y = data.y.cpu().detach().flatten().tolist()
    
        y_li.extend(y)
        true_y_li.extend(true_y)
    
    from sklearn import metrics
    fpr,tpr,thres_roc = metrics.roc_curve(true_y_li,y_li,pos_label=1)
    precision,recall,thres_pr = metrics.precision_recall_curve(true_y_li,y_li,pos_label=1)
    
    auprc = metrics.auc(recall,precision)
    auroc = metrics.auc(fpr,tpr)
    from collections import Counter
    with open(args.exp_name+".performance.txt",'w') as f:
        print("AUROC: {:.5f}".format(auroc),file=f)
        print("AUPRC: {:.5f} \t baseline: {:.5f}".format(auprc, Counter(true_y_li)[1]/len(true_y_li)), file=f)
    print("==================")
    file_name = args.exp_name + '.TransformerConv.best_model'
    torch.save(model.state_dict(),file_name)
