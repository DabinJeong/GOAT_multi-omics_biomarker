import sys 
sys.path.insert(1, 'modules')
import argparse
import networkx as nx
import pandas as pd
import torch
import os
import torch_geometric
from GCN_transformer import *
from prediction_model import * 

def resurrect_test_acc(test_iterator, PATH, num_nodes, dim_node_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = subtype_classifier(dim_node_features, 1, num_nodes, 8, 0.2).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    att_list = []
    y_li , true_y_li = [],[]
    for data in test_iterator:
        data = data.to(device)
        out, att_edge, att_weights = model(data)
    
        att_adj = torch.squeeze(torch_geometric.utils.to_dense_adj(att_edge, edge_attr=att_weights)).detach().numpy()
        att_list.append(att_adj)
        
        y = out.cpu().detach().flatten().tolist()
        true_y = data.y.cpu().detach().flatten().tolist()
    
        y_li.extend(y)
        true_y_li.extend(true_y)
    
    from sklearn import metrics
    fpr,tpr,thres_roc = metrics.roc_curve(true_y_li,y_li,pos_label=1)
    precision,recall,thres_pr = metrics.precision_recall_curve(true_y_li,y_li,pos_label=1)
    
    auprc = metrics.auc(recall,precision)
    auroc = metrics.auc(fpr,tpr)
    print(metrics.accuracy_score(true_y_li,np.array(y_li)>0.5))
    mean_att = np.mean(sum(att_list)/len(att_list), axis=2)
    #mean_att = sum(att_list)/len(att_list)
    return auroc, auprc, model, mean_att, true_y_li


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-label',help="")
    parser.add_argument('-train_samples')
    parser.add_argument('-test_samples')
    parser.add_argument('-model')
    parser.add_argument('-t')
    parser.add_argument('-m')
    parser.add_argument('-p')
    parser.add_argument('-clin')
    parser.add_argument('-nwk')
    parser.add_argument('-propOut1')
    parser.add_argument('-propOut2')
    parser.add_argument('-K',type=int)
    parser.add_argument('-att_thr',type=float)
    parser.add_argument('-out')
    args = parser.parse_args() 

    seed_everything(42)
    
    #=============================Data=============================
    transcriptome = pd.read_csv(args.t,sep='\t',index_col=0)
    transcriptome.columns = transcriptome.columns.astype(int)
    
    methylome= pd.read_csv(args.m,sep='\t',index_col=0)
    methylome.columns = methylome.columns.astype(int)
    
    proteome = pd.read_csv(args.p,sep='\t',index_col=0)
    proteome.columns = proteome.columns.astype(int)
    
    clinical_raw = pd.read_csv(args.clin,sep='\t',index_col=0)
    dict_clinical = clinical_raw.reset_index().groupby(args.label)['SUBJNO'].apply(list).to_dict()
    all_samples_clinical = list({v for i in dict_clinical.values() for v in i})
    samples_common_omics = set.intersection(set(transcriptome.columns), set(methylome.columns), set(proteome.columns))
    dict_clinical_r = clinical_raw.loc[:,args.label].to_dict()


    train_samples = [int(l.strip()) for l in open(args.train_samples).readlines()]
    test_samples = [int(l.strip()) for l in open(args.test_samples).readlines()]
   
    #==============================GNN==============================
    
    our_out1 = pd.read_csv(args.propOut1,sep='\t',names=['node','prop_score'],header=0).sort_values(by='prop_score',ascending=False)
    our_out2 = pd.read_csv(args.propOut2,sep='\t',names=['node','prop_score'],header=0).sort_values(by='prop_score',ascending=False)
    our_out_genes1 = our_out1.sort_values(by='prop_score',ascending=False)['node'].unique()
    our_out_genes2 = our_out2.sort_values(by='prop_score',ascending=False)['node'].unique()
    our_out_genes = list(set.union(set(our_out_genes1[:args.K]),set(our_out_genes2[:args.K])))

    genes = our_out_genes

    from sklearn.preprocessing import MinMaxScaler
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
    
    df_nwk = pd.read_csv(args.nwk,sep='\t',names=['g1','g2'])
    G = nx.from_pandas_edgelist(df_nwk, source='g1', target='g2')
    subgraph = G.subgraph(genes)
    
    lcc_nodes = max(nx.connected_components(subgraph), key=len)
    subgraph = subgraph.subgraph(lcc_nodes)

    def processing_topology(graph):
        '''
        input: edgeList (source, target)
        output: coo format for GNN    # .tocoo() alone does not directly returns the coo format
        '''
        nodes = sorted(list(graph.nodes()))
        adj_mx = np.array(nx.adjacency_matrix(graph, nodelist=nodes).todense())
        edge_index = sparse_mx_to_torch_sparse_tensor(sparse.csr_matrix(adj_mx).tocoo())
        return nodes, edge_index
    
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
                              
    data_list_test = []

    for sample in all_samples_clinical:
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
        if sample in test_samples:
            data_list_test.append(data)

    num_nodes = data_list_test[0].x.size()[0]
    dim_node_features = data_list_test[0].x.size()[1]
    
    auroc_ourBiomarker_GCN, auprc_ourBiomarker_GCN, model, mean_attr, labels = resurrect_test_acc(data_list_test, args.model, num_nodes, dim_node_features)
    degrees = pd.DataFrame(subgraph.degree(nodes))[1].to_numpy()
    attr = mean_attr * degrees

    attention_df = pd.DataFrame(attr,index=nodes, columns=nodes)

    li_important_pairs = [(attention_df.index[i], attention_df.columns[j]) for i,j in np.argwhere(attention_df.to_numpy() > args.att_thr)]
    li_important_nodes = sum(li_important_pairs,())

    nx_unionEdges = nx.from_pandas_edgelist(pd.DataFrame(li_important_pairs),source=0,target=1)
    lcc_nodes = max(nx.connected_components(nx_unionEdges),key=len)
    nx_unionEdges_lcc = nx_unionEdges.subgraph(lcc_nodes)
    nx.to_pandas_edgelist(nx_unionEdges_lcc).to_csv(args.out, sep='\t',index=False)


