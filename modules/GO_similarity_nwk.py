import argparse
import numpy as np
import pandas as pd
import networkx as nx
from itertools import permutations, product
from multiprocessing import Pool
import time 
import datetime

#def distance_in_GOtree(G, node1, node2):
#    def longest_path_length(G, n1, n2):
#        max_length = len(max(nx.all_simple_paths(G, n1, n2), key=lambda x: len(x)))
#        return max_length 
#    lca = nx.lowest_common_ancestor(G, node1, node2)
#    dist = longest_path_length(G,"GO:0008150",node1) + longest_path_length(G,"GO:0008150" ,node2)-2*longest_path_length(G,"GO:0008150",lca)
#    return dist



def distance_in_GOtree(G, node1, node2):
    if node1==node2:
        return 0
    else:
        lca = nx.lowest_common_ancestor(G, node1, node2)
        dist = nx.shortest_path_length(G,"GO:0008150",node1) + nx.shortest_path_length(G,"GO:0008150",node2) - 2*nx.shortest_path_length(G,"GO:0008150",lca)
        return dist
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--GOgraph',required=True)
    parser.add_argument('--gene2GO_annot',required=True)
    parser.add_argument('--templateNwk',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    
    start = time.time()
    # Load PPI nwk
    df_PPI = pd.read_csv(args.templateNwk, sep='\t', names=['source','target','weight'])
    nx_PPI = nx.from_pandas_edgelist(df_PPI)
    nodes_PPI = list(nx_PPI.nodes)

    # Load GO term hierarchical tree
    df_GOnwk = pd.read_csv(args.GOgraph, sep='\t')
    nx_GOnwk = nx.from_pandas_edgelist(df_GOnwk,create_using=nx.DiGraph())
   
    ## Filter GO terms
    df_gene2GO = pd.read_csv(args.gene2GO_annot, sep='\t',names=['gene','GO'])
    df_GO_filt = df_gene2GO.groupby('GO')['gene'].apply(list)
    df_GO_filt_overlap = df_GO_filt.apply(lambda x:len(set(x) & set(nodes_PPI)))
    GO_filt = list(df_GO_filt_overlap.loc[lambda x:x>1].index)

    ## Construct GO graph
    #nodes_connected = []
    #for node in list(nx_GOnwk.nodes):
    #    if node in GO_filt:
    #        if nx.has_path(nx_GOnwk, 'root', node):
    #            nodes_connected.append(node)
    #GO_graph = nx.subgraph(nx_GOnwk, nodes_connected)
    GO_graph = nx_GOnwk
    # Compute GO distances for all pair of genes
    def pairwise_dist(x):
        global nx_PPI
        global df_gene2GO
        global GO_graph
        g1, g2 = x

        GOs_1 = set(df_gene2GO.loc[lambda x:x.gene==g1,'GO'].unique()) & set(GO_graph.nodes)
        GOs_2 = set(df_gene2GO.loc[lambda x:x.gene==g2,'GO'].unique()) & set(GO_graph.nodes)
        
        if len(GOs_1)==0 or len(GOs_2)==0:
            return None 
        else:
            all_GO_combinations = product([GO_graph], GOs_1, GOs_2)
            distances = map(lambda x: distance_in_GOtree(x[0],x[1],x[2]), all_GO_combinations)
            mean_distance = np.array(list(distances)).mean()
            return (g1, g2, mean_distance)
    
    
    # Retrieve gene pairs in PPI network 
    nodes = list(set(nodes_PPI).intersection(set(df_gene2GO['gene'])))
    nodes_filt = [node for (node,val) in nx_PPI.degree if (val > 30) and (node in nodes)]


    df_gene_filt = df_gene2GO.groupby('gene')['GO'].apply(list)

    def filter_nodePairs(x):
        global nx_PPI
        global df_gene_filt
        n1, n2 = x 

        jaccard_tmp = len(set(df_gene_filt[n1]).intersection(set(df_gene_filt[n2])))/len(set(df_gene_filt[n1]).union(set(df_gene_filt[n2]))) 
        if nx_PPI.has_edge(n1,n2)==False:
            if jaccard_tmp > 0.5:
                if nx.shortest_path_length(nx_PPI, n1, n2) >= 2:
                    return (n1,n2)

    #node_pairs = [(n1,n2) for (n1, n2) in permutations(nodes_filt,2) if (nx_PPI.has_edge(n1,n2)==False) and len(set(df_gene_filt[n1]).intersection(set(df_gene_filt[n2])))/len(set(df_gene_filt[n1]).union(set(df_gene_filt[n2])))>0.5] 

    #node_pairs = [(n1,n2) for (n1, n2) in node_pairs if (nx.shortest_path_length(nx_PPI, n1, n2) >= 2)]
    
    with Pool(50) as p:
        node_pairs_iter = p.map(filter_nodePairs, permutations(nodes_filt,2))
        node_pairs = filter(None,list(node_pairs_iter))
    #out = list(filter(None, map(pairwise_dist,node_pairs)))
    
    with Pool(50) as p:
        out_iter = p.map(pairwise_dist,node_pairs)
        out = filter(None, list(out_iter))
    sec = time.time()-start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    times = times[0]
    print("elapsed_time: ",times)
   
    li_out=list(out)
    df_GOsim_nwk = pd.DataFrame(li_out,columns=["src","tgt","mean_distance"])
      
    df_GOsim_nwk['similarity']=1/(1+df_GOsim_nwk['mean_distance'])
    #df_GOsim_nwk.loc[:,['src','tgt','similarity']].to_csv(args.out,sep='\t',index=False, header=False)
    df_GOsim_nwk.loc[:,['src','tgt']].to_csv(args.out,sep='\t',index=False, header=False)
