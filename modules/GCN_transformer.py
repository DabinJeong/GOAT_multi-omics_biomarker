#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import argparse
import json
from copy import deepcopy
#### GCN #################
import os.path as osp
import os

import random

import networkx as nx

import torch
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss
import torch_geometric
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
import torch_geometric.nn as geom_nn
from torch.utils.data import random_split
from torch_geometric.nn import GATConv, GraphConv, TransformerConv

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
use_gpu = torch.cuda.is_available()


# Evaluation_metric
def calculate_accuracy(y_pred, y):
    from sklearn import metrics
    fpr,tpr,thres = metrics.roc_curve(y,y_pred,pos_label=1)
    idx = np.argmax(tpr - fpr)
    top_pred = (y_pred >thres[idx]).astype(np.int64)
    correct = (top_pred == y).astype(np.int64).sum()
    acc = correct.astype(np.int64) / len(y)
    return acc


### Define model
class GCNNet(torch.nn.Module):
        #def __init__(self, input_dim, output_dim, num_nodes, conv_feat_dim, dropout_rate):
        def __init__(self, input_dim, output_dim, num_nodes, conv_feat_dim):
            super(GCNNet, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_nodes = num_nodes
            self.conv_feat_dim = conv_feat_dim
            
            self.layer_cell = 1
            self.conv1 = torch_geometric.nn.TransformerConv(self.input_dim, self.conv_feat_dim,heads=3,beta=False)
            
        def forward(self, data):
            # input Data Object
            x, edge_index, batch = data.x, data.edge_index, data.batch

            ## GATConv layers
            x, (att_edge_idx, att_weights) = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.leaky_relu(x)

            return x, att_edge_idx, att_weights
        
        def get_embedding(self, data):
            # input Data Object
            x, edge_index, batch = data.x, data.edge_index, data.batch

            ## GATConv layers
            x, (edge_idx, attention_weights)  = self.conv1(x, edge_index)
            x = F.leaky_relu(x)
            embedding = x.clone().detach()
            
            return x, embedding, attention_edge_idx, attention_weights

        def grad_cam(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i in range(self.layer_cell):
                x = F.relu(self.conv1(x, edge_index)[0])
                if i == 0:
                    node = x
                    node.retain_grad()
            return node


class subtype_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, conv_feat_dim, dropout_rate):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.conv_feat_dim = conv_feat_dim
        self.dropout_rate = dropout_rate
        
        self.node_embed = GCNNet(self.input_dim, self.output_dim, self.num_nodes, self.conv_feat_dim)
        self.classification = torch.nn.Sequential (
                torch.nn.Linear(self.num_nodes*self.conv_feat_dim, 64),
                torch.nn.Dropout(p=self.dropout_rate),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64,256),
                torch.nn.Dropout(p=self.dropout_rate),
                torch.nn.LeakyReLU())
        self.last_layer = torch.nn.Linear(256,1)
        
    def forward(self, data):
        x, att_edge_idx, att_weights = self.node_embed(data)
        x = x.view(-1, self.num_nodes*self.conv_feat_dim)
        x = self.classification(x)
        x = self.last_layer(x) 
        return x, att_edge_idx, att_weights
    
    def embedding(self, data):
        x, att_edge_idx, att_weights = self.node_embed(data)
        x = x.view(-1, self.num_nodes*self.conv_feat_dim)
        #classifier_exclude_lastLayer = torch.nn.Sequential(*list(self.classification.children())[:-1])
        x = self.classification(x)
        return x
        
########################################################################################################################
def train_graph(model, optimizer, criterion, train_loader, device):
    model.train()
    optimizer.zero_grad()

    total_loss_train = 0
    total_acc = 0

    y_li , true_y_li = [],[]
    for data in train_loader:

        data = data.to(device)
        out, att_edge_idx, att_weights = model(data) 

        y = out.cpu().detach().flatten().tolist()
        true_y = data.y.cpu().detach().flatten().tolist()

        loss = criterion(torch.squeeze(out.to(torch.float32)), torch.squeeze(data.y.to(torch.float32)))

        y_li.extend(y)
        true_y_li.extend(true_y)        

        total_loss_train += loss.item()
        loss.backward()
        optimizer.step()

        
    train_acc = calculate_accuracy(y_li,true_y_li)   
    train_loss = criterion(torch.FloatTensor(y_li), torch.FloatTensor(true_y_li)).item()
    
    return model,train_loss,train_acc

def validate_graph(model, optimizer, criterion, val_loader, device):
    model.eval()
    optimizer.zero_grad()

    total_loss_val = 0
    total_acc = 0

    y_li , true_y_li = [],[]
    for data in val_loader:

        data = data.to(device)
        out, att_edge_idx, att_weights = model(data) 

        y = out.cpu().detach().flatten().tolist()
        true_y = data.y.cpu().detach().flatten().tolist()

        y_li.extend(y)
        true_y_li.extend(true_y)

    val_loss = criterion(torch.FloatTensor(y_li), torch.FloatTensor(true_y_li)).item()
    val_acc = calculate_accuracy(y_li,true_y_li) 
    
    return val_loss, val_acc

def test_graph(model, optimizer, criterion, test_loader, device):
    model.eval()
    optimizer.zero_grad()

    total_loss_eval = 0
    total_acc = 0

    y_li , true_y_li = [],[]
    att_li = []
    for data in test_loader:
        data = data.to(device)
        out, att_edge_idx, att_weights = model(data) ######
        
        y = out.cpu().detach().flatten().tolist()
        true_y = data.y.cpu().detach().flatten().tolist()
 
        att_li.append(att_weights.cpu().detach().numpy())
        y_li.extend(y)
        true_y_li.extend(true_y)
        
    eval_loss = criterion(torch.FloatTensor(y_li), torch.FloatTensor(true_y_li)).item()
    eval_acc = calculate_accuracy(y_li,true_y_li)    
    
    att_weights = sum(att_li)/len(test_loader)

    return eval_loss, eval_acc, att_weights

def experiment_graph(args, graph_train_loader, graph_val_loader, graph_test_loader):
    """
    Return model with minimum validation loss (epoch)
    """
    num_nodes = graph_train_loader.dataset[0].x.size()[0]
    dim_node_features = graph_train_loader.dataset[0].x.size()[1]
    
    device = args.device
    
    model = subtype_classifier(dim_node_features,1,num_nodes,8,args.dropout_rate).to(device)
    
    # === Loss === #
    criterion = torch.nn.BCEWithLogitsLoss()

    # === Optimizer === #
    optimizer = torch.optim.Adam([
            {'params': model.parameters()}
        ], lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # # === Scheduler === #
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    ##################################################################################################################
    # ====== Cross Validation Best Performance Dict ====== #
    best_performances = {}
    best_performances['best_epoch'] = 0
    best_performances['best_train_loss'] = float('inf')
    best_performances['best_train_acc'] = 0
    best_performances['best_valid_loss'] = float('inf')
    best_performances['best_valid_acc'] = 0
    # ==================================================== #
    
    list_model = []
    list_optimizer = [] 
    list_train_epoch_loss = []
    list_epoch_acc = []

    list_val_epoch_loss = []
    list_val_epoch_acc = []

    best_model = None 
    best_model_idx = None 
    for epoch in range(args.epochs):
        
        # ====== TRAIN Epoch ====== #
        model, train_loss, train_acc = train_graph(model, optimizer, criterion, graph_train_loader, device)

        list_model.append(deepcopy(model))
        list_optimizer.append(deepcopy(optimizer))
        list_train_epoch_loss.append(train_loss)
        list_epoch_acc.append(train_acc)
        
        # ====== VALID Epoch ====== #
        val_loss, val_acc = validate_graph(model, optimizer, criterion, graph_val_loader, device)
        
        list_val_epoch_loss.append(val_loss)
        list_val_epoch_acc.append(val_acc)
        
        if val_loss < best_performances['best_valid_loss']:
            best_performances['best_epoch'] = epoch
            best_performances['best_train_loss'] = train_loss
            best_performances['best_train_acc'] = train_acc
            best_performances['best_valid_loss'] = val_loss
            best_performances['best_valid_acc'] = val_acc
            best_model = model
            best_model_idx = epoch
        print(('Epoch: {:03d}, Train_loss: {:.4f}, Train_acc: {:.3f}%, Val_loss: {:.3f}, Val_acc: {:.3f},%').format(epoch, train_loss, train_acc*100, val_loss, val_acc*100))
   
    test_loss, test_acc, attention_weights  = test_graph(best_model, optimizer, criterion, graph_test_loader, device)
    return best_model, best_performances, test_loss, test_acc
