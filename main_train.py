from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import *
from torch.nn import functional as F
from collections import defaultdict
from importlib import import_module
import networkx as nx
import json
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=500,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')


#### added by ryy
parser.add_argument('--model', type=str, help='Model name: GCN/GAT/GPN/GraghSage', default='GAT')
num_repeat = 2  ### for debug use
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()



def train(class_selected, id_support, id_query, n_way, k_shot):
    model.train()
    optimizer.zero_grad()
    result = model(features, adj)
    embeddings, scores = result['emb'], result['score']
    z_dim = embeddings.size()[1]

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train

def test(class_selected, id_support, id_query, n_way, k_shot):
    model.eval()
    result = model(features, adj)
    embeddings, scores = result['emb'], result['score']
    z_dim = embeddings.size()[1]

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = (support_scores / torch.sum(support_scores, dim=1, keepdim=True)).unsqueeze(dim=-1)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':
    args.model = 'GraghSage'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    dataset = args.dataset
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)
    # adj = adj.to_dense()
    if args.model in ['GAT', 'GraghSage']:
        D = nx.DiGraph(adj.to_dense().numpy())
        edge_lst = nx.to_pandas_edgelist(D)
        edge_lst = [edge_lst['source'], edge_lst['target']]
        adj = torch.Tensor(edge_lst).long()
        del D, edge_lst
        
    # Model and optimizer
    m = import_module('models.' + args.model)
    model =  m.model(features.shape[1], args.hidden, args.dropout)
    # model =  m.model(features.shape[1], args.hidden)

    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        degrees = degrees.cuda()
    N_set=[5,10]
    K_set=[3,5]
    results=defaultdict(dict)
    meta_test_acc_total = np.zeros((5))
    meta_test_f1_total = np.zeros((5))
    for dataset in ['Amazon_clothing']:
        args.dataset = dataset
        for N in N_set:
            for K in K_set: # num of labeled nodes
                args.way = N
                args.shot = K
                print("Training %s for on %s (%d-way %d-shot)" % (args.model, args.dataset, N, K))
                meta_test_acc_total = np.zeros((5))
                meta_test_f1_total = np.zeros((5))
                for repeat in range(num_repeat):
                    print("Repeat %d: Training %s for on %s (%d-way %d-shot)" % (repeat, args.model, args.dataset, N, K))
                    n_way = args.way
                    k_shot = args.shot
                    n_query = args.qry
                    meta_test_num = 50
                    meta_valid_num = 50


                    # Sampling a pool of tasks for validation/testing
                    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
                    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

                    # Train model
                    t_total = time.time()
                    meta_train_acc = []
                    best_valid_acc = 0
                    for episode in range(args.episodes):
                        id_support, id_query, class_selected = \
                            task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)
                        acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
                        meta_train_acc.append(acc_train)
                        if episode > 0 and episode % 10 == 0:    
                            print("-------Episode {}-------".format(episode))
                            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

                            # validation
                            meta_test_acc = []
                            meta_test_f1 = []
                            for idx in range(meta_valid_num):
                                id_support, id_query, class_selected = valid_pool[idx]
                                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                                meta_test_acc.append(acc_test)
                                meta_test_f1.append(f1_test)
                            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                                        np.array(meta_test_f1).mean(axis=0)))
                            # testing
                            meta_test_acc = []
                            meta_test_f1 = []
                            for idx in range(meta_test_num):
                                id_support, id_query, class_selected = test_pool[idx]
                                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                                meta_test_acc.append(acc_test)
                                meta_test_f1.append(f1_test)
                            valid_acc = np.array(meta_test_acc).mean(axis=0)
                            if valid_acc > best_valid_acc:
                                # best_test_accs = temp_accs
                                best_valid_acc = valid_acc
                                meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                                meta_test_f1_total[repeat] = np.array(meta_test_f1).mean(axis=0)
                            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(meta_test_acc_total[repeat], meta_test_f1_total[repeat]))
                            # meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                            # meta_test_f1_total[repeat] = np.array(meta_test_f1).mean(axis=0)
                            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(meta_test_acc_total[repeat], meta_test_f1_total[repeat]))

                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                print("---------- F1 ------------")
                for repeat in range(num_repeat):
                    print(meta_test_f1_total[repeat])
                print("---------- Acc ------------")
                for repeat in range(num_repeat):
                    print(meta_test_acc_total[repeat])
                    results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]= meta_test_acc_total[repeat]

                    json.dump(results[dataset],open('./{}-result_{}.json'.format(args.model, dataset),'w'), indent=4) 


                accs=[]
                for repeat in range(num_repeat):
                    accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)])


            results[dataset]['{}-way {}-shot'.format(N,K)]=np.mean(accs)
            results[dataset]['{}-way {}-shot_print'.format(N,K)]='acc: {:.4f}'.format(np.mean(accs))

            json.dump(results[dataset],open('./{}-result_{}.json'.format(args.model, dataset),'w'), indent=4)   
    print("Done :)")