import numpy as np
from collections import defaultdict as dd

res_dir = './results/'
res_dir = './results_with_behav_soft/'
res_dir = './results_with_behav_pretrain/'
use_community = True
use_top = False

def load_data(epoch):
    uembeds = np.load(res_dir + 'user_' + epoch + '.npy') 
    iembeds = np.load(res_dir + 'item_' + epoch + '.npy') 
    assign = np.load(res_dir + 'assign_' + epoch + '.npy') 
    cembeds = np.load(res_dir + 'community_' + epoch + '.npy') 
    train = np.load('./data/train.npy')
    train = set(zip(train[0], train[1]))
    test = np.load('./data/test.npy')
    test = set(zip(test[0], test[1]))
    return uembeds, iembeds, assign, cembeds, train, test

def eval(epoch):
    u, i, assign, cembeds, train, test = load_data(epoch)
    # item_num = r.shape[1]
    # i = i[:item_num, :]
    pred_rating = np.dot(u, i.T)
    pred_rating /= np.linalg.norm(u,axis=1,keepdims=True)
    pred_rating /= np.linalg.norm(i.T,axis=0,keepdims=True)
    # community predicting
    
    if use_community:
        if use_top:
            u2c_embeds = np.zeros((assign.shape[0], cembeds.shape[1]))
            for uid in range(assign.shape[0]):
                cid = assign[uid].argsort()[-1]
                u2c_embeds[cid] = cembeds[cid]
        else:
            assign /= np.linalg.norm(assign, axis=1, keepdims=True)
            u2c_embeds = np.dot(assign, cembeds)
        pred_rating2 = np.dot(u2c_embeds, i.T)
        pred_rating2 /= np.linalg.norm(u2c_embeds, axis=1, keepdims=True)
        pred_rating2 /= np.linalg.norm(i.T, axis=0, keepdims=True)
        pred_rating += pred_rating2

    # some one has no test data (so should)
    test_user = dd(set)
    for u, i in test:
        test_user[u].add(i)
    
    recalls = 0.0
    for u in range(pred_rating.shape[0]):
        if u not in test_user:
            continue
        count = 0
        itemids = pred_rating[u].argsort()[-100:][::-1]
        pred = set()
        for i in itemids:
            if (u, i) in train:
                continue
            else:
                count += 1
            if count == 50:
                break 
            pred.add(i)
        recalls += float(len(pred.intersection(test_user[u]))) / len(test_user[u])
    recalls /= len(test_user)
    print recalls

epochs = [i for i in range(5)]
epochs = map(str, epochs)
for epoch in epochs:
    eval(epoch)
