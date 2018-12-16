# user feature: user ordinary features, behavior feature on frequent item list

# adjacency: user similarity in behaviors

# 0-based

import numpy as np
import random

def read_user_behavior_data():
    uids = []
    iids = [] 
    with open('./ratings.dat') as file:
        for line in file.readlines():
            userid, itemid, rating, _ = line.strip().split('::')
            userid, itemid = int(userid), int(itemid)
            uids.append(userid-1)
            iids.append(itemid-1)
        # assume max id = num
        user_num = max(uids) + 1
        item_num = max(iids) + 1
        print (user_num, item_num)
        rating_matrix = np.zeros((user_num, item_num))
        rating_matrix[uids, iids] = 1
        
    return rating_matrix, uids, iids

def read_user_features():
    with open('./users.dat') as file:
        lines = file.readlines()
        user_num = len(lines)
        user_feature = np.zeros((user_num, 4))
        user_num = len(lines)
        for line in lines:
            uid, gender, age, occu, _ = line.strip().split('::')
            uid = int(uid) - 1
            gender = 0 if gender == 'F' else 1
            age = int(age)/10 if age != '56' else 6
            occu = int(occu)
            user_feature[uid] = [uid, gender, age, occu]
    return user_feature

def adjacency(ratings):
    # cosine similarity
    user_num = ratings.shape[0] 
    u2u_sim = np.dot(ratings, ratings.T)
    threshold = 20
    edges = np.where(u2u_sim > threshold)
    print ('edges shape', len(edges[0]))
    adj = np.zeros((user_num, user_num))
    adj[edges[0], edges[1]] = 1
    diag = [i for i in range(user_num)]
    adj[diag] = 1
    return adj

def construct_user_behavior_feature(ratings, feature_num):
    # select the most frequent items
    item_freq = np.sum(ratings, 0)
    index = item_freq.argsort()[-feature_num:]
    return ratings[:,index]
    

def gen_user_feature():
    user_feature = read_user_features()
    ratings, uids, iids = read_user_behavior_data()
    user_behav_feature = construct_user_behavior_feature(ratings, 300)
    print ('user behav feature', user_behav_feature.shape)
    cat_feature = np.concatenate([user_feature, user_behav_feature], axis = 1)
    print ('cat feature ', cat_feature.shape)
    adj = adjacency(ratings)
    # split train test
    train_ratio = 0.9
    test_ratio = 0.9
    uituples = zip(uids, iids)
    random.shuffle(uituples)
    train = uituples[:int(train_ratio*len(uituples))]
    test = uituples[int(train_ratio*len(uituples)):]
    uids = [i for i,j in train]
    iids = [j for i,j in train]
    label = np.array([uids, iids])
    print (user_feature.shape, adj.shape, label.shape)
    np.save('./user_feature_with_behav.npy', cat_feature)
    np.save('./rating.npy', ratings)
    np.save('./adj.npy', adj)
    np.save('./train.npy', label)

    uids = [i for i,j in test]
    iids = [j for i,j in test]
    label = np.array([uids, iids])
    print (label.shape)
    np.save('./test.npy', label)

gen_user_feature()
