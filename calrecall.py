import numpy as np

res_dir = './results/'
res_dir = './results_only_user/'

def load_data(epoch):
    uembeds = np.load(res_dir + 'user_' + epoch + '.npy') 
    iembeds = np.load(res_dir + 'item_' + epoch + '.npy') 
    train = np.load('./data/train.npy')
    train = set(zip(train[0], train[1]))
    test = np.load('./data/test.npy')
    test = set(zip(test[0], test[1]))
    return uembeds, iembeds, train, test

def eval(epoch):
    u, i, train, test = load_data(epoch)
    # item_num = r.shape[1]
    # i = i[:item_num, :]
    pred_rating = np.dot(u, i.T)
    pred_rating /= np.linalg.norm(u,axis=1,keepdims=True)
    pred_rating /= np.linalg.norm(i.T,axis=0,keepdims=True)
    pred = set()
    for u in range(pred_rating.shape[0]):
        count = 0
        itemids = pred_rating[u].argsort()[-100:][::-1]
        for i in itemids:
            if (u, i) in train:
                continue
            else:
                count += 1
            if count == 50:
                break 
            pred.add((u,i))
    recall = float(len(pred.intersection(test)))/len(test)
    print recall

epochs = [i for i in range(4)]
epochs = map(str, epochs)
for epoch in epochs:
    eval(epoch)
