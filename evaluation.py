import numpy as np

res_dir = './results/'
res_dir = './results_with_behav/'

def load_data(epoch):
    uembeds = np.load(res_dir + 'user_' + epoch + '.npy') 
    iembeds = np.load(res_dir + 'item_' + epoch + '.npy') 
    ratings = np.load('./data/rating.npy')
    return uembeds, iembeds, ratings

def eval(epoch):
    u, i, r = load_data(epoch)
    item_num = r.shape[1]
    i = i[:item_num, :]
    pred_rating = np.dot(u, i.T)
    square_error = np.sum(np.square(r-pred_rating))
    print (square_error)

epochs = [i for i in range(8)]
epochs = map(str, epochs)
for epoch in epochs:
    eval(epoch)
