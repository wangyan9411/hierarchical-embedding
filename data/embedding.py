from gensim.models import Word2Vec
import numpy as np
import random
from collections import defaultdict as dd

def read_data():
    item_seqs = dd(list) 
    item_set = set()
    itemids = []
    with open('./ratings.dat') as file:
        for line in file.readlines():
            userid, itemid, rating, _ = line.strip().split('::')
            userid = int(userid)
            item_seqs[userid].append(itemid)
            item_set.add(itemid)
            itemids.append(int(itemid))
    item_num = max(itemids)
    return item_seqs.values(), item_set, item_num

class EmbeddingModel:

    def __init__(self, name="aminer"):
        self.model = None

    def train(self, data):
        self.model = Word2Vec(
            data, size=60, window=5, min_count=5, workers=20,
        )
        self.model.save('./item.model')

    def load(self):
        self.model = Word2Vec.load('./item.emb')
        return self.model

    def get_embedding(self, item_ids, item_num, idf=None):
        embed = np.zeros((item_num, 60))
        print (embed.shape)
        """
        weighted average of token embeddings
        :param tokens: input words
        :param idf: IDF dictionary
        :return: obtained weighted-average embedding
        """
        if self.model is None:
            self.load()
        for token in item_ids:
            if not token in self.model.wv:
                continue
            v = self.model.wv[token]
            embed[int(token)-1] = v
            
        return embed


if __name__ == '__main__':
    wf_name = 'aminer'
    # emb_model = EmbeddingModel.Instance()
    emb_model = EmbeddingModel()
    docs, item_set, item_num = read_data()
    emb_model.train(docs)
    np.save('./item.embed', emb_model.get_embedding(list(item_set), 4000))
