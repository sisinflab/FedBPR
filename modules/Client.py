import numpy as np
import random
from collections import defaultdict, deque


class Client:
    def __init__(self, client_id, model, train, train_user_list, sampler_size):
        self.id = client_id
        self.model = model
        self.train_set = train
        self.train_user_list = train_user_list
        self.sampler_size = sampler_size

    def predict(self, max_k):
        result = self.model.predict()
        result[list(self.train_user_list)] = -np.inf
        top_k = result.argsort()[-max_k:][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self, lr, positive_fraction):

        def operation(i, j):
            x_i = self.model.predict_one(i)
            x_j = self.model.predict_one(j)
            x_ij = x_i - x_j
            d_loss = 1 / (1 + np.exp(x_ij))

            wu = self.model.user_vec.copy()
            self.model.user_vec += lr * (d_loss * (hi - hj) - user_reg * wu)

            resulting_dic[j] = np.add(resulting_dic[j], d_loss * (-wu) - negative_item_reg * self.model.item_vecs[j])
            resulting_bias.update({j: resulting_bias[j] - d_loss - bias_reg * self.model.item_bias[j]})

            if positive_fraction:
                if random.random() >= 1 - positive_fraction:
                    resulting_dic[i] = np.add(resulting_dic[i],
                                              d_loss * wu - positive_item_reg * self.model.item_vecs[i])
                    resulting_bias.update({i: resulting_bias[i] + d_loss - bias_reg * self.model.item_bias[i]})

        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200
        resulting_dic = defaultdict(lambda: np.zeros(len(self.model.user_vec)))
        resulting_bias = defaultdict(float)

        sample = self.train_set.sample_user_triples()
        deque(map(lambda (i, j): operation(i, j), sample), maxlen=0)

        return resulting_dic, resulting_bias
