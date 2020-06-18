import pandas as pd
import numpy as np
import gensim
import sys
sys.path.append('./lib')
sys.path.append('./models')

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
from tools import *
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import NearestNeighbors

from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

# TODO: fix and implement product latent model
#  - get from dataset, not loading data
#  - set up as class and get model to acutally use it
#  - add tfidf or other NLP component
#  - add PCA

class LatentModel:
    def __init__(self):
        pass

    def fit(self, train_dataset):
        pass

    def plot(self, dataset, group_size=4, num_groups=3):

        product_ids = dataset.product_ids
        num_pids = len(product_ids)

        X = self.encode(product_ids)
        pca = PCA(n_components=2)
        pca.fit(X)

        neigh = NearestNeighbors(n_neighbors=group_size, radius=0.4)
        neigh.fit(X)

        labels = []
        embeds = []
        i = 0
        while i < num_groups:
            pid = np.random.randint(num_pids)
            if X[pid, 0] == 0:
                continue
            closest = neigh.kneighbors([X[pid, :]])[1][0, :]
            new_embeds = X[closest]

            new_labels = dataset.product_data.feature1.iloc[closest].values
            labels += list(new_labels)
            embeds += list(new_embeds)
            i += 1

        low_dim_embs = pca.transform(np.array(embeds))

        plt.figure(figsize=(10, 10))  # in inches
        texts = []
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            texts.append(plt.text(x, y, label, size=12))
            # xytext=(5, 2),
            # textcoords='offset points',
            # ha='right',
            # va='bottom',
            # size=15))

        #     plt.savefig(filename)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        plt.show()

class ProductLatent(LatentModel):
    def __init__(self, n_components=10):
        self.w2v = word2vecModel()
        self.tfidf = TFIDFModel()
        self.n_components = n_components

    def fit(self, dataset):
        self.w2v.fit(dataset)
        self.tfidf.fit(dataset)

        product_ids = dataset.product_ids
        X1 = self.w2v.encode(product_ids)
        X2 = self.tfidf.encode(product_ids)

        X = np.concatenate([X1,X2],axis=1)

        self.pcamodel = PCA(n_components=self.n_components)
        self.pcamodel.fit(X)


    def encode(self, product_ids):
        X1 = self.w2v.encode(product_ids)
        X2 = self.tfidf.encode(product_ids)
        X = np.concatenate([X1, X2],axis=1)
        return self.pcamodel.transform(X)


class word2vecModel(LatentModel):
    def __init__(self, encoding_dim=100, retrain=False, model_dir= './models/'):
        self.encoding_dim = encoding_dim
        self.model = None
        self.retrain = retrain
        self.model_dir = model_dir

    def fit(self,dataset):
        retrain = self.retrain or not os.path.isfile(self.model_dir + 'product2vec.model')

        self.model = get_word2vec_model(dataset.order_data, size=self.encoding_dim, retrain=retrain, model_dir=self.model_dir)

#        self.vocab = list(self.model.wv.vocab.keys())

    def encode(self, product_ids):
        X_enc = []
        for pid in product_ids:
            if str(pid) in self.model.wv:
                X_enc.append(self.model.wv[str(pid)])
            else:
                X_enc.append(np.zeros(self.encoding_dim))

        return np.array(X_enc)


class TFIDFModel(LatentModel):
    def __init__(self):
        pass

    def fit(self, dataset, n_components=20):
        self.dataset=dataset

        comb_frame = dataset.product_data.feature1.str.cat(" " + dataset.product_data.feature3.str.cat(" " + dataset.product_data.feature5))
        comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
        comb_frame.head()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer.fit(comb_frame)

        X = self.vectorizer.transform(comb_frame)

        self.tsvd = TruncatedSVD(n_components=n_components)
        self.tsvd.fit(X)

    def encode(self, product_ids):
        df = self.dataset.product_data.loc[product_ids]

        comb_frame = df.feature1.str.cat(" " + df.feature3.str.cat(" " + df.feature5))
        comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
        comb_frame.head()
        X_test = self.vectorizer.transform(comb_frame)

        return self.tsvd.transform(X_test)


class UserLatentAEM(LatentModel):
    def __init__(self, input_dim=0, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def fit(self, train_dataset, epochs=50, batch_size=25):

        self.input_dim = len(train_dataset.product_ids)
        input_dim = self.input_dim
        encoding_dim = self.encoding_dim

        input = Input(shape=(input_dim,))

        encoded = Dense(encoding_dim, activation='relu')(input)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input, decoded)
        self.encoder = Model(input, encoded)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.autoencoder.compile(optimizer='adadelta', loss=nonzero_loss)


        X_train = train_dataset.get_user_product_matrix()
        self.train_dataset=train_dataset

        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True)

    def encode(self, dataset, user_ids):
        X = dataset.get_user_product_matrix()
        ids = tuple([dataset.user_idx[uid] for uid in user_ids])
        X_to_enc = X[ids,:]
        encoded_users = self.encoder.predict(X_to_enc)

        return encoded_users

