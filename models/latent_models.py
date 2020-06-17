import pandas as pd
import numpy as np
import gensim

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
from tools import *
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

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

class ProductLatent(LatentModel):
    def __init__(self, encoding_dim=100, order_data=None, product_data=None, retrain=False, data_dir= './data/instacart_2017_05_01/', model_dir= './models/', verbose=0, plot=False):
        if verbose > 0:
            print("loading data...")

        self.encoding_dim = encoding_dim

        retrain = retrain or not os.path.isfile(model_dir + 'product2vec.model')

        train_orders = pd.read_csv(data_dir+"order_products__train.csv") if retrain else None
        prior_orders = pd.read_csv(data_dir+"order_products__prior.csv") if retrain else None
        products = pd.read_csv(data_dir+"products.csv").set_index('product_id')

        if verbose>0:
            print("getting model...")
        self.model = get_word2vec_model(train_orders, prior_orders, products, size=encoding_dim, retrain=retrain)

#        self.vocab = list(self.model.wv.vocab.keys())

        if plot:
            print("fitting pca...")
            pca = PCA(n_components=2)
            pca.fit(self.model.wv.syn0)


            print("plotting sample...")
            embeds = []
            labels = []
            for item in get_batch(vocab, model, n_batches=3):
                embeds.append(self.model[item])
                labels.append(products.loc[int(item)]['product_name'])
            embeds = np.array(embeds)
            embeds = pca.fit_transform(embeds)
            plot_with_labels(embeds, labels)


    def encode(self, product_ids):
        X_enc = []
        for pid in product_ids:
            if str(pid) in self.model.wv:
                X_enc.append(self.model.wv[str(pid)])
            else:
                X_enc.append(np.zeros(self.encoding_dim))

        return np.array(X_enc)




class UserLatentAEM(LatentModel):
    def __init__(self, input_dim=0, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def fit(self, train_dataset, epochs=5, batch_size=25):

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

