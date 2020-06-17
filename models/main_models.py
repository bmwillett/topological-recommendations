import sys
sys.path.append('./lib')
sys.path.append('./models')

from mapper_class import MapperClassifier
from latent_models import *

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pickle
import logging

log = logging.getLogger("TR_logger")

class BaseModel:
    def __init__(self):
        self.test_dataset = None
        self.X_pred = None

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pass

    def find_threshold(self, dataset, pts=20):
        totry = np.linspace(0,1,num=pts)
        best_t, best_f1 = 0, 0
        for t in totry:
            _,_,f1 = self.accuracy_test(dataset, threshold=t)
            if f1>best_f1:
                best_t, best_f1 = t, f1
        return best_t

    def accuracy_test(self, test_dataset, threshold=0.2):

        preds, test_labels, prior_orders = self.predict(test_dataset, getdf=True)

        user_true = {}
        user_pred = {}
        for i,row in enumerate(prior_orders.itertuples()):
            uid = row.user_id
            pid = row.product_id
            if uid not in user_true:
                user_true[uid], user_pred[uid] = [], []
            if test_labels[i] == 1:
                user_true[uid].append(pid)
            if preds[i] > threshold:
                user_pred[uid].append(pid)

        # TODO: add DCG if applicable
        precs, recs, f1s = [], [], []
        for uid in user_true:
            trues = set(user_true[uid])
            preds = set(user_pred[uid])

            tp = len(trues.intersection(preds))
            fp = len(preds) - tp
            fn = len(trues) - tp

            prec = tp/(tp+fp) if tp+fp>0 else 1
            rec = tp/(tp+fn) if tp+fn>0 else 1
            f1 = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0

            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

        return np.mean(precs), np.mean(recs), np.mean(f1s)


"""
Main non-topological model
"""

class UPLModel(BaseModel):
    def __init__(self, output_dim=1, user_latent_model = None, product_latent_model = None, h1_dim=50, h2_dim=50):
        super().__init__()

        if log.level==logging.DEBUG:
            print("creating UPLModel...")

        if user_latent_model is None:
            user_latent_model = UserLatentAEM()
        if product_latent_model is None:
            product_latent_model = ProductLatent()

        self.user_latent_model = user_latent_model
        self.product_latent_model = product_latent_model

        input_dim = self.user_latent_model.encoding_dim + self.product_latent_model.encoding_dim

        input = Input(shape=(input_dim,))
        h1 = Dense(h1_dim, activation='relu')(input)
        h2 = Dense(h2_dim, activation='relu')(h1)
        output = Dense(output_dim, activation='sigmoid')(h2)

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss= tf.keras.losses.BinaryCrossentropy())

    def fit(self, train_dataset, epochs=5, retrain_latent=True):

        if retrain_latent:
            if log.level == logging.DEBUG:
                print("fitting user latent model...")
            self.user_latent_model.fit(train_dataset)
            if log.level == logging.DEBUG:
                print("fitting product latent model...")
            self.product_latent_model.fit(train_dataset)

        prior_orders, labels = train_dataset.get_prior_products()
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(train_dataset, user_ids)
        product_latent = self.product_latent_model.encode(product_ids)

        X_train = np.concatenate( (user_latent, product_latent), axis=1)

        if log.level == logging.DEBUG:
            print("fitting final network...")
        self.model.fit(X_train, labels, epochs=epochs)


    def predict(self, test_dataset, getdf=False):
        prior_orders, test_labels = test_dataset.get_prior_products()
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(test_dataset, user_ids)
        product_latent = self.product_latent_model.encode(product_ids)

        X_test = np.concatenate((user_latent, product_latent), axis=1)

        preds = self.model.predict(X_test)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels



class TUPLModel(BaseModel):
    def __init__(self, output_dim=1, user_latent_model = None, product_latent_model = None, n_components=15, NRNN = 3, h1_dim=50, h2_dim=50):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating topological UPL model...")

        if user_latent_model is None:
            user_latent_model = UserLatentAEM()
        if product_latent_model is None:
            product_latent_model = ProductLatent()

        self.user_latent_model = user_latent_model
        self.product_latent_model = product_latent_model

        self.input_dim = self.user_latent_model.encoding_dim + self.product_latent_model.encoding_dim
        self.h1_dim, self.h2_dim =  h1_dim, h2_dim
        self.encoder_mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

    def fit(self, train_dataset, epochs=5, retrain_latent=True, batch_size=32):

        if retrain_latent:
            if log.level == logging.DEBUG:
                print("fitting user latent model...")
            self.user_latent_model.fit(train_dataset)
            if log.level == logging.DEBUG:
                print("fitting product latent model...")
            self.product_latent_model.fit(train_dataset)

        if log.level == logging.DEBUG:
            print("loading prior orders...")
        prior_orders, labels = train_dataset.get_prior_products()
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(train_dataset, user_ids)
        product_latent = self.product_latent_model.encode(product_ids)

        X_train = np.concatenate( (user_latent, product_latent), axis=1)

        if log.level == logging.DEBUG:
            print("fitting mapper model to latent data of size {} ...".format(X_train.shape))
        # run through mapper classifier to get graph-bin output
        X_map = self.encoder_mapper.fit(X_train, None)

        # finally, feed into neural network with two hidden layers and train to match X_train
        input_dim = X_map.shape[1]

        input = Input(shape=(input_dim,))
        h1 = Dense(self.h1_dim, activation='relu')(input)
        h2 = Dense(self.h2_dim, activation='relu')(h1)
        output = Dense(1, activation='sigmoid')(h2)

        self.mapper_model = Model(input, output)

        self.mapper_model.compile(optimizer='adadelta', loss=nonzero_loss)

        if log.level == logging.DEBUG:
            print("fitting final network...")
        self.mapper_model.fit(X_map, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True)


    def predict(self, test_dataset, getdf=False):
        prior_orders, test_labels = test_dataset.get_prior_products()
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(test_dataset, user_ids)
        product_latent = self.product_latent_model.encode(product_ids)

        X_test = np.concatenate((user_latent, product_latent), axis=1)

        # then project to graph bins using mapper
        X_test_map = self.encoder_mapper.project(X_test, None)

        # finally run through mapper model (NN after mapper)
        preds = self.mapper_model.predict(X_test_map)


        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels
