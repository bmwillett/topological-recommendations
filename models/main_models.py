import sys
sys.path.append('./lib')
sys.path.append('./models')

from base_model import BaseModel
from latent_models import *
from mapper_class import MapperClassifier
from tools import get_features

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import pickle
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger("TR_logger")


"""
Main non-topological model
"""

class UPLModel(BaseModel):
    def __init__(self, dataset, output_dim=1, user_latent_model = None, product_latent_model = None, h1_dim=50, h2_dim=50):
        super().__init__()

        if log.level==logging.DEBUG:
            print("creating UPLModel...")

        if user_latent_model is None:
            user_latent_model = UserLatentAEM()
        if product_latent_model is None:
            product_latent_model = ProductLatent(dataset)

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

        preds = self.model.predict(X_test).reshape

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


class TUPLModel(BaseModel):
    def __init__(self, dataset, output_dim=1, user_latent_model = None, product_latent_model = None, n_components=15, NRNN = 3, h1_dim=50, h2_dim=50):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating topological UPL model...")

        if user_latent_model is None:
            user_latent_model = UserLatentAEM()
        if product_latent_model is None:
            product_latent_model = ProductLatent(dataset)

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

        self.mapper_model.compile(optimizer='adadelta', loss= tf.keras.losses.BinaryCrossentropy())

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

"""
NNet using same features as baseline models
"""
class FNetModel(BaseModel):
    def __init__(self, h_dims=(50, 50), output_dim=1):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating FNetModel...")

        self.h_dims, self.output_dim = h_dims, output_dim
        self.verbose = 0*int(log.level == logging.DEBUG)
        self.model, self.input_dim = None, None

    def fit(self, train_dataset, epochs=5):

        prior_orders, labels = train_dataset.get_prior_products()
        df = get_features(train_dataset, prior_orders, verbose=self.verbose)

        # # apply one-hot encoding on categorical features and get array
        # X_train = df.join(pd.get_dummies(df['aisle_id'], prefix='aid')).join(
        #     pd.get_dummies(df['department_id'], prefix='did')).drop(columns=['aisle_id', 'department_id']).to_numpy()

        # drop categorical features
        X_train = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        self.input_dim = X_train.shape[1]

        input = Input(shape=(self.input_dim,))
        layers = [input]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        if log.level == logging.DEBUG:
            print("fitting network...")

        self.model.fit(X_train, labels, epochs=epochs)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df = get_features(test_dataset, prior_orders, verbose=self.verbose)

        X_test = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        preds = self.model.predict(X_test).squeeze()

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


"""
 FNet with topological features
"""
class FNetModel_top(BaseModel):
    def __init__(self, h_dims=(50, 50), output_dim=1, n_components=5, NRNN=3):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating FNetModel...")

        self.h_dims, self.output_dim = h_dims, output_dim
        self.verbose = 0*int(log.level == logging.DEBUG)
        self.model, self.input_dim = None, None
        self.encoder_mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

    def fit(self, train_dataset, epochs=5, retrain_mapper=True):

        prior_orders, labels = train_dataset.get_prior_products()
        df = get_features(train_dataset, prior_orders, verbose=self.verbose)

        # # apply one-hot encoding on categorical features and get array
        # X_train = df.join(pd.get_dummies(df['aisle_id'], prefix='aid')).join(
        #     pd.get_dummies(df['department_id'], prefix='did')).drop(columns=['aisle_id', 'department_id']).to_numpy()

        # drop categorical features
        X_train = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        # send X_train through mapper-classifier
        if retrain_mapper:
            self.X_map = self.encoder_mapper.fit(X_train, None)

        self.input_dim = self.X_map.shape[1]

        input = Input(shape=(self.input_dim,))
        layers = [input]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        if log.level == logging.DEBUG:
            print("fitting network...")

        self.model.fit(self.X_map, labels, epochs=epochs)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df = get_features(test_dataset, prior_orders, verbose=self.verbose)

        X_test = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        X_test_map = self.encoder_mapper.project(X_test, None)

        preds = self.model.predict(X_test_map)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels

"""
NNet using features and latent vectors
"""

class LFNetModel(BaseModel):
    def __init__(self, h_dims=(50, 50), output_dim=1, user_latent_model=None, product_latent_model=None):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating LFNetModel...")

        if user_latent_model is None:
            user_latent_model = EmptyLatentModel()
        if product_latent_model is None:
            product_latent_model = EmptyLatentModel()

        self.user_latent_model = user_latent_model
        self.product_latent_model = product_latent_model

        self.h_dims, self.output_dim = h_dims, output_dim
        self.verbose = 0 * int(log.level == logging.DEBUG)
        self.model, self.input_dim = None, None

    def fit(self, train_dataset, epochs=5):

        prior_orders, labels = train_dataset.get_prior_products()
        df = get_features(train_dataset, prior_orders, verbose=self.verbose)

        # # apply one-hot encoding on categorical features and get array
        # X_train = df.join(pd.get_dummies(df['aisle_id'], prefix='aid')).join(
        #     pd.get_dummies(df['department_id'], prefix='did')).drop(columns=['aisle_id', 'department_id']).to_numpy()

        # drop categorical features
        _X_train = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        # add latent vectors
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, train_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        # merge all vectors to get training matrix
        X_train = np.concatenate([_X_train, user_latent, product_latent], axis=1)

        # create model
        self.input_dim = X_train.shape[1]

        input = Input(shape=(self.input_dim,))
        layers = [input]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        if log.level == logging.DEBUG:
            print("fitting network...")

        self.model.fit(X_train, labels, epochs=epochs)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df = get_features(test_dataset, prior_orders, verbose=self.verbose)

        _X_test = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, test_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        X_test = np.concatenate([_X_test, user_latent, product_latent], axis=1)

        preds = self.model.predict(X_test).squeeze()

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


"""
NNet using features and latent vectors and topological encoding
"""
class TLFNetModel(BaseModel):
    def __init__(self, h_dims=(50, 50), output_dim=1, user_latent_model=None, product_latent_model=None, n_components=5, NRNN=3, bypass=True):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating TLFNetModel...")

        if user_latent_model is None:
            user_latent_model = EmptyLatentModel()
        if product_latent_model is None:
            product_latent_model = EmptyLatentModel()

        self.user_latent_model = user_latent_model
        self.product_latent_model = product_latent_model

        self.h_dims, self.output_dim, self.bypass = h_dims, output_dim, bypass
        self.verbose = int(log.level == logging.DEBUG)
        self.model, self.input_dim = None, None

        self.encoder_mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

    def fit(self, train_dataset, epochs=5, retrain_mapper=True):

        prior_orders, labels = train_dataset.get_prior_products()
        df = get_features(train_dataset, prior_orders, verbose=self.verbose-1)

        # # apply one-hot encoding on categorical features and get array
        # X_train = df.join(pd.get_dummies(df['aisle_id'], prefix='aid')).join(
        #     pd.get_dummies(df['department_id'], prefix='did')).drop(columns=['aisle_id', 'department_id']).to_numpy()

        # drop categorical features
        _X_train = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        # add latent vectors
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, train_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        # merge all vectors to get training matrix
        X_train = np.concatenate([_X_train, user_latent, product_latent], axis=1)

        # send X_train through mapper-classifier
        if retrain_mapper:
            self.X_map = self.encoder_mapper.fit(X_train, None)

        if self.bypass:
            self.X_map = np.concatenate([X_train, self.X_map], axis=1)

        self.input_dim = self.X_map.shape[1]

        if self.verbose>0:
            print("created mapper encoding of size self.X_map.shape[1]")

        # create model
        self.input_dim = self.X_map.shape[1]

        input = Input(shape=(self.input_dim,))
        layers = [input]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        if log.level == logging.DEBUG:
            print("fitting network...")

        self.model.fit(self.X_map, labels, epochs=epochs)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df = get_features(test_dataset, prior_orders, verbose=self.verbose-1)

        _X_test = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, test_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        X_test = np.concatenate([_X_test, user_latent, product_latent], axis=1)

        X_test_map = self.encoder_mapper.project(X_test, None)

        if self.bypass:
            X_test_map = np.concatenate([X_test, X_test_map], axis=1)

        preds = self.model.predict(X_test_map).squeeze()

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


"""
NNet using features and latent vectors and topological encoding
 - only user features encoded topologicall
"""
class TULFNetModel(BaseModel):
    def __init__(self, h_dims=(50, 50), output_dim=1, user_latent_model=None, product_latent_model=None, n_components=5, NRNN=3, bypass=True):
        super().__init__()

        if log.level == logging.DEBUG:
            print("creating TULFNetModel...")

        if user_latent_model is None:
            user_latent_model = EmptyLatentModel()
        if product_latent_model is None:
            product_latent_model = EmptyLatentModel()

        self.user_latent_model = user_latent_model
        self.product_latent_model = product_latent_model

        self.h_dims, self.output_dim, self.bypass = h_dims, output_dim, bypass
        self.verbose = int(log.level == logging.DEBUG)
        self.model, self.input_dim = None, None

        self.encoder_mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

    def fit(self, train_dataset, epochs=5, retrain_mapper=True):

        prior_orders, labels = train_dataset.get_prior_products()
        df = get_features(train_dataset, prior_orders, verbose=self.verbose-1)

        # # apply one-hot encoding on categorical features and get array
        # X_train = df.join(pd.get_dummies(df['aisle_id'], prefix='aid')).join(
        #     pd.get_dummies(df['department_id'], prefix='did')).drop(columns=['aisle_id', 'department_id']).to_numpy()

        # drop categorical features
        _X_train = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        # add latent vectors
        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, train_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        # merge all vectors to get training matrix
        X_train = np.concatenate([_X_train, user_latent, product_latent], axis=1)

        # get user specific features for encoder
        _X_train_user = df[['user_total_orders','user_total_items','total_distinct_items','user_average_basket']]
        X_train_user = np.concatenate([_X_train_user,user_latent], axis=1)

        # reduce to unique values
        uidx = {}
        for i, uid in enumerate(user_ids):
            if uid not in uidx:
                uidx[uid] = i

        exp = {}
        for i, idx in enumerate(uidx):
            exp[idx] = i

        X_train_user_red = X_train_user[[uidx[uid] for uid in uidx]]

        if log.level == logging.DEBUG:
            print("reducing from X_train_user.shape={} to X_train_user_red.shape={} for mapping...".format(X_train_user.shape,X_train_user_red.shape))

        # send X_train through mapper-classifier
        if retrain_mapper:
            self.X_map_red = self.encoder_mapper.fit(X_train_user_red, None)

        self.X_map = np.array([self.X_map_red[exp[uid],::] for uid in user_ids])

        if log.level == logging.DEBUG:
            print("expanded from self.X_map_red.shape={} to self.X_map.shape={}...".format(self.X_map_red.shape,
                                                                                           self.X_map.shape))

        if self.bypass:
            if log.level == logging.DEBUG:
                print("combining X_train.shape={} and self.X_map.shape={}...".format(X_train.shape,
                                                                                    self.X_map.shape))
            self.X_map = np.concatenate([X_train, self.X_map], axis=1)
            if log.level == logging.DEBUG:
                print("obtained self.X_map.shape={}".format(self.X_map.shape))



        self.input_dim = self.X_map.shape[1]

        if self.verbose>0:
            print("created mapper encoding of size self.X_map.shape[1]")

        # create model
        self.input_dim = self.X_map.shape[1]

        input = Input(shape=(self.input_dim,))
        layers = [input]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.model = Model(input, output)

        self.model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        if log.level == logging.DEBUG:
            print("fitting network...")

        self.model.fit(self.X_map, labels, epochs=epochs)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df = get_features(test_dataset, prior_orders, verbose=self.verbose-1)

        _X_test = df.drop(columns=['aisle_id', 'department_id']).to_numpy()

        user_ids = prior_orders['user_id'].values
        product_ids = prior_orders['product_id'].values

        user_latent = self.user_latent_model.encode(user_ids, test_dataset)
        product_latent = self.product_latent_model.encode(product_ids)

        X_test = np.concatenate([_X_test, user_latent, product_latent], axis=1)

        # get user specific features for encoder
        _X_test_user = df[['user_total_orders','user_total_items','total_distinct_items','user_average_basket']]
        X_test_user = np.concatenate([_X_test_user,user_latent], axis=1)

        # reduce to unique values
        uidx = {}
        for i, uid in enumerate(user_ids):
            if uid not in uidx:
                uidx[uid] = []
            uidx[uid].append(i)
        exp = {}
        for i, idx in enumerate(uidx):
            exp[idx] = i

        X_test_user_red = X_test_user[[uidx[uid][0] for uid in uidx]]

        X_test_map_red = self.encoder_mapper.project(X_test_user_red, None)

        X_test_map = np.array([X_test_map_red[exp[uid], ::] for uid in user_ids])

        if self.bypass:
            X_test_map = np.concatenate([X_test, X_test_map], axis=1)

        preds = self.model.predict(X_test_map).squeeze()

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels

