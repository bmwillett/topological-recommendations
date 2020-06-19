import sys
sys.path.append('./lib')

from tools import *
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from base_model import BaseModel
import pandas as pd
import numpy as np
import pickle
import logging

log = logging.getLogger("TR_logger")

class GetAllModel(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, train_dataset):
        pass

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        preds = np.ones(len(test_labels))

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


class RandomModel(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, train_dataset):
        pass

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        preds = np.random.rand(len(test_labels))

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


class LogisticModel(BaseModel):
    def __init__(self, verbose=0):
        super().__init__()
        self.verbose=verbose

    def fit(self, train_dataset):

        prior_orders, labels = train_dataset.get_prior_products()

        df_train = get_features(train_dataset, prior_orders, verbose=self.verbose)

        X_train = df_train.to_numpy()

        self.model = LogisticRegression(random_state=0)
        self.model.fit(X_train, labels)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df_test = get_features(test_dataset, prior_orders, verbose=self.verbose)

        X_test = df_test.to_numpy()
        preds = self.model.predict(X_test)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels

class RandomForestModel(BaseModel):
    def __init__(self, verbose=0):
        super().__init__()
        self.verbose=verbose

    def fit(self, train_dataset):

        prior_orders, labels = train_dataset.get_prior_products()

        df_train = get_features(train_dataset, prior_orders, verbose=self.verbose)

        X_train = df_train.to_numpy()

        self.model = RandomForestRegressor(random_state=0)
        self.model.fit(X_train, labels)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df_test = get_features(test_dataset, prior_orders, verbose=self.verbose)

        X_test = df_test.to_numpy()
        preds = self.model.predict(X_test)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels

class LGBoostModel(BaseModel):
    def __init__(self, verbose=0):
        super().__init__()
        self.verbose=verbose

    def fit(self, train_dataset):

        prior_orders, labels = train_dataset.get_prior_products()

        df_train = get_features(train_dataset, prior_orders, verbose=self.verbose)

        self.d_train = lgb.Dataset(df_train,
                              label=labels,
                              categorical_feature=['aisle_id', 'department_id'])

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': 96,
            'max_depth': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5
        }

        ROUNDS = 100

        self.model = lgb.train(params, self.d_train, ROUNDS)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df_test = get_features(test_dataset, prior_orders, verbose=self.verbose)

        preds = self.model.predict(df_test)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


class XGBoostModel(BaseModel):
    def __init__(self, verbose=0):
        super().__init__()
        self.verbose=verbose

    def fit(self, train_dataset):

        prior_orders, labels = train_dataset.get_prior_products()

        df_train = get_features(train_dataset, prior_orders, verbose=self.verbose)

        # self.d_train = lgb.Dataset(df_train,
        #                       label=labels,
        #                       categorical_feature=['aisle_id', 'department_id'])

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': 96,
            'max_depth': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
            'bagging_freq': 5
        }

        ROUNDS = 100

        self.model = xgb.train(params, df_train, ROUNDS)

    def predict(self, test_dataset, getdf=False):

        prior_orders, test_labels = test_dataset.get_prior_products()

        df_test = get_features(test_dataset, prior_orders, verbose=self.verbose)

        preds = self.model.predict(df_test)

        if getdf:
            return preds, test_labels, prior_orders
        else:
            return preds, test_labels


