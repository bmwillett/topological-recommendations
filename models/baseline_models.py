import sys
sys.path.append('./lib')

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




def get_features(dataset, prior_orders, verbose=-1):
    """
    further preprocessing on data to extract following features:

    'user_total_orders', 'user_total_items', 'total_distinct_items','user_average_basket',
    'aisle_id', 'department_id', 'product_orders', 'product_reorders',
    'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
    'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',

    returns: DataFrame with above columns, and rows equal to (user,product) pairs, where
    products are chosen from those user has previously purchased


    """

    order_data = dataset.order_data
    product_data = dataset.product_data

    if verbose==-1:
        verbose = int(log.level == logging.DEBUG)

    if verbose > 0:
        print('computing product f')

    # TODO: extract reorder count and rate (or load from file)
    prods_ = pd.DataFrame()
    prods_['orders'] = order_data.groupby(order_data.product_id).size().astype(np.int32)
    product_data = product_data.join(prods_, on='product_id')

    del prods_

    if verbose > 0:
        print('add order info to priors')


    ### user features

    usr_ = pd.DataFrame()
    usr_['nb_orders'] = order_data.groupby('user_id').size().astype(np.int16)
    users_ = pd.DataFrame()
    users_['total_items'] = order_data.groupby('user_id').size().astype(np.int16)
    users_['all_products'] = order_data.groupby('user_id')['product_id'].apply(set)
    users_['total_distinct_items'] = (users_.all_products.map(len)).astype(np.int16)
    users_ = users_.join(usr_)
    del usr_
    users_['average_basket'] = (users_.total_items / users_.nb_orders).astype(np.float32)

    ### userXproduct features

    if verbose > 0:
        print('compute userXproduct f - this is long...')
    order_data['user_product'] = list(zip(order_data.product_id, order_data.user_id))

    d_ = dict()
    for row in order_data.itertuples():
        z = row.user_product
        if z not in d_:
            d_[z] = (1,
                    row.order_number,
                    row.add_to_cart_order)
        else:
            d_[z] = (d_[z][0] + 1,
                max(d_[z][1], row.order_number),
                d_[z][2] + row.add_to_cart_order)


    if verbose > 0:
        print('to dataframe')

    userXproduct_ = pd.DataFrame.from_dict(d_, orient='index')
    del d_

    userXproduct_.columns = ['nb_orders', 'num_orders', 'sum_pos_in_cart']
    userXproduct_.nb_orders = userXproduct_.nb_orders.astype(np.int16)
    userXproduct_.sum_pos_in_cart = userXproduct_.sum_pos_in_cart.astype(np.int16)

    if verbose > 0:
        print('user X product f: ', len(userXproduct_))

    # construct user-product list with features
    df = prior_orders.copy()

    if verbose > 0:
        print('user related features')
    df['user_total_orders'] = df.user_id.map(users_.nb_orders)
    df['user_total_items'] = df.user_id.map(users_.total_items)
    df['total_distinct_items'] = df.user_id.map(users_.total_distinct_items)
    df['user_average_basket'] = df.user_id.map(users_.average_basket)

    if verbose > 0:
        print('product related features')
    df['aisle_id'] = df.product_id.map(product_data.feature2)
    df['department_id'] = df.product_id.map(product_data.feature4)
    df['product_orders'] = df.product_id.map(product_data.orders).astype(np.int32)

    if verbose > 0:
        print('user_X_product related features')
    df['z'] = list(zip(df.product_id, df.user_id))
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct_.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct_.sum_pos_in_cart) / df.UP_orders).astype(np.float32)

    if verbose > 0:
        print(df.dtypes)
        print(df.memory_usage())

    f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_basket', 'aisle_id', 'department_id', 'product_orders',
                'UP_orders', 'UP_orders_ratio', 'UP_average_pos_in_cart']

    return df[f_to_use]