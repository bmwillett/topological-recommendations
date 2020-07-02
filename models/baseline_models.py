"""
A set of baseline models used as points of comparison to main models.  includes

- GetAll - predicts all products will be reordered
- Random - predicts reorders randomly (with probability set by threshold)
-

"""
import lib.tools as tools
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from models.base_model import RecModel
import pandas as pd
import numpy as np
import pickle
import logging

log = logging.getLogger("TR_logger")


class GetAllModel(RecModel):
    """
    Baseline model that predicts all products will be reordered
    """
    def __init__(self):
        log.debug("creating getall model...")
        super().__init__()

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        self.preds = np.ones(dataset.size)
        return self.preds


class RandomModel(RecModel):
    """
    Baseline model that predicts product reorders randomly
    """
    def __init__(self):
        log.debug("creating random model...")
        super().__init__()

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        self.preds = np.random.rand(dataset.size)
        return self.preds


class LogisticModel(RecModel):
    def __init__(self):
        log.debug("creating logistic model...")
        super().__init__()
        self.model = LogisticRegression(random_state=0)

    def fit(self, dataset):
        log.debug("getting features...")
        X_train = tools.get_features(dataset)

        log.debug("fitting classifier...")
        self.model.fit(X_train, dataset.labels)

    def predict(self, dataset):
        log.debug("getting features...")
        X_test = tools.get_features(dataset)

        log.debug("predicting with model...")
        self.preds = self.model.predict(X_test)

        return self.preds


class RandomForestModel(RecModel):
    def __init__(self):
        log.debug("creating random forest model...")
        super().__init__()
        self.model = RandomForestRegressor(random_state=0)

    def fit(self, dataset):
        log.debug("getting features...")
        X_train = tools.get_features(dataset)

        log.debug("fitting classifier...")
        self.model.fit(X_train, dataset.labels)

    def predict(self, dataset):
        log.debug("getting features...")
        X_test = tools.get_features(dataset)

        log.debug("predicting with model...")
        self.preds = self.model.predict(X_test)

        return self.preds


class LGBoostModel(RecModel):
    def __init__(self):
        log.debug("creating lgboost model...")
        super().__init__()

    def fit(self, dataset, ROUNDS=100):
        log.debug("getting features...")
        df_train, categorical = tools.get_features(dataset, return_df=True, drop_categorical=False, return_categorical_list=True)

        self.d_train = lgb.Dataset(df_train, label=dataset.labels, categorical_feature=categorical)

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

        log.debug("fitting classifier...")
        self.model = lgb.train(params, self.d_train, ROUNDS)

    def predict(self, dataset):
        log.debug("getting features...")
        df_test, categorical = tools.get_features(dataset, return_df=True, drop_categorical=False, return_categorical_list=True)

        self.d_test = lgb.Dataset(df_test,
                                   label=dataset.labels,
                                   categorical_feature=categorical)

        log.debug("predicting with model...")
        self.preds = self.model.predict(df_test)

        return self.preds


# TESTING

def run_tests(IC_DATA_DIR):
    from lib.process_data import instacart_process
    from lib.data_class import DataSet

    # set random seed for consistent tests
    np.random.seed(42)

    # load data from instacart csv files (values below use testing directory)
    order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)

    # create dataset
    ic_dataset = DataSet(order_df=order_data, product_df=product_data)

    # check dataframes created correctly
    assert ic_dataset.order_df.shape == (31032, 4)
    assert ic_dataset.product_df.shape == (6126, 5)

    # perform train-test split
    train_dataset, test_dataset = ic_dataset.train_test_split()
    assert train_dataset.order_df.shape == (24939, 4)
    assert test_dataset.order_df.shape == (6093, 4)

    # getall model
    model = GetAllModel()
    model.fit(train_dataset)
    model.predict(test_dataset)
    assert model.preds.shape == test_dataset.labels.shape

    # random model
    model = RandomModel()
    model.fit(train_dataset)
    model.predict(test_dataset)
    assert model.preds.shape == test_dataset.labels.shape

    # logistic model
    model = LogisticModel()
    model.fit(train_dataset)
    model.predict(test_dataset)
    assert model.preds.shape == test_dataset.labels.shape

    # random forest model
    model = RandomForestModel()
    model.fit(train_dataset)
    model.predict(test_dataset)
    assert model.preds.shape == test_dataset.labels.shape

    # LGBoost model
    model = LGBoostModel()
    model.fit(train_dataset)
    model.predict(test_dataset)
    assert model.preds.shape == test_dataset.labels.shape

    log.info("baseline_models tests passed!")


if __name__ == '__main__':
    IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    run_tests(IC_DATA_DIR)