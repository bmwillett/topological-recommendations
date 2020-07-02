"""
Definition of main recommendation models

Summary:

    NonTopModel : TBA

    TopModel : TBA

"""
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import logging

from models.base_model import RecModel
from models.feature_models import EmptyFeatureModel
from models.latent_models import EmptyLatentModel
from lib.mapper_class import MapperClassifier

log = logging.getLogger("TR_logger")

class NonTopModel(RecModel):
    """
    Main non-topological recommendation model

    details TBA
    """
    def __init__(self, user_latent_model=None, product_latent_model=None, feature_model=None,
                 h_dims = (100,50), output_dim=1):
        super().__init__()
        log.debug("creating main non-topological model...")
        if user_latent_model is None:
            user_latent_model = EmptyLatentModel()
        if product_latent_model is None:
            product_latent_model = EmptyLatentModel()
        if feature_model is None:
            feature_model = EmptyFeatureModel()
        self.user_latent_model, self.product_latent_model, self.feature_model = user_latent_model, product_latent_model, feature_model

        self.input_dim, self.h_dims, self.output_dim = None, h_dims, output_dim
        self.final_model = None

    def fit(self, dataset, fit_latent=True, epochs=50):
        """
        Fits model to training dataset

        :param dataset: dataset used for training
        :param fit_latent: if true, fit the latent models using the dataset, otherwise assume they have already been fit
        :param epochs: number of epochs to use in training of final model
        :return: None
        """
        if fit_latent:
            log.debug("fitting latent models...")
            self.user_latent_model.fit(dataset)
            self.product_latent_model.fit(dataset)

        # process dataset through latent and feature models
        log.debug("encoding dataset using latent and feature models...")
        user_latent = self.user_latent_model.transform(dataset)
        product_latent = self.product_latent_model.transform(dataset)
        features = self.feature_model.transform(dataset)

        X_train = np.hstack([user_latent, product_latent, features])
        log.debug(f"created X_train of shape {X_train.shape}")
        self.input_dim = X_train.shape[1]

        # build and train final_model neural network
        log.debug("building final model...")
        input_layer = Input(shape=(self.input_dim,))
        layers = [input_layer]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output_layer = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.final_model = Model(input_layer, output_layer)
        self.final_model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        log.debug("fitting final model...")
        self.final_model.fit(X_train, dataset.labels, epochs=epochs, verbose=(log.getEffectiveLevel() == logging.DEBUG))


    def predict(self, dataset):
        """
        predict labels for given dataset

        :param dataset: test dataset
        :return: numpy array of predictions (also saved in self.preds)
        """
        # process dataset through latent and feature models
        log.debug("encoding dataset using latent and feature models...")
        user_latent = self.user_latent_model.transform(dataset)
        product_latent = self.product_latent_model.transform(dataset)
        features = self.feature_model.transform(dataset)

        X_test = np.hstack([user_latent, product_latent, features])
        log.debug(f"created X_test of shape {X_test.shape}")

        log.debug("predicting using final model...")
        self.preds = self.final_model.predict(X_test).squeeze()

        return self.preds


class TopModel(RecModel):
    """
    Main topological recommendation model

    details TBA
    """
    def __init__(self, user_latent_model=None, product_latent_model=None, feature_model=None,
                 h_dims = (100, 50), output_dim=1, n_components=5, NRNN=3, bypass=True):
        super().__init__()
        log.debug("creating main topological model...")
        if user_latent_model is None:
            user_latent_model = EmptyLatentModel()
        if product_latent_model is None:
            product_latent_model = EmptyLatentModel()
        if feature_model is None:
            feature_model = EmptyFeatureModel()
        self.user_latent_model, self.product_latent_model, self.feature_model = user_latent_model, product_latent_model, feature_model

        self.input_dim, self.h_dims, self.output_dim = None, h_dims, output_dim
        self.n_components, self.NRNN, self.bypass = n_components, NRNN, bypass
        self.mapper, self.final_model = MapperClassifier(), None
        self.X_map = None

    def fit(self, dataset, fit_latent=True, epochs=50, retrain_mapper=True):
        """
        Fits model to training dataset

        :param dataset: dataset used for training
        :param fit_latent: if true, fit the latent models using the dataset, otherwise assume they have already been fit
        :param epochs: number of epochs to use in training of final model
        :return: None
        """
        if fit_latent:
            log.debug("fitting latent models...")
            self.user_latent_model.fit(dataset)
            self.product_latent_model.fit(dataset)

        # process dataset through latent and feature models
        log.debug("encoding dataset using latent and feature models...")
        user_latent = self.user_latent_model.transform(dataset)
        product_latent = self.product_latent_model.transform(dataset)
        features = self.feature_model.transform(dataset)
        X_train = np.hstack([user_latent, product_latent, features])
        log.debug(f"created X_train of shape {X_train.shape}")

        # send X_train through mapper-classifier
        log.debug("encoding features with mapper-classifier...")
        if retrain_mapper or self.X_map is None:
            self.X_map = self.mapper.fit_transform(X_train)
        log.debug(f"created mapper encoding of size {self.X_map.shape[1]}")

        # if bypass, include original features alongside encoded features
        if self.bypass:
            self.X_map = np.hstack([X_train, self.X_map])
        log.debug(f"created mapper encoding of size {self.X_map.shape[1]}")

        self.input_dim = self.X_map.shape[1]

        # build and train final_model neural network
        log.debug("building final model...")
        input_layer = Input(shape=(self.input_dim,))
        layers = [input_layer]
        for h_dim in self.h_dims:
            layers.append(Dense(h_dim, activation='relu')(layers[-1]))
        output_layer = Dense(self.output_dim, activation='sigmoid')(layers[-1])

        self.final_model = Model(input_layer, output_layer)
        self.final_model.compile(optimizer='adadelta', loss=tf.keras.losses.BinaryCrossentropy())

        log.debug("fitting final model...")
        self.final_model.fit(self.X_map, dataset.labels, epochs=epochs, verbose=(log.getEffectiveLevel() == logging.DEBUG))


    def predict(self, dataset):
        """
        predict labels for given dataset

        :param dataset: test dataset
        :return: numpy array of predictions (also saved in self.preds)
        """
        # process dataset through latent and feature models
        log.debug("encoding dataset using latent and feature models...")
        user_latent = self.user_latent_model.transform(dataset)
        product_latent = self.product_latent_model.transform(dataset)
        features = self.feature_model.transform(dataset)

        X_test = np.hstack([user_latent, product_latent, features])
        log.debug(f"created X_test of shape {X_test.shape}")

        log.debug("projecting test features using mapper classifier...")
        X_test_map = self.mapper.transform(X_test)

        if self.bypass:
            X_test_map = np.hstack([X_test, X_test_map])

        log.debug("predicting using final model...")
        self.preds = self.final_model.predict(X_test_map).squeeze()

        return self.preds



# TESTING

def run_tests(IC_DATA_DIR):
    from lib.process_data import instacart_process
    from lib.data_class import DataSet
    from models.latent_models import UserModel, ProductModel
    from models.feature_models import MainFeatureModel

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

    # create user latent model, fit and transform
    user_latent = UserModel()
    user_latent.fit(train_dataset, epochs=2)
    assert user_latent.transform(train_dataset).shape == (9447, 32)

    # create product latent model, fit and transform
    product_latent = ProductModel()
    product_latent.fit(train_dataset)
    assert product_latent.transform(train_dataset).shape == (9447, 10)

    # create feature model
    feature_model = MainFeatureModel()
    X = feature_model.transform(train_dataset)
    assert X.shape == (9447, 8)

    # fit non-top model to train_dataset
    model = NonTopModel(user_latent_model=user_latent, product_latent_model=product_latent, feature_model=feature_model)
    model.fit(train_dataset, fit_latent=False, epochs=2)
    assert model.input_dim == 50

    # predict on test_dataset
    model.predict(test_dataset)
    assert model.preds.shape == (2767,)
    assert model.preds.shape == test_dataset.labels.shape

    # fit top model to train_dataset
    model = TopModel(user_latent_model=user_latent, product_latent_model=product_latent, feature_model=feature_model)
    model.fit(train_dataset, fit_latent=False, epochs=2)
    print(model.input_dim)

    # predict on test_dataset
    model.predict(test_dataset)
    assert model.preds.shape == (2767,)
    assert model.preds.shape == test_dataset.labels.shape

    log.info("main_models tests passed!")


if __name__ == '__main__':
    IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    run_tests(IC_DATA_DIR)
