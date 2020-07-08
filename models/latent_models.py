"""
Definition of latent models

These are simple models used by RecModels to encode various features of datasets into
latent spaces, where the ideas is that data points mapping to nearby points inn the
latent spaces should be similar in some sense.  This notion of a space encoding similarity
is important for the mapper classifier algorithm to perform correctly. These are used
together with latent models to build up recommendation models
"""

import os
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import pathlib
import logging

import lib.tools as tools
from lib.mapper_class import MapperClassifier

log = logging.getLogger("TR_logger")
model_dir = pathlib.Path(__file__).parent.absolute()

# base class for latent models, contains common functions
class LatentModel:
    """
    Base latent model class with common methods
    should not be used directly in models

    Methods:

        fit : given training DataSet, fit latent model

        transform : given DataSet, encode points innto latent space (must call after training),
                 returns encoded output

        fit_transform : perform fit followed by transform on a DataSet
                    returns encoded output
    """
    def fit(self, dataset):
        pass

    def transform(self, dataset):
        pass

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

class EmptyLatentModel(LatentModel):
    """
    Empty latent model class
    simply returns zero-length latent vector
    Used as placeholder when a latent model is not present
    """
    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return np.empty(shape=(dataset.size, 0))

# User model
class UserModel(LatentModel):
    """
    Basic user-encoding.  Uses autoencoder to encode users based on
    past purhcase history.
    """
    def __init__(self, encoding_dim=32):
        log.debug("creating UserModel...")
        self.input_dim = None
        self.encoding_dim = encoding_dim

        self.autoencoder, self.encoder, self.decoder = None, None, None

    def fit(self, dataset, epochs=50, batch_size=32):
        """
        :param dataset: input dataset used for training
        :return: None
        """
        self.input_dim = len(dataset.prod_ids)

        input = Input(shape=(self.input_dim,))

        encoded = Dense(self.encoding_dim, activation='relu')(input)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input, decoded)
        self.encoder = Model(input, encoded)

        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        # TODO: way to use product latent space in encoding?
        self.autoencoder.compile(optimizer='adadelta', loss=tools.nonzero_loss)

        X_train = dataset.user_prod_matrix

        log.debug("fitting autoencoder in User Model...")
        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             verbose=(log.getEffectiveLevel() == logging.DEBUG))

    def transform(self, dataset):
        """
        :param dataset: input dataset used for testing
        :return: numpy array consisting of transformed points
        """
        X = dataset.user_prod_matrix
        ids = tuple([dataset.user_idx[uid] for uid in dataset.prior_user_prod.user_id])
        X_to_enc = X[ids, :]
        encoded_users = self.encoder.predict(X_to_enc)

        return encoded_users

# Topological User model
class TopUserModel(LatentModel):
    """
    Topological user latent model
    Takes previous autoencoder and includes further mapper encoding for topological protection
    """
    def __init__(self, n_components=5, NRNN=3):
        log.debug("creating TopUserModel...")
        self.user_model = UserModel()
        self.n_components, self.NRNN = n_components, NRNN
        self.mapper = MapperClassifier()
        self.X_to_map, self.X_map = None, None

    def fit(self, dataset, epochs=50, batch_size=32, retrain_mapper=True):
        log.debug("fitting TopUserModel...")
        self.user_model.fit(dataset, epochs=epochs, batch_size=batch_size)
        self.X_to_map = self.user_model.transform(dataset)

        #TODO: add logic so only trains on each user once (code written somewhere...)

        log.debug("encoding features with mapper-classifier...")
        if retrain_mapper or self.X_map is None:
            self.mapper.fit(self.X_to_map)
        log.debug(f"created mapper encoding of size {self.mapper.mapper_features}")

    def transform(self, dataset):
        log.debug("transforming TopUserModel (test data)...")
        X_to_map = self.user_model.transform(dataset)
        return self.mapper.transform(X_to_map)

    def fit_transform(self, dataset, epochs=50, batch_size=32, retrain_mapper=True):
        log.debug("fit_transform in TopUserModel...")
        self.fit(dataset, epochs=epochs, batch_size=batch_size, retrain_mapper=retrain_mapper)
        return self.mapper.transform()

# Product model
class ProductModel(LatentModel):
    """
    Basic product-encoding.  Combines two product latent models, word2vec and TFIDF,
    defined below, and projects result to n_component dimensions using PCA.
    """
    def __init__(self, n_components=10):
        self.w2v = word2vecModel()
        self.tfidf = TFIDFModel()
        self.n_components = n_components

    def fit(self, dataset):
        """
        :param dataset: input dataset used for training
        :return: None
        """
        self.w2v.fit(dataset)
        self.tfidf.fit(dataset)

        X1 = self.w2v.transform(dataset)
        X2 = self.tfidf.transform(dataset)
        X = np.hstack([X1, X2])

        self.pcamodel = PCA(n_components=self.n_components)
        self.pcamodel.fit(X)

    def transform(self, dataset):
        """
        :param dataset: input dataset used for testing
        :return: numpy array consisting of transformed points
        """
        X1 = self.w2v.transform(dataset)
        X2 = self.tfidf.transform(dataset)
        X = np.hstack([X1, X2])
        return self.pcamodel.transform(X)


class word2vecModel(LatentModel):
    def __init__(self, encoding_dim=100, retrain=False, saved_model_dir=model_dir.joinpath('saved_models')):
        self.encoding_dim, self.retrain, self.saved_model_dir = encoding_dim, retrain, saved_model_dir
        self.model = None

    def fit(self, dataset):
        """
        :param dataset: input dataset used for training
        :return: None
        """
        retrain = self.retrain or not os.path.isfile(self.saved_model_dir.joinpath('product2vec.model'))

        self.model = tools.get_word2vec_model(dataset.order_df, size=self.encoding_dim, retrain=retrain,
                                              model_dir=self.saved_model_dir)

    def transform(self, dataset):
        """
        :param dataset: input dataset used for testing
        :return: numpy array consisting of transformed points
        """
        X_enc = []
        for pid in dataset.prior_user_prod.product_id:
            if str(pid) in self.model.wv:
                X_enc.append(self.model.wv[str(pid)])
            else:
                X_enc.append(np.zeros(self.encoding_dim))

        return np.array(X_enc)


class TFIDFModel(LatentModel):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, dataset):
        """
        :param dataset: input dataset used for training
        :return: None
        """
        log.debug("extracting text data from dataset...")
        comb_frame = dataset.product_df.feature1.str.cat(" " + dataset.product_df.feature3.str.cat(" " + dataset.product_df.feature5))
        comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
        comb_frame.head()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer.fit(comb_frame)

        log.debug("fitting TF-IDF vectorizier...")
        X = self.vectorizer.transform(comb_frame)

        log.debug("fitting SVD...")
        self.tsvd = TruncatedSVD(n_components=self.n_components)
        self.tsvd.fit(X)

    def transform(self, dataset):
        """
        :param dataset: input dataset used for testing
        :return: numpy array consisting of transformed points
        """
        df = dataset.product_df.loc[dataset.prior_user_prod.product_id]

        comb_frame = df.feature1.str.cat(" " + df.feature3.str.cat(" " + df.feature5))
        comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
        comb_frame.head()
        X_test = self.vectorizer.transform(comb_frame)

        return self.tsvd.transform(X_test)



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

    # create user latent model and fit to train_dataset
    user_latent = UserModel()
    user_latent.fit(train_dataset, epochs=2)
    assert user_latent.transform(train_dataset).shape == (9447, 32)

    # test encoding and decoding works as expected
    encoded_upm = user_latent.encoder.predict(train_dataset.user_prod_matrix)
    decoded_upm = user_latent.decoder.predict(encoded_upm)
    autoencoded_upm = user_latent.autoencoder.predict(train_dataset.user_prod_matrix)
    assert encoded_upm.shape == (154, 32)
    assert decoded_upm.shape == (154, 6126)
    assert (decoded_upm == autoencoded_upm).all()

    # create topological user latent model and fit to train_dataset
    top_user_latent = TopUserModel()
    top_user_latent.fit_transform(train_dataset, epochs=2)
    assert top_user_latent.transform(test_dataset).shape == (2767, 219)

    # # manually compare original UPM and autoencoder prediction
    # print(train_dataset.user_prod_matrix[:5, :6])
    # print(autoencoded_upm[:5, :6])

    # transform test_dataset
    assert user_latent.transform(test_dataset).shape == (2767, 32)

    # create word2vec latent model and fit to train_dataset and transform test_dataset
    # TODO: not retraining correctly...
    w2v_model = word2vecModel()
    w2v_model.fit(train_dataset)
    log.debug(w2v_model.model)
    assert w2v_model.transform(test_dataset).shape == (2767, 100)

    # create TFIDF latent model and fit to train_dataset and transform test_dataset
    tfidf_model = TFIDFModel()
    tfidf_model.fit(train_dataset)
    assert tfidf_model.transform(test_dataset).shape == (2767, 20)

    # create product latent model (combination of previous two) and fit to train_dataset and transform test_dataset
    product_latent = ProductModel()
    product_latent.fit(train_dataset)
    assert product_latent.transform(test_dataset).shape == (2767, 10)

    log.info("latent_models tests passed!")

if __name__ == '__main__':
    IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    model = run_tests(IC_DATA_DIR)
