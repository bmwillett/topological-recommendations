import sys
sys.path.append('./lib')

from mapper_class import MapperClassifier
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

class BaseModel:
    def __init__(self):
        self.test_dataset = None
        self.X_pred = None

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pass

    def test_user(self, user_id, X_user, test_dataset):
        user_products = test_dataset.order_data[test_dataset.order_data.user_id == user_id]['product_id'].values
        order_size = len(user_products)

        # pick from pool of products equal to products actually bought plus N_extra random TODO: remove this cheat
        N_extra = order_size
        test_products = np.concatenate((np.array(user_products), np.random.choice(test_dataset.product_ids, N_extra, replace=False)))

        product_ranking = np.argsort(X_user)

        top = [test_dataset.product_ids[b] for b in product_ranking[::-1] if test_dataset.product_ids[b] in test_products]

        true_rank = [i for i, pid in enumerate(top) if pid in user_products]
        predicted_products = top[:order_size]

        tp = len(set(predicted_products).intersection(set(user_products)))
        fp = order_size - tp
        # tn = len(subset) - 2 * order_size + tp
        fn = order_size - tp

        prec, rec = tp / (tp + fp), tp / (tp + fn)

        NDCG = np.log(2) * sum([1 / np.log(i + 2) for i in true_rank]) / order_size

        return prec, rec, NDCG

    def accuracy_test(self, test_dataset=None, N_tests=100):
        if test_dataset is not None:
            self.predict(test_dataset)
        elif self.test_dataset is None:
            print("Error: no test data")
            return

        user_ids = np.random.choice(self.test_dataset.user_ids, size=N_tests, replace=False)

        precs, recs, NDCGs = [], [], []

        for uid in user_ids:
            X_user = self.X_pred[self.test_dataset.user_idx[uid], :]
            prec, rec, NDCG = self.test_user(uid, X_user, self.test_dataset)

            precs.append(prec)
            recs.append(rec)
            NDCGs.append(NDCG)

        return np.mean(precs), np.mean(recs), np.mean(NDCGs)

    def robustness_test(self, test_dataset=None, N_tests=100):
        if test_dataset is not None:
            self.predict(test_dataset)
        elif self.test_dataset is None:
            print("Error: no test data")
            return

        X = self.test_dataset.get_user_product_matrix()
        (u, p) = X.shape
        X_adv = np.copy(X)

        #  change one random item for each user
        # TODO: change to modify dataset, call predict
        for i in range(u):
            j_old = np.random.choice(np.where(X[i, :] > 0)[0])
            j_new = np.random.randint(p)
            X_adv[i, j_new], X_adv[i, j_old] = X[i, j_old], X[i, j_new]
        X_pred_adv = self.predict(None, X_test=X_adv)

        user_ids = np.random.choice(self.test_dataset.user_ids, size=N_tests, replace=False)

        precs, recs, NDCGs = [], [], []

        for uid in user_ids:
            X_user = X_pred_adv[self.test_dataset.user_idx[uid], :]
            prec, rec, NDCG = self.test_user(uid, X_user, self.test_dataset)

            precs.append(prec)
            recs.append(rec)
            NDCGs.append(NDCG)

        return np.mean(precs), np.mean(recs), np.mean(NDCGs)

"""
Autoencoder Model

"""
class AEM(BaseModel):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()

        self._input_dim = input_dim
        self._encoding_dim = encoding_dim

        input = Input(shape=(input_dim,))

        encoded = Dense(encoding_dim, activation='relu')(input)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input, decoded)
        self.encoder = Model(input, encoded)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.autoencoder.compile(optimizer='adadelta', loss=nonzero_loss)

    def fit(self, train_dataset, epochs=50, batch_size=25):

        X_train = train_dataset.get_user_product_matrix()

        self.autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True)

    def predict(self, test_dataset, X_test=None):
        if test_dataset is not None:
            self.test_dataset = test_dataset
            X_test = test_dataset.get_user_product_matrix()

        self.X_pred = self.autoencoder.predict(X_test)
        return self.X_pred

class TAEM(BaseModel):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()

        self._input_dim = input_dim
        self._encoding_dim = encoding_dim

        self.aem = AEM(input_dim, encoding_dim=encoding_dim)


    def fit(self, train_dataset, epochs=50, batch_size=25, n_components = 15, NRNN = 3):

        # first train AE model on dataset to get user encoder
        self.aem.fit(train_dataset, epochs=epochs, batch_size=batch_size)

        X_train = train_dataset.get_user_product_matrix()

        # next train rest of model using encoded output of training data
        encoded_users = self.aem.encoder.predict(X_train)

        self.encoder_mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

        # run through mapper classifier to get graph-bin output
        X_map = self.encoder_mapper.fit(encoded_users, None)

        # finally, feed into neural network with two hidden layers and train to match X_train
        input_dim = X_map.shape[1]
        h1_dim = h2_dim = 100
        output_dim = X_train.shape[1]

        input = Input(shape=(input_dim,))
        h1 = Dense(h1_dim, activation='relu')(input)
        h2 = Dense(h2_dim, activation='relu')(h1)
        output = Dense(output_dim, activation='sigmoid')(h2)

        self.mapper_model = Model(input, output)

        self.mapper_model.compile(optimizer='adadelta', loss=nonzero_loss)

        self.mapper_model.fit(X_map, X_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=True)

    def predict(self, test_dataset, X_test=None):
        if test_dataset is not None:
            self.test_dataset = test_dataset
            X_test = test_dataset.get_user_product_matrix()

        # run through encoder
        encoded_users = self.aem.encoder.predict(X_test)

        # then project to graph bins using mapper
        X_test_map = self.encoder_mapper.project(encoded_users, None)

        # finally run through mapper model (NN after mapper)
        self.X_pred = self.mapper_model.predict(X_test_map)

        return self.X_pred

def nonzero_loss(y_true, y_pred):
    y_pred_nonzero = tf.where(y_true>0,y_pred,0)
    num_labels = tf.cast(tf.math.count_nonzero(y_true),'float32')
    mae = tf.math.divide(tf.reduce_sum(tf.abs(tf.subtract(y_pred_nonzero, y_true))),num_labels)
    return mae