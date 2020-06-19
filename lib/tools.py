import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from gensim.models import Word2Vec
import tensorflow as tf
import gensim
from contextlib import contextmanager
from timeit import default_timer
import pandas as pd

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def nonzero_loss(y_true, y_pred):
    y_pred_nonzero = tf.where(y_true>0,y_pred,0)
    num_labels = tf.cast(tf.math.count_nonzero(y_true),'float32')
    mae = tf.math.divide(tf.reduce_sum(tf.abs(tf.subtract(y_pred_nonzero, y_true))),num_labels)
    return mae

def get_word2vec_model(order_data, size=100, retrain=True, model_dir= './models/', verbose=1):
    """
    :param data_dir: directory to find data -> later replaced by standard data format
    :param model_dir: directory to save model
    :return:
    """
    if not retrain:
        return Word2Vec.load(model_dir+'product2vec.model')

    order_data["product_id"] = train_orders["product_id"].astype(str)

    if verbose > 0:
        print("assembling sentences...")
    sentences = order_data.groupby("order_id").apply(
        lambda order: order['product_id'].tolist())

    longest = np.max(sentences.apply(len))
    sentences = sentences.values

    if verbose>0:
        print("training model...")
    model = gensim.models.Word2Vec(sentences, size=size,
                                   window=longest, min_count=2, workers=4)

    model.save(model_dir+'product2vec.model')

    return model


def get_batch(vocab, model, n_batches=1):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(10, 10))  # in inches
    texts = []
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        texts.append(plt.text(x,y,label,size=12))
                     # xytext=(5, 2),
                     # textcoords='offset points',
                     # ha='right',
                     # va='bottom',
                     # size=15))

    #     plt.savefig(filename)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.show()



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
    usr_['nb_orders'] = order_data.groupby('user_id')['order_number'].max()
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