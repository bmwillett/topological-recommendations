import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from gensim.models import Word2Vec
import tensorflow as tf
import gensim
from contextlib import contextmanager
from timeit import default_timer

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

def get_word2vec_model(order_data, size=100, retrain=True, model_dir= './models/', verbose=0):
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
