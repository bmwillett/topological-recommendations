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

def get_word2vec_model(train_orders, prior_orders, products, size=100, retrain=True, model_dir= './models/', verbose=0):
    """
    :param data_dir: directory to find data -> later replaced by standard data format
    :param model_dir: directory to save model
    :return:
    """
    if not retrain:
        return Word2Vec.load(model_dir+'product2vec.model')

    train_orders["product_id"] = train_orders["product_id"].astype(str)
    prior_orders["product_id"] = prior_orders["product_id"].astype(str)

    if verbose > 0:
        print("assembling sentences...")
    train_products = train_orders.groupby("order_id").apply(
        lambda order: order['product_id'].tolist())
    prior_products = prior_orders.groupby("order_id").apply(
        lambda order: order['product_id'].tolist())

    sentences = prior_products.append(train_products)
    longest = np.max(sentences.apply(len))
    sentences = sentences.values

    if verbose>0:
        print("training model...")
    model = gensim.models.Word2Vec(sentences, size=size,
                                   window=longest, min_count=2, workers=4)

    model.save(model_dir+'product2vec.model')

    return model


def get_tfidf_model(product_data, data_dir='../data/', model_dir='../models/'):
    course_df = pd.read_csv(data_dir + "courses.csv")

    # 2. drop rows with NaN values for any column, specifically 'Description'
    # Course with no description won't be of much use
    course_df = course_df.dropna(how='any')
    # 3. Pre-processing step: remove words like we'll, you'll, they'll etc.
    course_df['Description'] = course_df['Description'].replace({"'ll": " "}, regex=True)
    # 4. Another Pre-preprocessing step: Removal of '-' from the CourseId field
    course_df['CourseId'] = course_df['CourseId'].replace({"-": " "}, regex=True)
    # 5. Combine three columns namely: CourseId, CourseTitle, Description
    comb_frame = course_df.CourseId.str.cat(" " + course_df.CourseTitle.str.cat(" " + course_df.Description))
    # 6. Remove all characters except numbers & alphabets
    # Numbers are retained as they are related to specific course series also
    comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(comb_frame)

    return vectorizer, X


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


    #
    #
    # together = [(0, 1.0, 0.4), (25, 1.0127692669427917, 0.41), (50, 1.016404709797609, 0.41),
    #             (75, 1.1043426359673716, 0.42), (100, 1.1610446924342996, 0.44), (125, 1.1685687930691457, 0.43),
    #             (150, 1.3486407784550272, 0.45), (250, 1.4013999168008104, 0.45)]
    # together.sort()
    #
    # text = [x for (x, y, z) in together]
    # eucs = [y for (x, y, z) in together]
    # covers = [z for (x, y, z) in together]
    #
    # p1 = plt.plot(eucs, covers, color="black", alpha=0.5)
    # texts = []
    # for x, y, s in zip(eucs, covers, text):
    #     texts.append(plt.text(x, y, s))
    #
    # plt.xlabel("Proportional Euclidean Distance")
    # plt.ylabel("Percentage Timewindows Attended")
    # plt.title("Test plot")
    # adjust_text(texts, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    # plt.show()