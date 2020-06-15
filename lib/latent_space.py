import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
from tools import *
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

def get_word2vec_model(train_orders, prior_orders, products, retrain=True, model_dir= './models/', verbose=1):
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
    model = gensim.models.Word2Vec(sentences, size=100,
                                   window=longest, min_count=2, workers=4)

    model.save(model_dir+'product2vec.model')

    return model

def get_tfidf_model(product_data, data_dir='../data/', model_dir='../models/'):

    course_df = pd.read_csv(data_dir+"courses.csv")

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

    return vectorizer,X


def latent_model(order_data=None, product_data=None, retrain=True, data_dir= './data/instacart_2017_05_01/', model_dir= './models/', verbose=1):
    # vocab = list(model.wv.vocab.keys())
    #
    # pca = PCA(n_components=2)
    # pca.fit(model.wv.syn0)
    #
    #
    # embeds = []
    # labels = []
    # for item in get_batch(vocab, model, n_batches=3):
    #     embeds.append(model[item])
    #     labels.append(products.loc[int(item)]['product_name'])
    # embeds = np.array(embeds)
    # embeds = pca.fit_transform(embeds)


    if verbose > 0:
        print("loading data...")

    retrain = retrain or not os.path.isfile(model_dir + 'product2vec.model')

    train_orders = pd.read_csv(data_dir+"order_products__train.csv") if retrain else None
    prior_orders = pd.read_csv(data_dir+"order_products__prior.csv") if retrain else None
    products = pd.read_csv(data_dir+"products.csv").set_index('product_id')

    if verbose>1:
        print("getting model...")
    model = get_word2vec_model(train_orders, prior_orders, products, retrain=retrain)

    vocab = list(model.wv.vocab.keys())

    print("fitting pca...")
    pca = PCA(n_components=2)
    pca.fit(model.wv.syn0)

    print("plotting sample...")
    embeds = []
    labels = []
    for item in get_batch(vocab, model, n_batches=3):
        embeds.append(model[item])
        labels.append(products.loc[int(item)]['product_name'])
    embeds = np.array(embeds)
    embeds = pca.fit_transform(embeds)
    plot_with_labels(embeds, labels)

