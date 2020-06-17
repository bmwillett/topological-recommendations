from tools import *
import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from gtda.mapper import *
import pickle
from joblib import Parallel, delayed
import logging
log = logging.getLogger("TP_logger")


DATA_DIR_TRAIN = '../data/data_train_fashion_unnormalized/'
DATA_DIR_TEST = '../data/data_test_fashion_unnormalized/'
SMALL_DATA_DIR_TRAIN = '../data/small_data_train_fashion_unnormalized/'
SMALL_DATA_DIR_TEST = '../data/small_data_test_fashion_unnormalized/'
FILE_TRAIN = FILE_TEST = 'trueexamples_in.csv'

TEMP_DATA = './temp_data/'

class MyPCA(sklearn.decomposition.PCA):
    def fit_transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return super().transform(X)


class LatentRep():
    #assumes that all proj in self.projs have .transform() method
    def __init__(self, projectors, label):
        self.projectors = projectors
        self.label = label

    def transform(self, X):
        rep = []
        for proj in self.projectors:
            rep.append( proj.transform(X) )
            #merge all projectors into one vector
            return np.hstack(rep)

def make_small_dataset(frac=0.01,
                       data_dir_train=DATA_DIR_TRAIN, data_dir_test=DATA_DIR_TEST,
                       small_data_dir_train=SMALL_DATA_DIR_TRAIN, small_data_dir_test=SMALL_DATA_DIR_TEST,
                       file_train = FILE_TRAIN, file_test= FILE_TEST, verbose=0):

    train_filepath = data_dir_train + file_train
    test_filepath = data_dir_test + file_test

    df_train = pd.read_csv(train_filepath, header=None)
    df_test = pd.read_csv(test_filepath, header=None)

    df_train = df_train[:int(frac*df_train.shape[0])]
    df_test = df_test[:int(frac*df_test.shape[0])]

    small_train_filepath = small_data_dir_train + file_train
    small_test_filepath = small_data_dir_test + file_test

    df_train.to_csv(small_train_filepath, header=None)
    df_test.to_csv(small_test_filepath, header=None)

def loadMapperData(mode = 'train', data_dir_train = DATA_DIR_TRAIN, data_dir_test = DATA_DIR_TEST,
    file_train = FILE_TRAIN, file_test = FILE_TEST, verbose=0):

    filepath_train = data_dir_train + file_train
    filepath_test = data_dir_test + file_test

    df_train = pd.read_csv(filepath_train, header=None)
    data = df_train.values
    data_train = data[:, 4:]
    data_header_train = data[:, :3]


    df_test = pd.read_csv(filepath_test, header=None)
    data = df_test.values
    data_test = data[:, 4:]
    data_header_test = data[:, :3]

    if verbose>0:
        print(df_train.shape)
        print(df_test.shape)

    return data_train, data_header_train, data_test, data_header_test

def getLatentRep(n_components, data, remake=True):
    label = "PCA%d" % (n_components)
    if not remake and os.path.exists("%s_latent" % (label)):
        frin = open(TEMP_DATA+"%s_latent" % (label), "rb")
        rep = pickle.load( frin )
        return rep

    pca = MyPCA(n_components=n_components)
    pca.fit(data)

    rep = LatentRep([pca], label)
    fr = open(TEMP_DATA+"%s_latent" % (label), "wb")
    pickle.dump(rep, fr)
    fr.close()

    return rep

def runMapper(n_components, data, rep, remake=True, verbose=0):
    label = "PCA%d" % (n_components)
    if not remake and os.path.exists("%s_firstsimplegap_graphs" % (label)):
        fgin = open(TEMP_DATA+"%s_firstsimplegap_graphs" % (label), "rb")
        graphs = pickle.load(fgin)

        fpin = open(TEMP_DATA+"%s_mapper_pipes" % (label), "rb")
        mapper_pipes = pickle.load(fpin)

        return graphs, mapper_pipes

    clusterer = FirstSimpleGap()
    mapper_pipes = []
    pca=rep.projectors[0]

    if log.level == logging.DEBUG:
        print("------> creating projection components...")
    for k in range(n_components):
        if log.level == logging.DEBUG:
            print("---------> on component {}/{}...".format(k+1,n_components))
        proj = Projection(columns = k)
        filter_func = Pipeline(steps=[('pca', pca), ('proj', proj)])
        cover = OneDimensionalCover(n_intervals=10, overlap_frac=0.33)
        mapper_pipe = make_mapper_pipeline(scaler=None,
                                       filter_func = filter_func,
                                       cover=cover,
                                       clusterer=clusterer,
                                       verbose=verbose>0,
                                       n_jobs = 1)
        mapper_pipe.set_params(filter_func__proj__columns = k)
        mapper_pipes.append( ("PCA%d" % (k+1), mapper_pipe ) )

    # try parallelization
    if log.level == logging.DEBUG:
        print("------> entering parallelization...")
    with elapsed_timer() as elapsed:
        graphs = Parallel(n_jobs=5, prefer="threads")(
            delayed(mapper_pipe[1].fit_transform)(data) for mapper_pipe in mapper_pipes
        )
    if log.level == logging.DEBUG:
        print("------> exiting parallelization after {} seconds".format(elapsed()))

    fg = open(TEMP_DATA+"%s_firstsimplegap_graphs" % (label), "wb")
    pickle.dump(graphs, fg)
    fg.close()

    fp = open(TEMP_DATA+"%s_mapper_pipes" % (label) , "wb")
    pickle.dump(mapper_pipes, fp)
    fp.close()

    return graphs, mapper_pipes

def makeGraphBins(n_components, data, data_header, graphs, remake=True):
    # an algorithm for binarizing training points using
    # the computed graph representation
    # computes and outputs a CSV file with all training
    # points representations , used for training a
    # NeuralNet classifier in the next step.

    label = "PCA%d" % (n_components)
    if not remake and os.path.exists(TEMP_DATA+'matrix_train_%s.csv' % (label)):
        total_graphbinm = np.loadtxt(TEMP_DATA+'matrix_train_%s.csv' % (label), delimiter=',',)
        return total_graphbinm

    total_graphbin = []
    if data_header is not None:
        total_graphbin.append(data_header)

    data_len = data.shape[0]
    featlen = 0
    for comp in range(n_components):
        graph = graphs[comp]  # graph created for given projection component
        nrnodes = len(graph['node_metadata']['node_id'])  # number of nodes in graph
        graphbin = np.zeros((data_len, nrnodes), dtype=int)  # matrix indicating which nodes train data lies in
        for node in range(nrnodes):
            for pt in graph['node_metadata']['node_elements'][node]:
                graphbin[pt][node] = 1

        stamp = str(comp + 1)
        total_graphbin.append(graphbin)
        featlen += graphbin.shape[1]

    total_graphbinm = np.hstack(total_graphbin)

    np.savetxt(TEMP_DATA+'matrix_train_%s.csv' % (label), total_graphbinm, delimiter=',', fmt='%f')

    return total_graphbinm, featlen, graphbin

def find_alphas(val, intervals, delta = 0.1):
    alphas = []
    if val < intervals[0][1]:
        return tuple([0])
    if val > intervals[-1][0]:
        return tuple([len(intervals)-1])
    for n, intv in enumerate(intervals):
        midv = (intv[0] + intv[1]) / 2
        if abs(val - midv) < abs(intv[1] - intv[0])*(1 + delta) / 2:
            alphas.append(n)
    return tuple(alphas)


def projectTestData(n_components, rep, data, datatest, datatest_header, graphs, mapper_pipes, total_graphbinm, featlen, NRNN, remake=True):
    label = "PCA%d" % (n_components)
    if not remake and os.path.exists(TEMP_DATA+'matrix_test_%s.csv' % (label)):
        total_test_rep = np.loadtxt(TEMP_DATA+'matrix_test_%s.csv' % (label), delimiter=',',)
        return total_test_rep

    # # Mapper testing
    testexamples = datatest.shape[0]

    # project the test data
    latent_datatest = rep.transform(datatest)

    # for each test data, translate its latent representation
    # into interval nrs
    intervals = []
    for mapper_pipe in mapper_pipes:
        intervals.append(mapper_pipe[1].get_mapper_params()['cover'].get_fitted_intervals())

    fullalphas = []
    for latent_testp in latent_datatest:
        alphas = []
        for n, latentv in enumerate(latent_testp):
            alphas.append(find_alphas(latentv, intervals[n]))
        fullalphas.append(alphas)

    # seach for NNeighbor in the preimage of intervals
    # precompute NNeighbor for preimage of each interval
    int_preim = []
    int_nn = {}

    # precompute dictionary of fitted NNeighbor for further use
    # currently use only single component
    # for n in range(len(intervals)):
    n = 0
    int_preim_int = []
    for i in range(len(intervals[n])):
        nodeids = graphs[n]['node_metadata']['node_elements'][i]
        knn = NearestNeighbors(n_neighbors=NRNN, metric='euclidean')
        knn.fit(data[nodeids])
        int_preim_int.append(nodeids)
        # knn is fitted knn using data[nodeids] subset of data nodeids are also saved to further reference them in the
        # whole data
        int_nn.update({(n, tuple([i])): [knn, data[nodeids], np.array(nodeids)]})
        if (i > 0):
            knn = NearestNeighbors(n_neighbors=NRNN, metric='euclidean')
            union = list(set(nodeids) | set(int_preim_int[i - 1]))
            knn.fit(data[union])
            int_nn.update({(n, tuple([i - 1, i])): [knn, data[union], np.array(union)]})

    l = 0
    # for each testpoint compute its representation by finding its NNeighbor
    nnids = []
    nndists = []
    for i in range(len(fullalphas)):
        nncompids = []
        nncompdists = []
        # for j in range(n_components):
        # compute this for only the first component (computing for all components is prohibitive)
        j = 0  # take the first component
        # compute knn with distances for each test point (search in the preimage of j-th component)
        inn = int_nn.get((j, fullalphas[i][j]))
        if (len(inn[2]) > NRNN):
            knn = inn[0].kneighbors([datatest[i]])
            ids = inn[2][knn[1]]
            nncompids.append(ids.squeeze())
            nncompdists.append(knn[0].squeeze())
        else:
            ids = inn[2]  # all available points
            dists = np.linalg.norm(datatest[i] - inn[1], axis=1)
            nncompids.append(ids)
            nncompdists.append(dists)

        nnids.append(
            nncompids)  # for each test input , nnids stores a list (per filter function) of 5 NN global ids found in the preimage
        nndists.append(
            nncompdists)  # for each test input , nndists stores a list (per filter function) of 5 NN distances (corresp to the ones in nnids)


    # generate data matrix with test points representations
    # and save to  CSV file,
    # produced by using NNeighbor algorithm

    mu = 1e-05
    # change names of variables here
    # neighbor_weights = np.ones_like( nndists )
    neighbor_weights = []

    test_rep = np.zeros((testexamples, featlen))

    # ASSUMING BELOW, THAT SINGLE COMPONENT HAS BEEN COMPUTED
    # compute the weights for weighting the NN representations


    for i in range(len(nndists)):
        ns = nndists[i][0]
        ns = 1. / (ns + mu)
        ns = ns / sum(ns)
        neighbor_weights.append(ns)

    # compute the representations using weighted NN
    for i in range(len(nndists)):
        ns = nnids[i]
        features = neighbor_weights[i].reshape(1, len(ns[0])).dot(total_graphbinm[ns].astype(float))
        test_rep[i] = features

    if datatest_header is not None:
        total_test_rep = np.hstack([datatest_header, test_rep])
    else:
        total_test_rep = np.hstack([test_rep])

    np.savetxt(TEMP_DATA+'matrix_test_%s.csv' % (label), total_test_rep, delimiter=',', fmt='%f')

    return total_test_rep
