"""
Main class and methods for "Mapper-based classifier" algorithm of:

Mapper Based Classifier
Jacek Cyranka, Alexander Georges, David Meyer
arxiv:1910.08103

Thanks to Alexander Georges and Jacek Cyranka for helpful discussions, and especially to Jacek Cyranka for
sharing a Jupyter notebook implementation of the algorithm on which much of the code below is based

"""

import numpy as np
import sklearn
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from gtda.mapper import FirstSimpleGap, Projection, OneDimensionalCover, make_mapper_pipeline
import pickle
from joblib import Parallel, delayed
import pathlib
import logging

log = logging.getLogger("TR_logger")

TEMP_DATA = str(pathlib.Path(__file__).parent.parent.joinpath('./temp_data/').absolute())+'/'


class MapperClassifier:
    """
    Implements topological encoding portion of mapper classifier algorithm
    To complete mapper classifier algorithm, should train output on ordinary neural network

    Implements batching using helper-class MapperClassifierBatch

    n_components: number of component PCA projection used on input data
    NRNN: number of nearest neighbors to use whenn projecting test points to clusters
    n_intervals: number of intervals used in constructing mapper graphs
    overlap_frac: overlap of intervals
    delta: parameter increasing effective size of intervals when projecting test points
    mu: parameter used in weighing train points when projecting test data
    """

    def __init__(self, n_batches=1, n_components=1, NRNN=3, remake=True, n_intervals=10, overlap_frac=0.33, delta=0.1, mu = 1e-05, label='mapper'):
        if not(type(n_batches) is int and n_batches>0):
            raise ValueError('n_batches must be a positive integer')
        self.n_batches = n_batches
        self.mappers = [MapperClassifierBatch(n_components=n_components, NRNN=NRNN, remake=remake,
                                         n_intervals=n_intervals, overlap_frac=overlap_frac,
                                         delta=delta, mu=mu, label=label+'_'+str(i)) for i in range(self.n_batches)]

    def fit(self, data):
        self.data_batches = np.array_split(data, self.n_batches)
        for mapper, data_batch in zip(self.mappers, self.data_batches):
            log.debug(f"fitting {mapper.label}...")
            mapper.fit(data_batch)

    def fit_transform(self, data):
        self.data_batches = np.array_split(data, self.n_batches)

        # fit mappers along diagonal
        diagonal = []
        for mapper, data_batch in zip(self.mappers, self.data_batches):
            log.debug(f"fitting {mapper.label}...")
            diagonal.append(mapper.fit_transform(data_batch))

        # fill rest of matrix
        output = []
        for i in range(self.n_batches):
            log.debug(f"--->transforming {i}th batch...")
            output.append([])
            for j in range(self.n_batches):
                log.debug(f"------>using {j}th mapper...")
                if i==j:
                    output[-1].append(diagonal[i])
                else:
                    output[-1].append(self.mappers[j].transform(self.data_batches[i]))
            output[-1] = np.concatenate(output[-1], axis=1)

        return np.concatenate(output)

    def transform(self, data):
        output = []
        for mapper in self.mappers:
            output.append(mapper.transform(data))

        return np.concatenate(output, axis=1)


class MapperClassifierBatch:
    """
    Same as above, but handles a single batch

    n_components: number of component PCA projection used on input data
    NRNN: number of nearest neighbors to use whenn projecting test points to clusters
    n_intervals: number of intervals used in constructing mapper graphs
    overlap_frac: overlap of intervals
    delta: parameter increasing effective size of intervals when projecting test points
    mu: parameter used in weighing train points when projecting test data
    """
    def __init__(self, n_components=1, NRNN=3, remake=True, n_intervals=10, overlap_frac=0.33, delta=0.1, mu = 1e-05, label='mapper'):
        log.debug("creating mapper-classifier...")
        self.n_components = n_components
        assert self.n_components >= 1
        self.NRNN = NRNN
        assert self.NRNN >= 1
        self.remake = remake
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.delta = delta
        self.mu = mu
        self.label = label
        self.data = None
        self.data_len, self.data_features, self.mapper_features = None, None, None

    def fit(self, data):
        """
        constructs mapper graphs using train data and uses to transform training points to encoded space

        :param data: numpy array whose rows are training datapoints to fit to mapper-classifier
        :return: None
        """
        if len(data.shape) > 2:
            log.debug("flattening data...")
        self.data =np.reshape(data, (data.shape[0], -1))
        (self.data_len, self.data_features) = self.data.shape

        log.debug(f"fitting mapper with data of total length {self.data_len} and {self.data_features} features.")

        # get (n_components)-dimensional PCA projection of data
        log.debug("--->getting latent space rep...")
        self._getLatentRep()

        # create mapper graphs
        log.debug("--->creating mapper graphs...")
        self._runMapper()

        # create mapper graphs
        log.debug("--->pre-computing nearest neighbor classifiers...")
        self._precomputeNNs()

    def transform(self, test_data=None):
        if test_data is None:
            # uses self.data, set in fit function
            log.debug("transforming with test_data=None (transforming training data)...")

            self._makeGraphBins()
            return self.total_graphbinm

        else:
            # assign train points to graph node bins
            log.debug(f"transforming with test_data.shape = {test_data.shape}..")

            return self._test_transform(test_data)

    def fit_transform(self, data, n_batches=1):
        log.debug("fit_transform in MapperClassifier...")
        self.fit(data)
        return self.transform()

    def _getLatentRep(self):
        """
        gets latent space (PCA) representation of train data

        :return: None
        """
        if not self.remake and os.path.exists(TEMP_DATA + "%s_latent" % self.label):
            frin = open(TEMP_DATA + "%s_latent" % self.label, "rb")
            self.rep = pickle.load(frin)
            return

        log.debug("------> fitting {}-component PCA to data of shape {}...".format(self.n_components, self.data.shape))
        pca = MyPCA(n_components=self.n_components)
        pca.fit(self.data)

        self.rep = pca
        fr = open(TEMP_DATA + "%s_latent" % self.label, "wb")
        pickle.dump(self.rep, fr)
        fr.close()

    def _runMapper(self):
        """
        creates mapper graphs based on train data

        :return: None
        """
        log.debug("--->creating mappers...")
        if not self.remake and os.path.exists(TEMP_DATA + "%s_firstsimplegap_graphs" % self.label):
            fgin = open(TEMP_DATA + "%s_firstsimplegap_graphs" % self.label, "rb")
            self.graphs = pickle.load(fgin)

            fpin = open(TEMP_DATA + "%s_mapper_pipes" % self.label, "rb")
            self.mapper_pipes = pickle.load(fpin)
            return

        clusterer = FirstSimpleGap()
        self.mapper_pipes = []

        log.debug("------> creating projection components...")

        for k in range(self.n_components):
            log.debug("---------> on component {}/{}...".format(k + 1, self.n_components))
            proj = Projection(columns=k)
            filter_func = Pipeline(steps=[('pca', self.rep), ('proj', proj)])
            filtered_data = filter_func.fit_transform(self.data)
            cover = OneDimensionalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac, kind='balanced')
            cover.fit(filtered_data)
            mapper_pipe = make_mapper_pipeline(scaler=None,
                                               filter_func=filter_func,
                                               cover=cover,
                                               clusterer=clusterer,
                                               verbose=(log.getEffectiveLevel() == logging.DEBUG),
                                               n_jobs=1)
            mapper_pipe.set_params(filter_func__proj__columns=k)
            self.mapper_pipes.append(("PCA%d" % (k + 1), mapper_pipe))

        # try parallelization
        log.debug("------> entering parallelization...")

        self.graphs = [mapper_pipe[1].fit_transform(self.data) for mapper_pipe in self.mapper_pipes]

        #
        # self.graphs = Parallel(n_jobs=5, prefer="threads")(
        #     delayed(mapper_pipe[1].fit_transform)(self.data) for mapper_pipe in self.mapper_pipes
        # )

        fg = open(TEMP_DATA + "%s_firstsimplegap_graphs" % self.label, "wb")
        pickle.dump(self.graphs, fg)
        fg.close()

        fp = open(TEMP_DATA + "%s_mapper_pipes" % self.label, "wb")
        pickle.dump(self.mapper_pipes, fp)
        fp.close()

    def _precomputeNNs(self):
        # for each test data, translate its latent representation into interval nrs
        log.debug("--->precomputing NNs for mappers...")
        self.intervals = []
        for mapper_pipe in self.mapper_pipes:
            self.intervals.append(mapper_pipe[1].get_mapper_params()['cover'].get_fitted_intervals())

        # precompute dictionary of fitted NNeighbor for future use
        self.int_nn = {}        # = dictionary with { key = (n (= component), (i,) or  (i,i+1) = interval) :
                                # value = [knn classifier fit to points in interval, data points in interval, ids in interval] }

        for n in range(self.n_components):
            nodeids = [[] for _ in range(len(self.intervals[n]))]
            for i, x in enumerate(self.graphs[n]['node_metadata']['pullback_set_label']):
                nodeids[x] += list(self.graphs[n]['node_metadata']['node_elements'][i])
            for i in range(len(self.intervals[n])):
                if len(nodeids[i]) == 0:
                    continue
                knn = NearestNeighbors(n_neighbors=self.NRNN, metric='euclidean')
                knn.fit(self.data[nodeids[i]])
                # knn is fitted knn using data[nodeids] subset of data nodeids are also saved to further reference
                self.int_nn.update({(n, tuple([i])): [knn, self.data[nodeids[i]], np.array(nodeids[i])]})
                if i > 0:
                    knn = NearestNeighbors(n_neighbors=self.NRNN, metric='euclidean')
                    union = list(set(nodeids[i]) | set(last_nodeids))
                    knn.fit(self.data[union])
                    self.int_nn.update({(n, tuple([i - 1, i])): [knn, self.data[union], np.array(union)]})
                last_nodeids = nodeids[i]

    def _makeGraphBins(self):
        """
        transforms training points into bins according to which nodes
        of the mapper graphs they lie in.  result is a binary vector of shape
        
        (self.data_len, self.mapper_features)
        
        which is placed in self.total_graphbinm
        """
        log.debug("--->assigning train points to graph node bins...")
        
        if not self.remake and os.path.exists(TEMP_DATA + 'matrix_train_%s.csv' % self.label):
            self.total_graphbinm = np.loadtxt(TEMP_DATA + 'matrix_train_%s.csv' % self.label, delimiter=',', fmt='%f')
            return

        total_graphbin = []

        self.mapper_features = 0
        for n in range(self.n_components):
            graph = self.graphs[n]  # graph created for given projection component
            nrnodes = len(graph['node_metadata']['node_id'])  # number of nodes in graph
            graphbin = np.zeros((self.data_len, nrnodes), dtype=int)  # matrix indicating which nodes train data lies in
            for node in range(nrnodes):
                for pt in graph['node_metadata']['node_elements'][node]:
                    graphbin[pt][node] = 1

            total_graphbin.append(graphbin)
            self.mapper_features += graphbin.shape[1]

        self.total_graphbinm = np.hstack(total_graphbin)

        np.savetxt(TEMP_DATA + 'matrix_train_%s.csv' % self.label, self.total_graphbinm, delimiter=',', fmt='%f')

    def _test_transform(self, test_data_):
        """
        transforms test data points to mapper representation
        assumes fit_transform has already been called on training dataset

        :param test_data_: data to be transformed
        :return: transformed test points
        """
        log.debug("--->assigning test points to graph node bins...")

        if len(test_data_.shape) > 2:
            log.debug("flattening data...")
        test_data = np.reshape(test_data_, (test_data_.shape[0], -1))


        if not self.remake and os.path.exists(TEMP_DATA + 'matrix_test_%s.csv' % self.label):
            total_test_rep = np.loadtxt(TEMP_DATA + 'matrix_test_%s.csv' % self.label, delimiter=',', fmt='%f')
            return total_test_rep


        assert test_data.shape[1]==self.data_features
        test_data_len = test_data.shape[0]

        # project the test data
        latent_test_data = self.rep.transform(test_data)


        # for each test point compute its representation by finding its NNeighbors in train set

        def find_alphas(val, intervals, delta=self.delta):
            alphas = []
            if val < intervals[0][1]:
                return tuple([0])
            if val > intervals[-1][0]:
                return tuple([len(intervals) - 1])
            for n, intv in enumerate(intervals):
                midv = (intv[0] + intv[1]) / 2
                if abs(val - midv) < abs(intv[1] - intv[0]) * (1 + delta) / 2:
                    alphas.append(n)
            return tuple(alphas[:2])

        # for each test point,  alphas = list of intervals containing point for each component
        fullalphas = []
        for latent_testp in latent_test_data:
            alphas = []
            for n, latentv in enumerate(latent_testp):
                alphas.append(find_alphas(latentv, self.intervals[n]))
            fullalphas.append(alphas)

        test_rep = np.zeros((test_data_len, self.mapper_features))

        for i in range(test_data_len):
            nncompids = []
            nncompdists = []
            for n in range(self.n_components):
                inn = self.int_nn.get((n, fullalphas[i][n]))
                try:
                    if len(inn[2]) > self.NRNN:
                        knn = inn[0].kneighbors([test_data[i]])
                        ids = inn[2][knn[1]]
                        nncompids.append(ids.squeeze())
                        nncompdists.append(knn[0].squeeze())
                    else:
                        ids = inn[2]  # all available points
                        dists = np.linalg.norm(test_data[i] - inn[1], axis=1)
                        nncompids.append(ids)
                        nncompdists.append(dists)
                except:
                    print(inn)
                    print((n, fullalphas[i][n]))
                    print(list(self.int_nn.keys()))
                    assert False

            all_ids = np.concatenate([nncompids[j] for j in range(self.n_components)])
            all_dists = np.concatenate([nncompdists[j] for j in range(self.n_components)])
            sort_idx = np.argsort(all_dists)[:self.NRNN]
            best_ids = all_ids[sort_idx]
            best_dists = all_dists[sort_idx]

            ns = np.array(best_dists[:])
            ns = 1. / (ns + self.mu)
            ns = ns / sum(ns)
            features = ns.reshape(1, -1).dot(self.total_graphbinm[best_ids].astype(float)).squeeze()
            test_rep[i] = features

        return test_rep

class MyPCA(sklearn.decomposition.PCA):
    def fit_transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return super().transform(X)

class LatentRep():
    # assumes that all proj in self.projs have .fit and .transform() methods
    def __init__(self, projectors, label):
        self.projectors = projectors
        self.label = label

    def fit(self, *args, **kwargs):
        for proj in self.projectors:
            proj.fit(*args, **kwargs)

    def transform(self, X):
        rep = []
        for proj in self.projectors:
            rep.append(proj.transform(X))
        # merge all projectors into one vector
        return np.hstack(rep)


# TESTING

def run_tests():
    # set random seed for consistent tests
    np.random.seed(42)

    mapper = MapperClassifier()
    X_train = np.random.rand(200, 100)
    X_train_map = mapper.fit_transform(X_train)
    assert X_train_map.shape == (200, 285)
    X_test = np.random.rand(100, 100)
    X_test_map = mapper.transform(X_test)
    assert X_test_map.shape == (100, 285)

    log.info("mapper_class tests passed!")

if __name__ == '__main__':
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    run_tests()
