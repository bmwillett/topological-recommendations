import numpy as np
import sklearn
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from gtda.mapper import FirstSimpleGap, Projection, OneDimensionalCover, make_mapper_pipeline
import pickle
from joblib import Parallel, delayed
from contextlib import contextmanager
from timeit import default_timer

TEMP_DATA = './temp_data/'

class MapperClassifier:
    def __init__(self, n_components=1, NRNN=3, remake=True, n_intervals=10, overlap_frac=0.33, delta=0.1, verbose=0):
        self.n_components = n_components
        self.NRNN = NRNN
        self.remake = remake
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.delta = delta
        self.verbose = verbose
        self.label = "PCA%d" % self.n_components
        self.data = None

    def fit(self, data):
        if len(data.shape)>2 and self.verbose>0:
            print("flattening data...")
        self.data = np.reshape(data, (data.shape[0],-1))
        self.data_len, self.data_features = self.data.shape

        if self.verbose>0:
            print("fitting mapper...")

        # get (n_components)-dimensional PCA projection of data
        if self.verbose>0:
            print("--->getting latent space rep...")
        self._getLatentRep()

        # create mapper graphs
        if self.verbose>0:
            print("--->creating mapper graphs...")

        self._runMapper()

        # assign train points to graph node bins
        if self.verbose>0:
            print("--->assigning train points to graph node bins...")
        self._makeGraphBins()

        return self.total_graphbinm

    def _getLatentRep(self):

        if not self.remake and os.path.exists("%s_latent" % self.label):
            frin = open(TEMP_DATA + "%s_latent" % self.label, "rb")
            self.rep = pickle.load(frin)
            return

        if self.verbose>0:
            print("------> fitting {}-component PCA to data of shape {}...".format(self.n_components, self.data.shape))
        pca = MyPCA(n_components=self.n_components)
        pca.fit(self.data)

        self.rep = pca #LatentRep([pca], self.label)
        fr = open(TEMP_DATA + "%s_latent" % self.label, "wb")
        pickle.dump(self.rep, fr)
        fr.close()

    def _runMapper(self):
        if not self.remake and os.path.exists("%s_firstsimplegap_graphs" % self.label):
            fgin = open(TEMP_DATA + "%s_firstsimplegap_graphs" % self.label, "rb")
            self.graphs = pickle.load(fgin)

            fpin = open(TEMP_DATA + "%s_mapper_pipes" % self.label, "rb")
            self.mapper_pipes = pickle.load(fpin)
            return

        clusterer = FirstSimpleGap()
        self.mapper_pipes = []
        pca = self.rep

        if self.verbose>0:
            print("------> creating projection components...")

        for k in range(self.n_components):
            if self.verbose>0:
                print("---------> on component {}/{}...".format(k + 1, self.n_components))
            proj = Projection(columns=k)
            filter_func = Pipeline(steps=[('pca', pca), ('proj', proj)])
            cover = OneDimensionalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)
            mapper_pipe = make_mapper_pipeline(scaler=None,
                                               filter_func=filter_func,
                                               cover=cover,
                                               clusterer=clusterer,
                                               verbose=(self.verbose > 0),
                                               n_jobs=1)
            mapper_pipe.set_params(filter_func__proj__columns=k)
            self.mapper_pipes.append(("PCA%d" % (k + 1), mapper_pipe))

        # try parallelization
        if self.verbose>0:
            print("------> entering parallelization...")
        self.graphs = Parallel(n_jobs=5, prefer="threads")(
            delayed(mapper_pipe[1].fit_transform)(self.data) for mapper_pipe in self.mapper_pipes
        )

        fg = open(TEMP_DATA + "%s_firstsimplegap_graphs" % self.label, "wb")
        pickle.dump(self.graphs, fg)
        fg.close()

        fp = open(TEMP_DATA + "%s_mapper_pipes" % self.label, "wb")
        pickle.dump(self.mapper_pipes, fp)
        fp.close()

    def _makeGraphBins(self):
        # an algorithm for binarizing training points using
        # the computed graph representation
        # computes and outputs a CSV file with all training
        # points representations , used for training a
        # NeuralNet classifier in the next step.

        if not self.remake and os.path.exists(TEMP_DATA + 'matrix_train_%s.csv' % self.label):
            self.total_graphbinm = np.loadtxt(TEMP_DATA + 'matrix_train_%s.csv' % self.label, delimiter=',', fmt='%f')
            return

        total_graphbin = []

        self.mapper_features = 0
        for comp in range(self.n_components):
            graph = self.graphs[comp]  # graph created for given projection component
            nrnodes = len(graph['node_metadata']['node_id'])  # number of nodes in graph
            graphbin = np.zeros((self.data_len, nrnodes), dtype=int)  # matrix indicating which nodes train data lies in
            for node in range(nrnodes):
                for pt in graph['node_metadata']['node_elements'][node]:
                    graphbin[pt][node] = 1

            total_graphbin.append(graphbin)
            self.mapper_features += graphbin.shape[1]

        self.total_graphbinm = np.hstack(total_graphbin)

        np.savetxt(TEMP_DATA + 'matrix_train_%s.csv' % self.label, self.total_graphbinm, delimiter=',', fmt='%f')

    def project(self, test_data_):
        if self.verbose>0:
            print("--->projecting data to grapher bins...")

        if len(test_data_.shape) > 2 and self.verbose > 0:
            print("flattening data...")
        test_data = np.reshape(test_data_, (test_data_.shape[0], -1))


        if not self.remake and os.path.exists(TEMP_DATA + 'matrix_test_%s.csv' % self.label):
            total_test_rep = np.loadtxt(TEMP_DATA + 'matrix_test_%s.csv' % self.label, delimiter=',', fmt='%f')
            return self.total_test_rep


        assert test_data.shape[1]==self.data_features
        test_data_len = test_data.shape[0]

        # project the test data
        latent_test_data = self.rep.transform(test_data)

        # for each test data, translate its latent representation
        # into interval nrs
        intervals = []
        for mapper_pipe in self.mapper_pipes:
            intervals.append(mapper_pipe[1].get_mapper_params()['cover'].get_fitted_intervals())

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
            return tuple(alphas)

        fullalphas = []
        for latent_testp in latent_test_data:
            alphas = []
            for n, latentv in enumerate(latent_testp):
                alphas.append(find_alphas(latentv, intervals[n]))
            fullalphas.append(alphas)

        # seach for NNeighbor in the preimage of intervals
        # precompute NNeighbor for preimage of each interval
        int_nn = {}

        # precompute dictionary of fitted NNeighbor for further use
        for n in range(self.n_components):
            int_preim_int = []
            nodeids = [[] for _ in range(len(intervals[n]))]
            for i, x in enumerate(self.graphs[n]['node_metadata']['pullback_set_label']):
                nodeids[x].append(self.graphs[n]['node_metadata']['node_elements'][i])
            for i in range(len(intervals[n])):
                #nodeids = self.graphs[n]['node_metadata']['node_elements'][i]
                nodeidsf = np.concatenate(nodeids[i])
                knn = NearestNeighbors(n_neighbors=self.NRNN, metric='euclidean')
                knn.fit(self.data[nodeidsf])
                int_preim_int.append(nodeidsf)
                # knn is fitted knn using data[nodeids] subset of data nodeids are also saved to further reference them in the
                # whole data
                int_nn.update({(n, tuple([i])): [knn, self.data[nodeidsf], np.array(nodeidsf)]})
                if i > 0:
                    knn = NearestNeighbors(n_neighbors=self.NRNN, metric='euclidean')
                    union = list(set(nodeidsf) | set(int_preim_int[i - 1]))
                    knn.fit(self.data[union])
                    int_nn.update({(n, tuple([i - 1, i])): [knn, self.data[union], np.array(union)]})

        # for each test point compute its representation by finding its NNeighbor
        mu = 1e-05
        test_rep = np.zeros((test_data_len, self.mapper_features))

        for i in range(test_data_len):
            nncompids = []
            nncompdists = []
            for j in range(self.n_components):
                # compute this for only the first component (computing for all components is prohibitive)
                # compute knn with distances for each test point (search in the preimage of j-th component)
                inn = int_nn.get((j, fullalphas[i][j]))
                if (len(inn[2]) > self.NRNN):
                    knn = inn[0].kneighbors([test_data[i]])
                    ids = inn[2][knn[1]]
                    nncompids.append(ids.squeeze())
                    nncompdists.append(knn[0].squeeze())
                else:
                    ids = inn[2]  # all available points
                    dists = np.linalg.norm(test_data[i] - inn[1], axis=1)
                    nncompids.append(ids)
                    nncompdists.append(dists)

            all_ids = np.concatenate([nncompids[j] for j in range(self.n_components)])
            all_dists = np.concatenate([nncompdists[j] for j in range(self.n_components)])
            sort_idx = np.argsort(all_dists)[:self.NRNN]
            best_ids = all_ids[sort_idx]
            best_dists = all_dists[sort_idx]

            ns = np.array(best_dists[:])
            ns = 1. / (ns + mu)
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