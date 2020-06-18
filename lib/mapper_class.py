from mapper_tools import *
import logging
log = logging.getLogger("TR_logger")

class MapperClassifier:
    def __init__(self, n_components=1, NRNN=3, remake=True):
        self.n_components = n_components
        self.NRNN = NRNN
        self.remake = remake

    def fit(self, data, data_header):
        self.data, self.data_header = data, data_header
        if log.level==logging.DEBUG:
            print("fitting mapper...")

        # get n_copmonents-dimensional PCA projection of data
        if log.level == logging.DEBUG:
            print("--->getting latent space rep...")
        self._getLatentRep()

        # create mapper graphs
        if log.level == logging.DEBUG:
            print("--->creating mapper graphs...")
        self._runMapper()

        # assign train points to graph node bins
        if log.level == logging.DEBUG:
            print("--->assigning train points to graph node bins...")
        self._makeGraphBins()

        return self.total_graphbinm

    def _getLatentRep(self):
        self.rep = getLatentRep(self.n_components, self.data, remake=self.remake)

    def _runMapper(self):
        self.graphs, self.mapper_pipes = runMapper(self.n_components, self.data, self.rep, remake=self.remake)

    def _makeGraphBins(self):
        self.total_graphbinm, self.featlen, self.total_graphbin = makeGraphBins(self.n_components, self.data, self.data_header, self.graphs, remake=self.remake)

    def project(self, datatest, datatest_header):
        if log.level == logging.DEBUG:
            print("--->projecting data to grapher bins...")
        self.total_test_rep = projectTestData(self.n_components, self.rep, self.data, datatest, datatest_header,
                            self.graphs, self.mapper_pipes, self. total_graphbinm, self.featlen, self.NRNN, remake=self.remake)

        return self.total_test_rep
