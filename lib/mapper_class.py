from mapper_tools import *

class MapperClassifier:
    def __init__(self, n_components=1, NRNN=3):
        self.n_components = n_components
        self.NRNN = NRNN

    def fit(self, data, data_header):
        self.data, self.data_header = data, data_header

        # get n_copmonents-dimensional PCA projection of data
        self._getLatentRep()

        # create mapper graphs
        self._runMapper()

        # assign train points to graph node bins
        self._makeGraphBins()

        return self.total_graphbinm

    def _getLatentRep(self, remake=True):
        self.rep = getLatentRep(self.n_components, self.data, remake=remake)

    def _runMapper(self, remake=True):
        self.graphs, self.mapper_pipes = runMapper(self.n_components, self.data, self.rep, remake=remake)

    def _makeGraphBins(self, remake=True):
        self.total_graphbinm, self.featlen = makeGraphBins(self.n_components, self.data, self.data_header, self.graphs, remake=remake)

    def project(self, datatest, datatest_header, remake=True):
        self.total_test_rep = projectTestData(self.n_components, self.rep, self.data, datatest, datatest_header,
                            self.graphs, self.mapper_pipes, self. total_graphbinm, self.featlen, self.NRNN, remake=remake)

        return self.total_test_rep
