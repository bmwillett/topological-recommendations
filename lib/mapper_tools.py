from mapper_tools import *
import pandas as pd
import numpy as np
import sklearn
from gtda.mapper import *
import pickle
from joblib import Parallel, delayed

DATA_DIR_TRAIN = '../data/data_train_fashion_unnormalized/'
DATA_DIR_TEST = '../data/data_train_fashion_unnormalized/'
FILE_TRAIN = FILE_TEST = 'trueexamples_in.csv'


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

def getLatentRep(mode='train', data_dir_train=DATA_DIR_TRAIN, data_dir_test=DATA_DIR_TEST):
    # load data
    filepath=DATA_DIR_TRAIN+FILE_TRAIN if mode=='train' else DATA_DIR_TRAIN+FILE_TRAIN
    df = pd.read_csv(filepath, header=None)
    data = df.values
    data_header = data[:, :3]
    data = data[:, 4:]

    # PCA projections
    n_components = 15
    label = "PCA%d" % (n_components)
    pca = MyPCA(n_components=n_components)
    pca.fit(data)

    rep = LatentRep([pca], label)
    fr = open("%s_latent" % (label), "wb")
    pickle.dump(rep, fr)
    fr.close()

    return pca, data, data_header

def runMapper(n_components, data, pca):
    clusterer = FirstSimpleGap()
    mapper_pipes = []
    for k in range(n_components):
        proj = Projection(columns = k)
        filter_func = Pipeline(steps=[('pca', pca), ('proj', proj)])
        cover = OneDimensionalCover(n_intervals=10, overlap_frac=0.33)
        mapper_pipe = make_mapper_pipeline(scaler=None,
                                       filter_func = filter_func,
                                       cover=cover,
                                       clusterer=clusterer,
                                       verbose=True,
                                       n_jobs = 1)
        mapper_pipe.set_params(filter_func__proj__columns = k)
        mapper_pipes.append( ("PCA%d" % (k+1), mapper_pipe ) )

    # try parallelization

    graphs = Parallel(n_jobs=5, prefer="threads")(
        delayed(mapper_pipe[1].fit_transform)(data) for mapper_pipe in mapper_pipes
    )


    fg = open( "%s_firstsimplegap_graphs" % (label), "wb")
    pickle.dump(graphs, fg)
    fg.close()

    fp = open("%s_mapper_pipes" % (label) , "wb")
    pickle.dump(mapper_pipes, fp)
    fp.close()
