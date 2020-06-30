"""
Definition of feature models

these extract features directly from the dataset, rather than the
latent models which train a machine learning algorithm to find features.  These are used
together with latent models to build up recommendation models
"""
import numpy as np
import lib.tools as tools
import logging

log = logging.getLogger("TR_logger")

class FeatureModel:
    """
    Base feature model other feature models inherit from
    """
    def transform(self, dataset):
        return None


class EmptyFeatureModel(FeatureModel):
    """
    Empty feature model
    """
    def transform(self, dataset):
        return np.empty(shape=(dataset.size, 0))


class MainFeatureModel(FeatureModel):
    """
    Feature model used by main recommendation models
    """
    def transform(self, dataset, drop_categorical=True, return_df=False):
        """
        extracts features from dataset and returns as numpy array

        :param dataset: dataset to extract features from
        :param drop_categorical: if true, drop categorical features
        :param return_df: if true, return result as pandas dataframe, else as numpy array of values
        :return: numpy array with features for each sample in dataset
        """
        return tools.get_features(dataset, drop_categorical=drop_categorical, return_df=return_df)


# TESTING

def run_tests(IC_DATA_DIR):
    from lib.process_data import instacart_process
    from lib.data_class import DataSet

    # set random seed for consistent tests
    np.random.seed(42)

    # load data from instacart csv files (values below use testing directory)
    order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)

    # create dataset
    ic_dataset = DataSet(order_df=order_data, product_df=product_data)

    # check dataframes created correctly
    assert ic_dataset.order_df.shape == (31032, 4)
    assert ic_dataset.product_df.shape == (6126, 5)

    # create feature model and transform dataset
    feature_model = MainFeatureModel()
    X = feature_model.transform(ic_dataset)
    assert X.shape == (12214, 8)

    feature_model = MainFeatureModel()
    X = feature_model.transform(ic_dataset, return_df=True, drop_categorical=False)
    assert X.shape == (12214, 10)

    log.info("feature_models tests passed!")


if __name__ == '__main__':
    IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    run_tests(IC_DATA_DIR)
