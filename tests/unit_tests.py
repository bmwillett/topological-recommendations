import lib.data_class
import lib.mapper_class
import models.feature_models
import models.latent_models
import models.baseline_models
import models.main_models
import logging

log = logging.getLogger("TR_logger")

IC_DATA_DIR = '../data/instacart_2017_05_01_testing/'
logging.basicConfig()
log.setLevel(logging.DEBUG)

# run all tests
lib.data_class.run_tests(IC_DATA_DIR)
lib.mapper_class.run_tests()
models.feature_models.run_tests(IC_DATA_DIR)
models.latent_models.run_tests(IC_DATA_DIR)
models.baseline_models.run_tests(IC_DATA_DIR)
models.main_models.run_tests(IC_DATA_DIR)
log.info("all tests passed!")

#TODO:
# - fix mapper class tests
# - get mnist tests working
# - remake file to run models (model A, model B, lgboost)
# - run on tiny, small datasets
# - see how will run on medium/large, think about cloud
# - update plots in talk if applicable