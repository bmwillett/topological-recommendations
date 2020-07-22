from lib.data_class import *
from models.main_models import *
from models.latent_models import *
from models.feature_models import *
from models.baseline_models import *
from lib.process_data import *
import logging

logging.basicConfig()
log = logging.getLogger("TR_logger")
log.setLevel(logging.DEBUG)

USE_DATASET = 'small'
IC_DATA_DIR = {'tiny': './data/instacart_2017_05_01_tiny/',
               'small': './data/instacart_2017_05_01_small/',
               'medium': './data/instacart_2017_05_01_medium/',
               'full': './data/instacart_2017_05_01/'}[USE_DATASET]

# load data (using hard-coded function for Instacart dataset)
order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)


# create train, test, val datasets
IC_dataset = DataSet(order_data, product_data)
train_dataset, test_val_dataset = IC_dataset.train_test_split(test_frac=0.3)
val_dataset, test_dataset = test_val_dataset.train_test_split(test_frac=0.5)
del test_val_dataset

# make adversarial tests
test_adv = test_dataset.make_adversarial(num_switches=1)


# # create and train baseline random model
# rmodel = RandomModel()
# rmodel.fit(train_dataset)
# rmodel.find_threshold(val_dataset)
# rmodel.get_stats(test_dataset, plot_roc=True)
#
# r_metrics = rmodel.evaluate(test_dataset)
# r_metrics_adv = rmodel.evaluate(test_adv)


# create and train baseline LGBM model
lgmodel = LGBoostModel()
lgmodel.fit(train_dataset)
lgmodel.find_threshold(val_dataset)

lgmodel.get_stats(test_dataset, plot_roc=True)
lg_metrics = lgmodel.evaluate(test_dataset)
lg_metrics_adv = lgmodel.evaluate(test_adv)


# create and train main non-topological model

# first create latent and feature models, fit and transform
user_latent = UserModel()
user_latent.fit(train_dataset, epochs=20)

product_latent = ProductModel()
product_latent.fit(train_dataset)

feature_model = MainFeatureModel()

# fit non-top model to train_dataset
ntmodel = NonTopModel(user_latent_model=user_latent, product_latent_model=product_latent, feature_model=feature_model)
ntmodel.fit(train_dataset, fit_latent=False, epochs=20)
ntmodel.find_threshold(val_dataset)

ntmodel.get_stats(test_dataset, plot_roc=True)
nt_metrics = ntmodel.evaluate(test_dataset)
nt_metrics_adv = ntmodel.evaluate(test_adv)


# create and train main topological model

# use same pretrained feature and latent models as above
# fit non-top model to train_dataset
tmodel = TopModel(user_latent_model=user_latent, product_latent_model=product_latent, feature_model=feature_model)
tmodel.fit(train_dataset, fit_latent=False, epochs=20)
tmodel.find_threshold(val_dataset)

tmodel.get_stats(test_dataset, plot_roc=True)
t_metrics = tmodel.evaluate(test_dataset)
t_metrics_adv = tmodel.evaluate(test_adv)

import IPython; IPython.embed()
