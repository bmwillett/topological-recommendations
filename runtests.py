from lib.data_class import *
from models.main_models import *
from models.baseline_models import *
from lib.process_data import *
import logging

logging.basicConfig()
log = logging.getLogger("TR_logger")
log.setLevel(logging.DEBUG)

USE_SMALL = True
IC_DATA_DIR = './data/instacart_2017_05_01_small/' if USE_SMALL else './data/instacart_2017_05_01/'


# load data (using hard-coded function for Instacart dataset)
order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)


# create datasets
IC_dataset = DataSet(order_data, product_data)
train_dataset, val_dataset, test_dataset = IC_dataset.test_train_val_split()

test_adv = test_dataset.make_adversarial()


# create and train baseline random model
rmodel = RandomModel()
rmodel.fit(train_dataset)
t = rmodel.find_threshold(val_dataset)
print("random: t=",t)

prec_r, rec_r, f1_r = rmodel.accuracy_test(test_dataset, threshold=t)
prec_r_adv, rec_r_adv, f1_r_adv = rmodel.accuracy_test(test_adv, threshold=t)


# create and train baseline LGBM model
lgmodel = LGBoostModel()
lgmodel.fit(train_dataset)
t = lgmodel.find_threshold(val_dataset)
print("lgmodel: t=",t)

prec_lg, rec_lg, f1_lg = lgmodel.accuracy_test(test_dataset, threshold=t)
prec_lg_adv, rec_lg_adv, f1_lg_adv = lgmodel.accuracy_test(test_adv, threshold=t)


# # create and train baseline RandomForest model
# rfmodel = RandomForestModel()
# rfmodel.fit(train_dataset)
# t = rfmodel.find_threshold(val_dataset)
# print("rfmodel: t=",t)
#
# prec_rf, rec_rf, f1_rf = rfmodel.accuracy_test(test_dataset, threshold=t))
# prec_rf_adv, rec_rf_adv, f1_rf_adv = rfmodel.accuracy_test(test_adv, threshold=t))


# create and train UP latent model
upmodel = UPLModel()
upmodel.fit(train_dataset,epochs=10)
t = upmodel.find_threshold(val_dataset)
print("upmodel: t=",t)

prec_nt, rec_nt, f1_nt = upmodel.accuracy_test(test_dataset, threshold=t)
prec_nt_adv, rec_nt_adv, f1_nt_adv = upmodel.accuracy_test(test_adv, threshold=t)

# create and train topological UP latent model
tupmodel = TUPLModel(n_components=4)
tupmodel.fit(train_dataset)
t = tupmodel.find_threshold(val_dataset)
print("tupmodel: t=",t)

prec_t, rec_t, f1_t = tupmodel.accuracy_test(test_dataset, threshold=t)
prec_t_adv, rec_t_adv, f1_t_adv = tupmodel.accuracy_test(test_adv, threshold=t)


print("accuracy test for random model: prec = %f, rec = %f, f1 = %f" % (prec_r, rec_r, f1_r))
print("robustness test for random model: prec = %f, rec = %f, f1 = %f" % (prec_r_adv, rec_r_adv, f1_r_adv))

print("accuracy test for LGBoost model: prec = %f, rec = %f, f1 = %f" % (prec_lg, rec_lg, f1_lg))
print("robustness test for LGBoost model: prec = %f, rec = %f, f1 = %f" % (prec_lg_adv, rec_lg_adv, f1_lg_adv))

# print("accuracy test for RandomForest model: prec = %f, rec = %f, f1 = %f" % (prec_rf, rec_rf, f1_rf))
# print("robustness test for RandomForest model: prec = %f, rec = %f, f1 = %f" % (prec_rf_adv, rec_rf_adv, f1_rf_adv))

print("accuracy test for UPL model: prec = %f, rec = %f, f1 = %f" % (prec_nt, rec_nt, f1_nt))
print("robustness test for UPL model: prec = %f, rec = %f, f1 = %f" % (prec_nt_adv, rec_nt_adv, f1_nt_adv))

print("accuracy test for topological model: prec = %f, rec = %f, f1 = %f" % (prec_t, rec_t, f1_t))
print("robustness test for topological model: prec = %f, rec = %f, f1 = %f" % (prec_t_adv, rec_t_adv, f1_t_adv))






# TODO:
#  - add remaining product latent space (if reasonable, and make new plots)
#  - modify models to save data
#  - do simple checks that this model is running correctly
#  - normalize inputs? other tuning of model (layers, etc)
#  - try to improve performance (implement dask? knn with gpu?)
#  - implement on cloud
#  - scale up dataset as much as possible


#
#
# EPOCHS = 3
# BATCH_SIZE = 32
# input_size = len(IC_dataset.product_ids)
#
#
# # create autoencoder model
# aem = AEM(input_size)
# aem.fit(IC_train, epochs=EPOCHS)
#
# print("accuracy test for AEM: prec = %f, rec = %f, NDCG = %f" % aem.accuracy_test(IC_test))
# print("robustness test for AEM: prec = %f, rec = %f, NDCG = %f" % aem.robustness_test(IC_test))
#
#
# # create topological autoencoder model
# taem = TAEM(input_size)
# taem.fit(IC_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
#
# print("accuracy test for TAEM: prec = %f, rec = %f, NDCG = %f" % taem.accuracy_test(IC_test))
# print("robustness test for TAEM: prec = %f, rec = %f, NDCG = %f" % taem.robustness_test(IC_test))
