from models.data_class import *
from models.model_class import *
from lib.process_data import *

USE_SMALL = True
IC_DATA_DIR = './data/instacart_2017_05_01_small/' if USE_SMALL else './data/instacart_2017_05_01/'


# load data (using hard-coded function for Instacart dataset)
order_data, product_data = instacart_process(data_dir=IC_DATA_DIR)


# create data (using library methods)
IC_dataset = DataSet(order_data, product_data)
IC_train, IC_test = IC_dataset.split()


EPOCHS = 3
BATCH_SIZE = 32
input_size = len(IC_dataset.product_ids)


# create autoencoder model
aem = AEM(input_size)
aem.fit(IC_train, epochs=EPOCHS)

print("accuracy test for AEM: prec = %f, rec = %f, NDCG = %f" % aem.accuracy_test(IC_test))
print("robustness test for AEM: prec = %f, rec = %f, NDCG = %f" % aem.robustness_test(IC_test))


# create topological autoencoder model
taem = TAEM(input_size)
taem.fit(IC_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("accuracy test for TAEM: prec = %f, rec = %f, NDCG = %f" % taem.accuracy_test(IC_test))
print("robustness test for TAEM: prec = %f, rec = %f, NDCG = %f" % taem.robustness_test(IC_test))
