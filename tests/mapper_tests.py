import sys

sys.path.append('../lib/')

from mapper_tools import *
from mapper_class import MapperClassifier

# file directories
DATA_DIR_TRAIN = '../data/data_train_fashion_unnormalized/'
DATA_DIR_TEST = '../data/data_test_fashion_unnormalized/'
SMALL_DATA_DIR_TRAIN = '../data/small_data_train_fashion_unnormalized/'
SMALL_DATA_DIR_TEST = '../data/small_data_test_fashion_unnormalized/'
FILE_TRAIN = FILE_TEST = 'trueexamples_in.csv'

n_components=15 # number of components in projection
NRNN = 3 # number of nearest neighbors to use in projecting test data
label = "PCA%d" % (n_components)


# load data
data, data_header, datatest, datatest_header = loadMapperData(data_dir_train = SMALL_DATA_DIR_TRAIN,
                                      data_dir_test = SMALL_DATA_DIR_TEST)


mapper = MapperClassifier(n_components=n_components, NRNN=NRNN)

total_graphbinm = mapper.fit(data, data_header)

total_test_rep = mapper.project(datatest, datatest_header)

print(total_graphbinm.shape)
print(total_graphbinm[0][0:10])
print(total_test_rep.shape)
print(total_test_rep[0][0:10])


# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
# # # Training of a neural classifier
# experiment_stamp = 'PCA15'
#
# datatrain = pd.read_csv( 'matrix_train_%s.csv' % (experiment_stamp), sep = ',', header=None, error_bad_lines=False )
# datatrain = pd.DataFrame( datatrain, dtype= float )
#
# nrcols = datatrain.iloc[0, 3:].shape[0]
# N, D_in, H1, H2, D_out = datatrain.shape[0], nrcols, 2000, 1000, 10
#
# learning_rate = 0.001
#
# archstamp = '%dnodes_%flr' % (H1, learning_rate)
#
# print('nrcols in data %d' % (nrcols))
# print('nr of examples=%d' % (datatrain.shape[0]))
#
# x = torch.Tensor(np.array(datatrain.iloc[:, 3:]))
# y = np.squeeze(torch.LongTensor(datatrain.iloc[:,1]))
#
# if verbose>0:
#     print(y)
#
# #training is super-slow on CPU,
# #but can use Google Colab to train and download the model
# #just use this link (may need to upload train_matrix.csv to colab)
# #https://colab.research.google.com/drive/1iSYwm6z8amKwL_BZbt3npE7Dsv_8nmxm
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H1),
#     torch.nn.Sigmoid(),
#     #torch.nn.Dropout(p=0.5),
#     torch.nn.Linear(H1, H2),
#     torch.nn.Sigmoid(),
#     #torch.nn.Dropout(p=0.5),
#     #torch.nn.Linear(H2, H3),
#     #torch.nn.ReLU(),
#     ##MNIST -- 2 layer
#     torch.nn.Linear(H2, D_out),
#     ##CIFAR -- 3 layer
#     #torch.nn.Linear(H3, D_out),
#     torch.nn.LogSoftmax(dim = 1),
# )#.cuda()
#
# loss_fn = torch.nn.NLLLoss()
#
# losses = []
#
# batchs = 256
#
# trainset = torch.utils.data.TensorDataset(x, y)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchs,
#                                           shuffle=True, num_workers=1)
#
# optimizer = optim.Adam( model.parameters(), lr=learning_rate )
#
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#
# batchesn = int( N /  batchs )
# EPOCHS = 10
#
#
#
# for epoch in range(EPOCHS):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, datap in enumerate(trainloader, 0):
#         #print(i)
#         # get the inputs
#         inputs, labels = datap
#         #inputs = inputs.to('cuda', non_blocking=True)
#         #labels = labels.to('cuda', non_blocking=True)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % batchesn == batchesn-1:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.10f' %
#                   (epoch + 1, i + 1, running_loss / batchesn))
#             running_loss = 0.0
#     scheduler.step()
#
#
# np.save("../experiments/losses.npy", np.array(losses))
# filen = "matrix_trueexamples_in_merged_mappers_all_%s_full" % ( experiment_stamp )
# model_filen = filen + '_' + archstamp + '_' + 'fashion10k' + '.pt'
#
# torch.save(model, model_filen)
#
#
# # # testing the classifier
#
# model_filen = 'matrix_trueexamples_in_merged_mappers_all_PCA15_full_2000nodes_0.001000lr_fashion10k'  + '.pt'
# model = torch.load(model_filen, map_location=torch.device('cpu'))
#
# x = torch.Tensor(np.array(datatrain.iloc[:, 3:]))
# y = np.squeeze(torch.LongTensor(datatrain.iloc[:,1]))
#
# model.eval()
# outputs = model(x)
# probals, predicted = torch.max(outputs, 1)
#
# c = (predicted == y).squeeze()
# correct = c.sum().item()
#
# print('# train samples correct=%d' % (correct))
# print('training accuracy=%f' % ((float)(correct) / N))
#
# testdata = pd.read_csv( 'matrix_test_%s.csv' % (label) , sep = ',', header=None )
# testdata = pd.DataFrame( testdata, dtype= float )
#
# nrcols = testdata.iloc[0, 3:].shape[0]
# print('nrcols in test data %d' % (nrcols))
# print('nr of test examples=%d' % (testexamples
#                                  ))
#
# x = torch.Tensor(np.array( testdata.iloc[:, 3:] ))
# y = np.squeeze(torch.LongTensor(testdata.iloc[:, 1]))
#
# model.eval()
# outputs = model(x)
# probals, predicted = torch.max(outputs, 1)
#
# c = (predicted == y).squeeze()
# correct = c.sum().item()
#
#
# print('# test samples correct=%d' % (correct))
# print('test accuracy=%f' % ((float)(correct) / testexamples))
#
