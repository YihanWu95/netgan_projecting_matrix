import numpy as np

seed = 42
np.random.seed(seed)

dim = 10

mean1 = np.zeros(dim)
cov1 = np.ones((dim,dim))/2 + np.diag(np.ones(dim))

train_data1 = np.random.multivariate_normal(mean = mean1, cov = cov1, size = 1000)
print(np.shape(train_data1))
print(train_data1[1])

mean2 = np.ones(dim)*10
cov2 = np.ones((dim,dim))/2 + np.diag(np.ones(dim)/2)

train_data2 = np.random.multivariate_normal(mean = mean2, cov = cov2, size = 1000)
print(np.shape(train_data2))
print(train_data2[2])

train_data = np.concatenate((train_data1,train_data2),axis=0)
print(np.shape(train_data))
np.save("../train_data.npy",train_data)