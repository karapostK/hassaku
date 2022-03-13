import torch

from sgd_alg import SGDBaseline, SGDMatrixFactorization, UProtoMF, IProtoMF, UIProtoMF

x = torch.randint(10, size=(5,))
y = torch.randint(10, (5, 3))

alg = SGDBaseline(10, 10)
print(alg.forward(x, y))

alg = SGDMatrixFactorization(10, 10)
print(alg.forward(x, y))

alg = UProtoMF(10, 10)
print(alg.forward(x, y))

alg = IProtoMF(10, 10)
print(alg.forward(x, y))

alg = UIProtoMF(10, 10)
print(alg.forward(x, y))

from scipy import sparse as sp
import numpy as np

mtx = sp.csr_matrix(np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1]]))

from knn_algs import UserKNN, ItemKNN

alg = UserKNN()
alg.fit(mtx)
print(alg.predict(torch.tensor([0, 1]), torch.tensor([[0], [1]])))
alg = ItemKNN()
alg.fit(mtx)
print(alg.predict(torch.tensor([0, 1]), torch.tensor([[0], [1]])))


from mf_algs import RBMF

alg = RBMF(1)
alg.fit(mtx)
print(alg.predict(torch.tensor([0, 1]), torch.tensor([[0], [1]])))