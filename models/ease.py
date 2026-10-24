import numpy as np

class EASE():
  def __init__(self, reg=500):
    self.reg = reg

  def fit(self, R):
    X = R.copy()
    X.data[:] = 1.0
    self.X = X

    G = (X.T @ X).toarray().astype(np.float32)
    G[np.diag_indices_from(G)] += self.reg

    P = np.linalg.inv(G)
    d = np.diag(P).copy()
    P /= -d
    np.fill_diagonal(P, 0)

    self.B = P
    return self

  def predict(self, users):
    return self.X[users] @ self.B