import numpy as np

class ALS():
  def __init__(self, n_factors=32, reg=0.1, n_iters=10):
    self.n_factors = n_factors
    self.reg = reg
    self.n_iters = n_iters

  def fit(self, R):
    Rt = R.T.tocsr()
    n_users, n_items = R.shape

    self.mu = R.data.mean()
    self.P = np.zeros((n_users, self.n_factors))
    self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

    for it in range(self.n_iters):
      self.P = self.update(R, self.Q)
      self.Q = self.update(Rt, self.P)
    return self

  def update(self, M, other):
    new = np.zeros((M.shape[0], self.n_factors))
    for x in range(M.shape[0]):
      a, b = M.indptr[x], M.indptr[x + 1]
      idx = M.indices[a:b]
      r = M.data[a:b] - self.mu

      F = other[idx]
      A = F.T @ F + self.reg * len(idx) * np.eye(self.n_factors)
      new[x] = np.linalg.solve(A, F.T @ r)
    return new

  def predict(self, users):
    return self.mu + self.P[users] @ self.Q.T