import numpy as np
import scipy.sparse as sp

class CollabFiltration():
  def __init__(self, R):
    n_users = R.shape[0]

    self.user_mean = np.asarray(R.sum(axis=1)).ravel() / R.getnnz(axis=1)

    self.Rc = R.copy()
    for u in range(n_users):
      a, b = self.Rc.indptr[u], self.Rc.indptr[u + 1]
      self.Rc.data[a:b] -= self.user_mean[u]

    norms = np.sqrt(np.asarray(self.Rc.multiply(self.Rc).sum(axis=1)).ravel())
    self.Rn = self.Rc.copy()
    for u in range(n_users):
      if norms[u] > 0:
        a, b = self.Rn.indptr[u], self.Rn.indptr[u + 1]
        self.Rn.data[a:b] /= norms[u]

    self.B = R.copy()
    self.B.data[:] = 1

  def fit(self, k=200):
    S = (self.Rn @ self.Rn.T).toarray()
    np.fill_diagonal(S, 0)
    S[S < 0] = 0

    for u in range(S.shape[0]):
      border = np.sort(S[u])[-k]
      S[u][S[u] < border] = 0

    self.S = sp.csr_matrix(S)
    self.Sabs = abs(self.S)
    return self

  def predict(self, users):
    num = (self.S[users] @ self.Rc).toarray()
    den = (self.Sabs[users] @ self.B).toarray()
    den[den == 0] = 1
    return self.user_mean[users][:, None] + num / den