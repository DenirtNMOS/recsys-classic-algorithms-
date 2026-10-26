import numpy as np
import scipy.sparse as sp
import torch

class TimeSVDpp():
  def __init__(self, n_factors=16, reg=0.05, lr=0.01, n_epochs=10, n_bins=10, beta=0.4, use_time=True, batch_size=4096):
    self.n_factors = n_factors
    self.reg = reg
    self.lr = lr
    self.n_epochs = n_epochs
    self.n_bins = n_bins if use_time else 1
    self.beta = beta
    self.use_time = use_time
    self.batch_size = batch_size

  def fit(self, R, T):
    n_users, n_items = R.shape
    k = self.n_factors
    year = 86400 * 365

    coo = R.tocoo()
    u, i, r = coo.row, coo.col, coo.data
    t = np.asarray(T[u, i]).ravel() / year

    cnt = R.getnnz(axis=1)
    self.t_user = np.asarray(T.sum(axis=1)).ravel() / cnt / year

    self.edges = np.quantile(t, np.arange(1, self.n_bins) / self.n_bins)
    bins = np.searchsorted(self.edges, t)

    d = t - self.t_user[u]
    dev = np.sign(d) * np.abs(d) ** self.beta * self.use_time

    B = R.copy()
    B.data[:] = 1
    A = (sp.diags(1 / np.sqrt(cnt)) @ B).tocsr()
    A_coo = A.tocoo()
    A_t = torch.sparse_coo_tensor(
      torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long),
      torch.tensor(A_coo.data, dtype=torch.float32),
      A.shape,
      check_invariants=False,
    )

    self.mu = float(r.mean())
    bu = torch.zeros(n_users, requires_grad=True)
    au = torch.zeros(n_users, requires_grad=True)
    bi = torch.zeros(n_items, requires_grad=True)
    bib = torch.zeros(n_items, self.n_bins, requires_grad=True)
    P = (torch.randn(n_users, k) * 0.1).requires_grad_()
    aP = torch.zeros(n_users, k, requires_grad=True)
    Q = (torch.randn(n_items, k) * 0.1).requires_grad_()
    Y = (torch.randn(n_items, k) * 0.1).requires_grad_()
    opt = torch.optim.Adam([bu, au, bi, bib, P, aP, Q, Y], lr=self.lr)

    u_t = torch.tensor(u, dtype=torch.long)
    i_t = torch.tensor(i, dtype=torch.long)
    bin_t = torch.tensor(bins, dtype=torch.long)
    dev_t = torch.tensor(dev, dtype=torch.float32)
    y_t = torch.tensor(r, dtype=torch.float32)
    n = len(r)

    for _ in range(self.n_epochs):
      perm = torch.randperm(n)
      for start in range(0, n, self.batch_size):
        b = perm[start:start + self.batch_size]
        ub, ib, bb, db = u_t[b], i_t[b], bin_t[b], dev_t[b]

        Z = torch.sparse.mm(A_t, Y)
        pu = P[ub] + aP[ub] * db[:, None] + Z[ub]
        pred = self.mu + bi[ib] + bib[ib, bb] + bu[ub] + au[ub] * db + (Q[ib] * pu).sum(dim=1)

        mse = ((pred - y_t[b]) ** 2).mean()
        pen = (bu[ub] ** 2 + au[ub] ** 2 + bi[ib] ** 2 + bib[ib, bb] ** 2
               + (P[ub] ** 2).sum(dim=1) + (aP[ub] ** 2).sum(dim=1) + (Q[ib] ** 2).sum(dim=1)).mean()
        pen = pen + (Y ** 2).sum() / n_items

        loss = mse + self.reg * pen
        opt.zero_grad()
        loss.backward()
        opt.step()

    self.bu = bu.detach().numpy()
    self.au = au.detach().numpy()
    self.bi = bi.detach().numpy()
    self.bib = bib.detach().numpy()
    self.P = P.detach().numpy()
    self.aP = aP.detach().numpy()
    self.Q = Q.detach().numpy()
    self.Z = A @ Y.detach().numpy()
    return self

  def set_time(self, t):
    t = t / (86400 * 365)
    d = t - self.t_user
    self.dev = np.sign(d) * np.abs(d) ** self.beta * self.use_time
    self.bin = np.searchsorted(self.edges, t)

  def predict(self, users):
    dev = self.dev[users][:, None]
    p = self.P[users] + self.aP[users] * dev + self.Z[users]
    bias_u = self.mu + self.bu[users] + self.au[users] * self.dev[users]
    bias_i = self.bi[None, :] + self.bib[:, self.bin[users]].T
    return bias_u[:, None] + bias_i + p @ self.Q.T