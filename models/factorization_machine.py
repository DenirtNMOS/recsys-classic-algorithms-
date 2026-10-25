import numpy as np
import scipy.sparse as sp
import torch

class FM():
  def __init__(self, n_factors=16, reg=0.01, lr=0.01, n_epochs=10, batch_size=4096):
    self.n_factors = n_factors
    self.reg = reg
    self.lr = lr
    self.n_epochs = n_epochs
    self.batch_size = batch_size

  def to_torch(self, X):
    X = X.tocoo()
    idx = torch.tensor(np.vstack([X.row, X.col]), dtype=torch.long)
    val = torch.tensor(X.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, X.shape, check_invariants=False)

  def forward(self, Xb):
    lin = torch.sparse.mm(Xb, self.w).squeeze(1)
    s = torch.sparse.mm(Xb, self.V)
    q = torch.sparse.mm(Xb, self.V ** 2)
    pair = 0.5 * (s ** 2 - q).sum(dim=1)
    return self.w0 + lin + pair

  def penalty(self, Xb):
    norm = self.w ** 2 + (self.V ** 2).sum(dim=1, keepdim=True)
    return torch.sparse.mm(Xb, norm).mean()

  def fit(self, X, y, n_users, Fi):
    n, n_feat = X.shape
    self.w0 = torch.tensor(float(y.mean()), requires_grad=True)
    self.w = torch.zeros(n_feat, 1, requires_grad=True)
    self.V = (torch.randn(n_feat, self.n_factors) * 0.01).requires_grad_()
    opt = torch.optim.Adam([self.w0, self.w, self.V], lr=self.lr)
    y = torch.tensor(y, dtype=torch.float32)

    for _ in range(self.n_epochs):
      perm = np.random.permutation(n)
      for start in range(0, n, self.batch_size):
        b = perm[start:start + self.batch_size]
        Xb = self.to_torch(X[b])
        loss = ((self.forward(Xb) - y[b]) ** 2).mean() + self.reg * self.penalty(Xb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    w = self.w.detach().numpy().ravel()
    V = self.V.detach().numpy()
    self.w0_ = self.w0.item()
    self.w_u, self.V_u = w[:n_users], V[:n_users]
    w_i, V_i = w[n_users:], V[n_users:]
    self.S = Fi @ V_i
    Q = Fi @ (V_i ** 2)
    self.item_term = Fi @ w_i + 0.5 * (self.S ** 2 - Q).sum(axis=1)
    return self

  def predict(self, users):
    return self.w0_ + self.w_u[users][:, None] + self.item_term[None, :] + self.V_u[users] @ self.S.T