"""
Step 0.1.1 -- Split conformal, by hand

Goal: see distribution-free marginal coverage appear empirically.

We train one small classifier on a toy multi-class problem (synthetic Gaussian
blobs -- swap in the CIFAR-10 `Net` from the classifier tutorial if you want a
heavier example; the conformal procedure is identical either way). Then, for
many random calibration/test splits, we:

  1. Compute nonconformity scores s_i = 1 - f_{y_i}(x_i) on a calibration set
     (held out, not used for training).
  2. Set q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score.
  3. Form prediction sets on test points: {y : 1 - f_y(x) <= q_hat}.
  4. Measure the fraction of test points whose true label landed in the set.

Assumption used: exchangeability of (calibration, test) points -- nothing else.
No assumption on the model, the data distribution, or correctness of f.

Pitfall avoided: we use the ceil((n+1)(1-alpha))-th order statistic (not the
plain n-quantile). The "+1" is what gives the finite-sample guarantee;
dropping it undercovers, especially for small n.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. A toy classifier
# ---------------------------------------------------------------------------
# Synthetic Gaussian blobs, K classes in D dimensions. Trains in a couple
# seconds, which matters since we re-run the cal/test split many times.
# (If you'd rather use the CIFAR-10 `Net` from the classifier tutorial, train
# it once, set `model = net`, `X_pool, y_pool = <test set tensors>`, and skip
# straight to Section 2 -- the conformal code only assumes model(x) returns
# logits.)

K = 5            # number of classes
D = 8            # feature dimension
N_TRAIN = 2000
N_POOL = 4000    # pool we'll repeatedly re-split into calibration/test


def make_gaussian_blobs(n, K, D, class_sep=2.5):
    centers = np.random.randn(K, D) * class_sep
    y = np.random.randint(0, K, size=n)
    X = centers[y] + np.random.randn(n, D)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), centers


X_train, y_train, centers = make_gaussian_blobs(N_TRAIN, K, D)

# pool drawn from the SAME centers so train/pool are exchangeable
y_pool = torch.randint(0, K, (N_POOL,))
X_pool = torch.tensor(centers[y_pool.numpy()], dtype=torch.float32) + torch.randn(N_POOL, D)


class Net(nn.Module):
    def __init__(self, d_in, k_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, k_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = Net(D, K)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

print(f"final train loss: {loss.item():.4f}")
with torch.no_grad():
    train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
print(f"train accuracy: {train_acc:.3f}")

# ---------------------------------------------------------------------------
# 2. Split conformal, by hand
# ---------------------------------------------------------------------------
# Note this classifier is deliberately imperfect (overlapping Gaussian
# blobs) -- conformal coverage should hold regardless of how good or bad it
# is, which is the whole point: the guarantee comes from exchangeability of
# the calibration/test split, not from model accuracy.

ALPHA = 0.10  # target miscoverage -> we want ~90% coverage


@torch.no_grad()
def softmax_probs(model, X):
    return F.softmax(model(X), dim=1)


def conformal_quantile(scores, alpha):
    """q_hat = ceil((n+1)(1-alpha))-th smallest score, clipped to <= 1."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)  # if k > n, q_hat is effectively infinite (use max score)
    return np.sort(scores)[k - 1]


def one_split_coverage(model, X_pool, y_pool, n_cal, alpha):
    n_pool = X_pool.shape[0]
    perm = torch.randperm(n_pool)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

    X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]
    X_test, y_test = X_pool[test_idx], y_pool[test_idx]

    probs_cal = softmax_probs(model, X_cal)
    s_cal = (1 - probs_cal[torch.arange(len(y_cal)), y_cal]).numpy()
    q_hat = conformal_quantile(s_cal, alpha)

    probs_test = softmax_probs(model, X_test)
    nonconf_test = 1 - probs_test  # shape (n_test, K): nonconformity per candidate label
    pred_sets = nonconf_test <= q_hat  # bool mask, True where label y is included

    covered = pred_sets[torch.arange(len(y_test)), y_test]
    avg_set_size = pred_sets.float().sum(dim=1).mean().item()
    return covered.float().mean().item(), avg_set_size, q_hat


N_SPLITS = 500
N_CAL = 300

coverages = []
set_sizes = []
for _ in range(N_SPLITS):
    cov, size, _ = one_split_coverage(model, X_pool, y_pool, N_CAL, ALPHA)
    coverages.append(cov)
    set_sizes.append(size)

coverages = np.array(coverages)
print(f"target coverage: {1 - ALPHA:.3f}")
print(f"mean realized coverage over {N_SPLITS} splits: {coverages.mean():.4f}")
print(f"std of realized coverage: {coverages.std():.4f}")
print(f"mean prediction-set size: {np.mean(set_sizes):.2f} (of {K} classes)")

plt.figure(figsize=(6, 4))
plt.hist(coverages, bins=30, color="steelblue", edgecolor="black")
plt.axvline(1 - ALPHA, color="red", linestyle="--", label=f"target = 1-alpha = {1-ALPHA:.2f}")
plt.axvline(coverages.mean(), color="black", linestyle="-", label=f"mean = {coverages.mean():.3f}")
plt.xlabel("realized coverage on test split")
plt.ylabel("count")
plt.title(f"Split conformal coverage across {N_SPLITS} random cal/test splits\n(n_cal={N_CAL}, alpha={ALPHA})")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 3. The pitfall, demonstrated
# ---------------------------------------------------------------------------
# Use the plain n-th smallest score (no "+1") instead of
# ceil((n+1)(1-alpha)), and watch mean coverage fall measurably below
# 1-alpha -- most noticeably at small n_cal.


def buggy_quantile(scores, alpha):
    """WRONG: uses ceil(n * (1 - alpha)) instead of ceil((n+1) * (1 - alpha)). Undercovers."""
    n = len(scores)
    k = int(np.ceil(n * (1 - alpha)))
    k = min(max(k, 1), n)
    return np.sort(scores)[k - 1]


def one_split_coverage_buggy(model, X_pool, y_pool, n_cal, alpha):
    n_pool = X_pool.shape[0]
    perm = torch.randperm(n_pool)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]
    X_test, y_test = X_pool[test_idx], y_pool[test_idx]

    probs_cal = softmax_probs(model, X_cal)
    s_cal = (1 - probs_cal[torch.arange(len(y_cal)), y_cal]).numpy()
    q_hat = buggy_quantile(s_cal, alpha)

    probs_test = softmax_probs(model, X_test)
    pred_sets = (1 - probs_test) <= q_hat
    covered = pred_sets[torch.arange(len(y_test)), y_test]
    return covered.float().mean().item()


N_CAL_SMALL = 30  # small n makes the +1 effect obvious
buggy_cov = [one_split_coverage_buggy(model, X_pool, y_pool, N_CAL_SMALL, ALPHA) for _ in range(N_SPLITS)]
correct_cov = [one_split_coverage(model, X_pool, y_pool, N_CAL_SMALL, ALPHA)[0] for _ in range(N_SPLITS)]

print(f"n_cal = {N_CAL_SMALL}, target = {1 - ALPHA:.3f}")
print(f"correct (+1) mean coverage:   {np.mean(correct_cov):.4f}")
print(f"buggy (no +1) mean coverage:  {np.mean(buggy_cov):.4f}")

# ---------------------------------------------------------------------------
# Done-when check
# ---------------------------------------------------------------------------
# - Mean realized coverage above (Section 2) should sit close to 1-alpha --
#   run the script a few times / increase N_SPLITS if it looks off.
# - The only assumption used anywhere in one_split_coverage is that
#   calibration and test points are exchangeable draws from the same pool --
#   nothing about the model's accuracy, the data's distribution shape, or
#   the number of classes was assumed.
# - Quantile to recall from memory: q_hat = the ceil((n+1)(1-alpha))-th
#   smallest calibration nonconformity score (Section 3 shows what breaks
#   without the "+1").
