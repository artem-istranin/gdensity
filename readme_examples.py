import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from estimators import GaussianTrajectoryDensity

# Plot a density estimation example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
supp = [-5, 10]
X_plot = np.linspace(supp[0], supp[1], 1000)[:, np.newaxis]

fig, ax = plt.subplots()
# true density:
true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0])
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="Input distribution")
# show simulated points:
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

# kernel density estimator:
kde = KernelDensity(kernel="epanechnikov", bandwidth=0.5).fit(X)
kde_dens = np.exp(kde.score_samples(X_plot))
ax.plot(X_plot[:, 0], kde_dens, label="Kernel estimator\n(Epanechnikov kernel)")

# pinsker trajectory estimator:
pte = GaussianTrajectoryDensity(estimation_algorithm='pinsker', pdf_supp_min=supp[0], pdf_supp_max=supp[1]).fit(X)
pte_dens = [pte.score_density_value(t) for t in X_plot]
ax.plot(X_plot[:, 0], pte_dens, label="Pinsker estimator")

# wavelet trajectory estimator:
wte = GaussianTrajectoryDensity(estimation_algorithm='wavelet', pdf_supp_min=supp[0], pdf_supp_max=supp[1]).fit(
    X, db_order=4, resolution=3)
wte_dens = [wte.score_density_value(t) for t in X_plot]
ax.plot(X_plot[:, 0], wte_dens, label="Wavelet estimator\n(order 4, resolution 3)")

ax.text(6, 0.38, f"{N=} points")
ax.legend(loc="upper left")
ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()
