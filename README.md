# Asymptotic equivalence of the density estimation problem and Gaussian white noise model

This project implements nonparametric density estimation using the asymptotic equivalence of a nonparametric density estimation problem to a Gaussian white noise model. The construction can be described in two steps:
- construction of observations in sample space of continuous trajectories of stochastic process defined on supp(f) from given samples in the initial space that corresponds to the given estimation problem,
- estimate the corresponding mean of a trajectory of a Gaussian with noise model.

Available trajectory mean estimators:
- **Pinsker estimator** (where for the nonperiodic functions the folding strategy is implemented)
- **Wavelet estimator** (where zero and folding extensions and boundary corrected wavelets are implemented to estimate compact trajectory)

![alt text](https://github.com/artem-istranin/gdensity/blob/master/kernel_pinsker_wavelet_example.png)

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from gdensity.estimators import GaussianTrajectoryDensity

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
```

## References
- [1] D. Blackwell. “Comparison of Experiments”. In: 1951.
- [2] C. Blatter. “Wavelets: A Primer”. In: 1999.
- [3] P. Butzer and R. Nessel. “Fourier analysis and approximation”. In: 1971.
- [4] L. L. Cam. “Asymptotic methods in statistical decision theory”. In: 1986.
- [5] A. Carter. “Deficiency distance between multinomial and multivariate normal experiments”. In: Annals of Statistics 30 (2002), pp. 708–730.
- [6] N. Chentsov. “A bound for an unknown distribution density in terms of the observations”. In: Dokl. Akad. Nauk SSSR 147 (1962), pp. 45–48.
- [7] A. Cohen, I. Daubechies, and P. Vial. “Wavelets on the Interval and Fast Wavelet Transforms”. In: Applied and Computational Harmonic Analysis 1 (1993), pp. 54–81.
- [8] I. Daubechies and C. Heil. “Ten lectures on wavelets”. In: Computers in Physics 6 (1992), pp. 697–697.
- [9] E. Giné and R. Nickl. “Mathematical Foundations of Infinite-Dimensional Statistical Models”. In: 2015.
- [10] W. Härdle et al. “Wavelets, Approximation, and Statistical Applications”. In: 1998.
- [11] G. Kaiser. “A friendly guide to wavelets / Gerald Kaiser”. In: 2011.
- [12] W. Lawton. “Necessary and sufficient conditions for constructing orthonormal wavelet bases”. In: 1991.
- [13] S. Mallat. “Multiresolution approximations and wavelet orthonormal bases of L2(R)”. In: 1989.
- [14] S. Mallat. “A Wavelet Tour of Signal Processing - The Sparse Way, 3rd Edition”. In: 2008.
- [15] E. Mariucci. “Le cam theory on the comparison of statistical models”. In: Graduate J. Math. 1 (2016), pp. 81–91.
- [16] M. Mehra. “Wavelets Theory and Its Applications: A First Course”. In: 2018.
- [17] E. Parzen. “On estimation of a probability density function and more”. In: Annals of Mathematical Statistics 33 (1962), pp. 1065–1076.
- [18] M. Rosenblatt. “Remarks on Some Nonparametric Estimates of a Density Function”. In: Annals of Mathematical Statistics 27 (1956), pp. 832–837.
[19] A. Tsybakov. “Introduction to Nonparametric Estimation”. In: Springer series in statistics. 2009.
