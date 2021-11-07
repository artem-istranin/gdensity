# Asymptotic equivalence of the density estimation problem and Gaussian white noise model

This project implements nonparametric density estimation using the asymptotic equivalence of a nonparametric density estimation problem to a Gaussian white noise model. The construction can be described in two steps:
- construction of observations in sample space of continuous trajectories of stochastic process defined on supp(f) from given samples in the initial space that corresponds to the given estimation problem,
- estimate the corresponding mean of a trajectory of a Gaussian with noise model.

Available trajectory mean estimators:
- **Pinsker estimator** (where for the nonperiodic functions the folding strategy is implemented)
- **Wavelet estimator** (where zero and folding extensions and boundary corrected wavelets are implemented to estimate compact trajectory)

![alt text](https://github.com/artem-istranin/gdensity/blob/master/kernel_pinsker_wavelet_example.png)

TODO: code examples

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
