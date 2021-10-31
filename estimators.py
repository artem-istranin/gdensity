# ==============================
# Application of the asymptotic equivalence of the density estimation problem and Gaussian white noise model
# ==============================

import numpy as np
from numpy import sqrt as sqrt
import math
import copy
import inspect
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import wavelets


def get_unit_interval_uniform_partition(m):
    return [[i / m, (i + 1) / m] for i in range(m)]


def get_unit_interval_partition_midpoints(m):
    return list(np.around(np.mean(get_unit_interval_uniform_partition(m), axis=1), 5))


def transform_samples_to_multinomial(samples_list, m):
    partition = get_unit_interval_uniform_partition(m)
    multinomial_count = [0] * m
    for i, part in enumerate(partition):
        if i == len(partition) - 1:
            count = len(np.where((part[0] <= np.asarray(samples_list)) & (np.asarray(samples_list) <= part[1]))[0])
        else:
            count = len(np.where((part[0] <= np.asarray(samples_list)) & (np.asarray(samples_list) < part[1]))[0])
        multinomial_count[i] = count
    return multinomial_count


def transform_iid_samples_to_trajectory(samples_list, gamma=1e-05):
    """"
        Transform iid samples list to a continuous trajectory. Resulted trajectory is constructed as
        estimated linear interpolation of integrals of squared mean function over the uniform
        subdivisions of interval [0, 1] multiplied by the number of subdivisions.

        :param samples_list: list
            List of iid samples. Sample values must be in interval [0, 1].
        :param gamma: float
            Factor controlling the number of subdivisions. For n given iis samples the number of
            subdivisions is calculated as the greatest integer less than or equal to n^{1 / (2 + gamma)}.
            gamma must be greater than 0 and less or equal to 1.

        :returns
            trajectory: function
                Resulted continuous trajectory function taking a single argument: float coordinate in interval [0, 1].

            (path_values, path_coordinates): tuple (list, list)
                Interpolated trajectory values in corresponding coordinates.
    """

    if not (0 <= min(samples_list) and max(samples_list) <= 1):
        raise ValueError('Transformation is specified for samples in interval [0, 1].')
    if not (0 < gamma <= 1):
        raise ValueError('Parameter `gamma` must be in interval (0, 1].')

    n = len(samples_list)
    m = int(n ** (1 / (2 + gamma)))

    # 1. Initial samples transformation to the sample of multinomial model
    multinomial_sample = transform_samples_to_multinomial(samples_list, m)
    thetas_mle = [m / np.sum(multinomial_sample) for m in multinomial_sample]  # MLE for thetas

    # 2. Estimation of thetas'
    sqrt_thetas_mle = [sqrt(theta / m) for theta in thetas_mle]

    # 3. Construction of continuous trajectory
    midpoints = get_unit_interval_partition_midpoints(m)

    interior_trajectory = interp1d(midpoints, [sqrt_theta * m for sqrt_theta in sqrt_thetas_mle], kind='linear')

    def trajectory(t):
        if not (0 <= t <= 1):
            raise ValueError('Trajectory values are supported on interval [0, 1].')

        if t <= midpoints[0]:
            t = midpoints[0]
        if t >= midpoints[-1]:
            t = midpoints[-1]
        return interior_trajectory(t)

    path_coordinates = [0] + midpoints + [1]
    path_values = [trajectory(t) for t in path_coordinates]
    return trajectory, (path_values, path_coordinates)


class BaseEstimator(object):
    """Base class for all estimators"""

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for estimator"""
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # Introspect the constructor arguments to find the model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self']
        return sorted([p.name for p in parameters])

    def get_params(self):
        """"Get parameters of estimator"""
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
        return out

    def set_params(self, **params):
        """"Set parameters of estimator"""
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter {}. '
                                 'Check the list of available parameters '
                                 'with `get_params().keys()`.'.format(key))
            setattr(self, key, value)
            valid_params[key] = value
        return self


class TrajectoryMeanEstimator(BaseEstimator):
    """Base class for all estimators of functions observed under white noise on the interval [0, 1]"""

    def fit(self, trajectory_func):
        """"Fit estimator of function observed under white noise on the given trajectory"""
        raise NotImplementedError('Not yet implemented method.')

    def score_trajectory_mean(self, t):
        """"Compute the estimated trajectory mean function value under the fitted model"""
        raise NotImplementedError('Not yet implemented method.')


class PinskerEstimator(TrajectoryMeanEstimator):

    def __init__(self, epsilon, L=1, beta=1, reflected_extension=True):
        if not (0 < epsilon < 1):
            raise ValueError('Parameter `epsilon` must be positive the value in (0, 1).')
        self.epsilon = epsilon

        if L <= 0:
            raise ValueError('Parameter `L` must be positive.')
        self.L = L

        if beta <= 0:
            raise ValueError('Parameter `beta` must be positive.')
        self.beta = beta

        self.reflected_extension = reflected_extension
        self.trajectory_mean_estimation = None
        self.coefficients_nb = None

    @staticmethod
    def _get_trigonometric_basis_value(j, x):
        # for j = 1, 2, ...
        if j == 1:
            return 1
        elif j % 2 == 0:
            return sqrt(2) * np.cos(np.pi * j * x)
        else:
            return sqrt(2) * np.sin(np.pi * (j - 1) * x)

    @staticmethod
    def _get_a_value(j, beta):
        # for j = 1, 2, ...
        if j % 2 == 0:
            return j ** beta
        else:
            return (j - 1) ** beta

    def fit(self, trajectory_func, integration_subdivisions=1000):
        if not isinstance(integration_subdivisions, int) or integration_subdivisions < 1:
            raise ValueError('Parameter `integration_subdivisions` must be positive integer.')

        path_coordinates = list(np.linspace(0, 1, integration_subdivisions))
        path_values = [trajectory_func(t) for t in path_coordinates]

        if self.reflected_extension:
            path_values = path_values[:-1] + path_values[::-1]
            path_coordinates = path_coordinates + [p + 1 for p in path_coordinates[1:]]
            path_coordinates = [c / 2 for c in path_coordinates]

        # Computation of l^*_j (weights):
        q = (self.L ** 2) / (np.pi ** (2 * self.beta))
        k = ((self.beta / ((2 * self.beta + 1) * (self.beta + 1) * q)) ** (self.beta / (2 * self.beta + 1))) * \
            self.epsilon ** ((2 * self.beta) / (2 * self.beta + 1))
        dim = max(math.ceil((1 / k) ** (1 / self.beta)), math.ceil(((1 - k) / k) ** (1 / self.beta)))
        a_list = [self._get_a_value(j, self.beta) for j in range(1, dim + 1)]
        weights = [(1 - (k * a)) for a in a_list]

        # Computation of y_j (projector coefficients):
        projector_coefficients = []
        for j in range(1, dim + 1):
            _basis_values = [self._get_trigonometric_basis_value(j, c) for c in path_coordinates]
            _integrate_values = [_basis_values[i] * path_values[i] for i in range(len(path_values))]
            projector_coefficients.append(integrate.simps(_integrate_values, path_coordinates))
        self.coefficients_nb = len(projector_coefficients)

        def estimator(t):
            # Computation of phi_j(x) (value of j-th trigonometric basis function):
            basis_values = [self._get_trigonometric_basis_value(j, t) for j in range(1, dim + 1)]

            return sum([weights[i] * projector_coefficients[i] * basis_values[i] for i in range(dim)])

        if self.reflected_extension:

            def corrected_estimator(t):
                return estimator(t / 2)

            self.trajectory_mean_estimation = corrected_estimator
        else:
            self.trajectory_mean_estimation = estimator
        return self

    def score_trajectory_mean(self, t):
        if self.trajectory_mean_estimation is None:
            raise TypeError('This TrajectoryMeanEstimator instance is not fitted yet. '
                            'Run `fit()` with appropriate arguments before using this estimator.')
        if not (0 <= t <= 1):
            raise ValueError('Pinsker estimator is specified for values in interval [0, 1].')
        return self.trajectory_mean_estimation(t)


def get_folded_function(f):
    # Generate 2 periodic folded function, symmetric about 0 and 1, and equal to f on [0, 1].
    def folded_function(t):
        # print('ask for t={}'.format(t))
        k1 = int((-1 + t) / 2)
        if 0 <= t - 2 * k1 <= 1:
            # print('return value in t={}'.format(t - 2 * k1))
            return f(t - 2 * k1)
        else:
            k2 = int((1 + t) / 2)
            # print('return value in t={}'.format(2 * k2 - t))
            return f(2 * k2 - t)

    return folded_function


class WaveletEstimator(TrajectoryMeanEstimator):

    def __init__(self, db_order, resolution, edge_strategy='folding_extension', level=10):
        self.db_order = db_order
        self.resolution = resolution
        self.edge_strategy = edge_strategy
        self.level = level

        self.phi = None
        self.psi = None
        self.t_grid = None

        self.trajectory_mean_estimation = None
        self.coefficients_nb = None

    def set_phi_psi_t_grid(self):
        self.phi, self.psi, self.t_grid = wavelets.db_scaling_function_and_wavelet(
            db_order=self.db_order, level=self.level)

    def get_scaled_translated_phi_psi(self, k):
        # phi_{-resolution, k} = (2 ** (resolution / 2)) * phi((2 ** resolution * t) - k)
        # if supp(phi) = [0, 2 * db_order - 1],
        # then supp(phi_{-resolution, k}) = [k / (2 ** resolution), ((2 * db_order - 1 + k) / (2 ** resolution))])
        scaled_translated_t_grid = []
        scaled_translated_phi = []
        scaled_translated_psi = []

        for i, t in enumerate(self.t_grid):
            scaled_translated_t = (t + k) / (2 ** self.resolution)
            if self.edge_strategy == 'zero_extension' and not (0 <= scaled_translated_t <= 1):
                continue
            scaled_translated_t_grid.append(scaled_translated_t)
            scaled_translated_phi.append((2 ** (self.resolution / 2)) * self.phi[i])
            scaled_translated_psi.append((2 ** (self.resolution / 2)) * self.psi[i])
        return scaled_translated_phi, scaled_translated_psi, scaled_translated_t_grid

    def fit(self, trajectory_func):
        if self.edge_strategy == 'zero_extension':
            self.set_phi_psi_t_grid()

            basis_functions = []

            # Computation of projector coefficients:
            projector_coefficients = []
            for k in range(-2 * self.db_order + 2, ((2 ** self.resolution) - 1) + 1):
                scaled_translated_phi, _, scaled_translated_t_grid = self.get_scaled_translated_phi_psi(k)
                scaled_translated_phi_func = interp1d(scaled_translated_t_grid, scaled_translated_phi, kind='linear',
                                                      bounds_error=False, fill_value=0.0)
                basis_functions.append(scaled_translated_phi_func)

                path_values = [trajectory_func(t) for t in scaled_translated_t_grid]
                _integrate_values = [scaled_translated_phi[i] * path_values[i] for i in range(len(path_values))]
                projector_coefficients.append(integrate.simps(_integrate_values, scaled_translated_t_grid))

            def estimator(t):
                est_val = 0
                for i, basis_phi in enumerate(basis_functions):
                    est_val += projector_coefficients[i] * basis_phi(t)
                return est_val

            self.trajectory_mean_estimation = estimator

        elif self.edge_strategy == 'folding_extension':
            self.set_phi_psi_t_grid()

            folded_trajectory_func = get_folded_function(trajectory_func)

            basis_functions = []

            # Computation of projector coefficients:
            projector_coefficients = []
            for k in range(-2 * self.db_order + 2, ((2 ** self.resolution) - 1) + 1):
                scaled_translated_phi, _, scaled_translated_t_grid = self.get_scaled_translated_phi_psi(k)
                scaled_translated_phi_func = interp1d(scaled_translated_t_grid, scaled_translated_phi, kind='linear',
                                                      bounds_error=False, fill_value=0.0)

                basis_functions.append(scaled_translated_phi_func)

                path_values = [folded_trajectory_func(t) for t in scaled_translated_t_grid]
                _integrate_values = [scaled_translated_phi[i] * path_values[i] for i in range(len(path_values))]
                projector_coefficients.append(integrate.simps(_integrate_values, scaled_translated_t_grid))

            def estimator(t):
                est_val = 0
                for i, basis_phi in enumerate(basis_functions):
                    est_val += projector_coefficients[i] * basis_phi(t)
                return est_val

            self.trajectory_mean_estimation = estimator

        elif self.edge_strategy == 'boundaries_correction':
            initial_db_order = self.db_order
            if (self.db_order not in wavelets.VALID_BC_WAVELETS_ORDER and
                    self.db_order - 1 in wavelets.VALID_BC_WAVELETS_ORDER):
                # solving using scaling functions + wavelets
                self.db_order = self.db_order - 1
            elif self.db_order not in wavelets.VALID_BC_WAVELETS_ORDER:
                raise NotImplementedError('Boundaries correction for wavelets '
                                          'of order {} is not yet developed'.format(self.db_order))

            self.set_phi_psi_t_grid()

            # first we translate the support interval for phi to the interval [-db_order + 1, db_order]
            self.t_grid = [t - self.db_order + 1 for t in self.t_grid]

            bc_db_collection = wavelets.bc_db_scaling_functions_and_wavelets(db_order=self.db_order,
                                                                             level=self.level)

            left_interpolation_edge = 0
            right_interpolation_edge = 1

            basis_functions = []

            # Computation of projector coefficients:
            projector_coefficients = []
            for k in range(0, ((2 ** self.resolution) - 1) + 1):
                if 0 <= k <= self.db_order - 1:
                    left_bc_phi, left_bc_psi, left_t_grid = bc_db_collection['left'][k]

                    if k == 0:
                        left_interpolation_edge = left_t_grid[1] / (2 ** self.resolution)

                    scaled_left_bc_phi = []
                    scaled_left_bc_psi = []
                    scaled_left_t_grid = []

                    for i, t in enumerate(left_t_grid):
                        scaled_left_t_grid.append(t / (2 ** self.resolution))
                        scaled_left_bc_phi.append((2 ** (self.resolution / 2)) * left_bc_phi[i])
                        scaled_left_bc_psi.append((2 ** (self.resolution / 2)) * left_bc_psi[i])

                    scaled_left_bc_phi_func = interp1d(scaled_left_t_grid, scaled_left_bc_phi, kind='linear',
                                                       bounds_error=False, fill_value=0.0)
                    basis_functions.append(scaled_left_bc_phi_func)

                    path_values = [trajectory_func(t) for t in scaled_left_t_grid]
                    _integrate_values = [scaled_left_bc_phi[i] * path_values[i] for i in range(len(path_values))]
                    projector_coefficients.append(integrate.simps(_integrate_values, scaled_left_t_grid))

                    if initial_db_order == (self.db_order + 1):
                        scaled_left_bc_psi_func = interp1d(scaled_left_t_grid, scaled_left_bc_psi, kind='linear',
                                                           bounds_error=False, fill_value=0.0)
                        basis_functions.append(scaled_left_bc_psi_func)

                        _integrate_values = [scaled_left_bc_psi[i] * path_values[i] for i in range(len(path_values))]
                        projector_coefficients.append(integrate.simps(_integrate_values, scaled_left_t_grid))

                elif self.db_order <= k <= (2 ** self.resolution) - self.db_order - 1:
                    scaled_translated_phi, scaled_translated_psi, scaled_translated_t_grid = \
                        self.get_scaled_translated_phi_psi(k)
                    scaled_translated_phi_func = interp1d(scaled_translated_t_grid, scaled_translated_phi,
                                                          kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                    basis_functions.append(scaled_translated_phi_func)

                    path_values = [trajectory_func(t) for t in scaled_translated_t_grid]
                    _integrate_values = [scaled_translated_phi[i] * path_values[i] for i in range(len(path_values))]
                    projector_coefficients.append(integrate.simps(_integrate_values, scaled_translated_t_grid))

                    if initial_db_order == (self.db_order + 1):
                        scaled_translated_psi_func = interp1d(scaled_translated_t_grid,
                                                              scaled_translated_psi, kind='linear',
                                                              bounds_error=False, fill_value=0.0)
                        basis_functions.append(scaled_translated_psi_func)

                        _integrate_values = [scaled_translated_psi[i] * path_values[i] for i in range(len(path_values))]
                        projector_coefficients.append(integrate.simps(_integrate_values, scaled_translated_t_grid))

                elif (2 ** self.resolution) - self.db_order <= k <= (2 ** self.resolution) - 1:
                    right_bc_phi, right_bc_psi, right_t_grid = bc_db_collection['right'][k - (2 ** self.resolution)]

                    if k == (2 ** self.resolution) - 1:
                        right_interpolation_edge = (right_t_grid[-2] / (2 ** self.resolution)) + 1

                    scaled_right_bc_phi = []
                    scaled_right_bc_psi = []
                    scaled_right_t_grid = []

                    for i, t in enumerate(right_t_grid):
                        scaled_right_t_grid.append((t / (2 ** self.resolution)) + 1)
                        scaled_right_bc_phi.append((2 ** (self.resolution / 2)) * right_bc_phi[i])
                        scaled_right_bc_psi.append((2 ** (self.resolution / 2)) * right_bc_psi[i])

                    scaled_right_bc_phi_func = interp1d(scaled_right_t_grid, scaled_right_bc_phi, kind='linear',
                                                        bounds_error=False, fill_value=0.0)
                    basis_functions.append(scaled_right_bc_phi_func)

                    path_values = [trajectory_func(t) for t in scaled_right_t_grid]
                    _integrate_values = [scaled_right_bc_phi[i] * path_values[i] for i in range(len(path_values))]
                    projector_coefficients.append(integrate.simps(_integrate_values, scaled_right_t_grid))

                    if initial_db_order == (self.db_order + 1):
                        scaled_right_bc_psi_func = interp1d(scaled_right_t_grid, scaled_right_bc_psi, kind='linear',
                                                            bounds_error=False, fill_value=0.0)
                        basis_functions.append(scaled_right_bc_psi_func)

                        _integrate_values = [scaled_right_bc_psi[i] * path_values[i] for i in range(len(path_values))]
                        projector_coefficients.append(integrate.simps(_integrate_values, scaled_right_t_grid))

            def estimator(t):
                if t < left_interpolation_edge:
                    t = left_interpolation_edge
                if t > right_interpolation_edge:
                    t = right_interpolation_edge

                est_val = 0
                for i, basis_phi in enumerate(basis_functions):
                    est_val += projector_coefficients[i] * basis_phi(t)
                return est_val

            self.trajectory_mean_estimation = estimator

        else:
            raise NotImplementedError('{} edge strategy is not developed'.format(self.edge_strategy))

        self.coefficients_nb = len(projector_coefficients)

        return self

    def score_trajectory_mean(self, t):
        if self.trajectory_mean_estimation is None:
            raise TypeError('This TrajectoryMeanEstimator instance is not fitted yet.'
                            'Run `fit()` with appropriate arguments before using this estimator.')
        if not (0 <= t <= 1):
            raise ValueError('Wavelet estimator is specified for values in interval [0, 1].')
        return self.trajectory_mean_estimation(t)


def get_scaled_compact_pdf_func(pdf_func, pdf_supp_min, pdf_supp_max):
    """Scale and translate the domain of pdf_func from [pdf_supp_min, pdf_supp_max] to the interval [0, 1]"""
    scaling_factor = abs(pdf_supp_max - pdf_supp_min)

    def scaled_pdf(x):
        if not(0 <= x <= 1):
            raise ValueError('Scaled function is specified for values in interval [0, 1].')
        return scaling_factor * pdf_func(x * scaling_factor + pdf_supp_min)

    return scaled_pdf, (pdf_supp_min, pdf_supp_max)


def get_inverse_scaled_compact_pdf_func(scaled_pdf, pdf_supp_min, pdf_supp_max):
    """Scale and translate the domain of pdf_func from [0, 1] to the interval [pdf_supp_min, pdf_supp_max]"""
    scaling_factor = abs(pdf_supp_max - pdf_supp_min)

    def pdf_func(x):
        return scaled_pdf(abs(x - pdf_supp_min) / scaling_factor) / scaling_factor

    return pdf_func, (pdf_supp_min, pdf_supp_max)


VALID_ESTIMATION_ALGORITHMS = [
    'pinsker',
    'wavelet',
]


class GaussianTrajectoryDensity(BaseEstimator):
    """"
        Class for nonparametric density estimation using the asymptotic equivalence with the estimation
        of the mean function in the corresponding gaussian white noise trajectory.

        Note: all "fit" methods scale and translate the samples to the interval [0, 1], generate
        the corresponding gaussian white noise trajectory, estimate the underlying mean function and
        scale it back to obtain the density estimation in the original domain.
    """

    def __init__(self, estimation_algorithm='pinsker', gamma=1e-05, pdf_supp_min=None, pdf_supp_max=None):
        self.gamma = gamma
        self.pdf_supp_min = pdf_supp_min
        self.pdf_supp_max = pdf_supp_max
        if self.pdf_supp_min is not None and self.pdf_supp_max is not None:
            if not (pdf_supp_min < pdf_supp_max):
                raise ValueError('Parameter `pdf_supp_min` must be less then `pdf_supp_max`.')
            self.scaling_factor = abs(self.pdf_supp_max - self.pdf_supp_min)
        else:
            self.scaling_factor = None

        self._samples_list = None
        self._scaled_samples_list = None
        self._samples_nb = None
        self.trajectory = None

        if estimation_algorithm not in VALID_ESTIMATION_ALGORITHMS:
            raise ValueError('Invalid estimation algorithm: {}.'.format(estimation_algorithm))
        self.estimation_algorithm = estimation_algorithm
        self._scaled_estimator = None
        self._estimator = None
        self._normalization_factor = None
        self.coefficients_nb = None

    def _get_scaled_samples_list(self):
        scaled_samples_list = []
        for sample in copy.deepcopy(self._samples_list):
            scaled_samples_list.append(abs(sample - self.pdf_supp_min) / self.scaling_factor)
        return scaled_samples_list

    def _initialize_samples_list(self, samples_list):
        self._samples_list = samples_list

        if self.pdf_supp_min is None:
            self.pdf_supp_min = min(self._samples_list)
        else:
            if min(self._samples_list) < self.pdf_supp_min:
                raise ValueError('Minimum value of samples can not be smaller than {}. '
                                 'To change it use `set_params(pdf_supp_min=<value>)`.'.format(self.pdf_supp_min))

        if self.pdf_supp_max is None:
            self.pdf_supp_max = max(self._samples_list)
        else:
            if max(self._samples_list) > self.pdf_supp_max:
                raise ValueError('Maximum value of samples can not be greater than {}. '
                                 'To change it use `set_params(pdf_supp_max=<value>)`.'.format(self.pdf_supp_max))

        self.scaling_factor = abs(self.pdf_supp_max - self.pdf_supp_min)
        self._scaled_samples_list = self._get_scaled_samples_list()
        self._samples_nb = len(self._samples_list)
        self.trajectory, _ = transform_iid_samples_to_trajectory(self._scaled_samples_list, gamma=self.gamma)

    def fit(self, samples_list, **params):
        self._initialize_samples_list(samples_list)

        if self.estimation_algorithm == 'pinsker':
            if 'epsilon' in params:
                raise ValueError('Parameter `epsilon` is internally specified for {}.'.format(self))
            epsilon = 1 / (2 * sqrt(self._samples_nb))
            if 'L' not in params:
                params['L'] = self.scaling_factor
            pinsker = PinskerEstimator(epsilon=epsilon, **params)
            self._scaled_estimator = pinsker.fit(trajectory_func=self.trajectory).score_trajectory_mean
            self._estimator, _ = get_inverse_scaled_compact_pdf_func(self._scaled_estimator,
                                                                     self.pdf_supp_min,
                                                                     self.pdf_supp_max)
            self.coefficients_nb = pinsker.coefficients_nb
        elif self.estimation_algorithm == 'wavelet':
            w_estimator = WaveletEstimator(**params)
            self._scaled_estimator = w_estimator.fit(trajectory_func=self.trajectory).score_trajectory_mean
            self._estimator, _ = get_inverse_scaled_compact_pdf_func(self._scaled_estimator,
                                                                     self.pdf_supp_min,
                                                                     self.pdf_supp_max)
            self.coefficients_nb = w_estimator.coefficients_nb
        else:
            raise NotImplementedError('{} estimation algorithm is not developed'.format(self.estimation_algorithm))

        # scaling the resulted estimation to be the probability density function
        integration_subdivisions = 10000
        x_grid = np.linspace(self.pdf_supp_min, self.pdf_supp_max, integration_subdivisions)
        _integrate_values = [self._estimator(x) ** 2 for x in x_grid]
        res_estimator_area = integrate.simps(_integrate_values, x_grid)
        self._normalization_factor = 1 / res_estimator_area

        return self

    def score_density_value(self, x):
        """"The value of probability density evaluation"""
        return self._normalization_factor * self._estimator(x) ** 2


if __name__ == '__main__':
    # import mt_examples as mt
    # import matplotlib.pyplot as plt
    # import time

    # --------------------------------------
    # pinsker = PinskerEstimator()
    # sigma = 0.1
    # points_nb = 1000
    # epsilon = sigma / sqrt(points_nb)
    # g = mt.Generators()
    # path_values, path_coordinates = g.generate_gaussian_white_noise_path(std=sigma, points_nb=points_nb)
    # estimator = pinsker.pinsker_estimator(path_values=path_values, path_coordinates=path_coordinates, epsilon=epsilon)
    # plt.scatter(path_coordinates, path_values)
    # plt.plot(path_coordinates, [estimator(x) for x in path_coordinates], color='red')
    # plt.show()
    # print([estimator(x) for x in np.linspace(0, 1, 10)])
    # ref_pinsker_estimator = pinsker.ref_pinsker_estimator(path_values, path_coordinates, epsilon, L=1)
    # print([ref_pinsker_estimator(x) for x in np.linspace(0, 1, 10)])
    # plt.scatter(path_coordinates, path_values)
    # plt.plot(path_coordinates, [ref_pinsker_estimator(x) for x in path_coordinates], color='red')
    # plt.show()

    # --------------------------------------
    # g = mt.Generators()
    # a = 1
    # b = 10
    # pdf_func = g.get_reciprocal_pdf_function(a, b)
    # scaled_pdf_func, (pdf_supp_min, pdf_supp_max) = get_scaled_compact_pdf_func(
    #     pdf_func, pdf_supp_min=a, pdf_supp_max=b)
    # print('Pdf(a)={}, ScaledPdf(0)={} (must NOT be equal!)'.format(pdf_func(a), scaled_pdf_func(0)))
    # plt.plot(np.linspace(a, b, 100), [pdf_func(t) for t in np.linspace(a, b, 100)], color='blue')
    # plt.plot(np.linspace(0, 1, 100), [scaled_pdf_func(t) for t in np.linspace(0, 1, 100)], color='red')
    # print('Test0: area of pdf=', round(integrate.quad(pdf_func, a=a, b=b)[0], 5))
    # print('Test1: area of scaled pdf=', round(integrate.quad(scaled_pdf_func, a=0, b=1)[0], 5))
    # print('Scaling inverse')
    # pdf_func1, _ = get_inverse_scaled_compact_pdf_func(scaled_pdf_func, pdf_supp_min, pdf_supp_max)
    # print('Test 1: pdf_func == pdf_func in several points {}'.format(
    #     all([pdf_func1(x) == pdf_func(x) for x in np.linspace(a, b, 100)])))
    # print([pdf_func1(x) == pdf_func(x) for x in np.linspace(a, b, 100)])
    # print('pdf_func: ', [pdf_func(x) for x in np.linspace(a, b, 100)])
    # print('pdf_func1: ', [pdf_func1(x) for x in np.linspace(a, b, 100)])
    # plt.plot(np.linspace(a, b, 100), [pdf_func1(t) for t in np.linspace(a, b, 100)], color='green')
    # plt.show()

    # --------------------------------------
    # import mt_examples as mt
    # import matplotlib.pyplot as plt
    # import time
    # g = mt.Generators()
    # reciprocal_samples = g.generate_reciprocal_simulation(a=0.1, b=1.1, samples_nb=1000, random_seed=930824)
    # reciprocal_samples = [s - 0.1 for s in reciprocal_samples]
    # trajectory, (path_values, path_coordinates) = transform_iid_samples_to_trajectory(reciprocal_samples, gamma=1e-05)
    # plt.plot(path_coordinates, path_values, lw=5, color='blue')
    # print('len(path_coordinates): ', len(path_coordinates))
    # print('len(path_values): ', len(path_values))
    #
    # pinsker = PinskerEstimator()
    # n = len(reciprocal_samples)
    # epsilon = 1 / (2 * sqrt(n))
    # ref_pinsker_estimator = pinsker.ref_pinsker_estimator(path_values=path_values,
    #                                                       path_coordinates=path_coordinates,
    #                                                       epsilon=epsilon, L=1)
    # t_grid = np.linspace(0, 1, 100)
    # plt.plot(t_grid, [ref_pinsker_estimator(t) for t in t_grid])
    # plt.show()

    # --------------------------------------
    # a = 0.1
    # b = 1.5
    # gt_density = GaussianTrajectoryDensity(estimation_algorithm='pinsker', gamma=1, pdf_supp_min=a, pdf_supp_max=b)
    #
    # g = mt.Generators()
    # reciprocal_pdf_func = g.get_reciprocal_pdf_function(a, b)
    # reciprocal_samples = g.generate_reciprocal_simulation(a=a, b=b, samples_nb=100000, random_seed=930824)
    # gt_density.fit(reciprocal_samples)
    #
    # t_grid = np.linspace(a, b, 100)
    # plt.plot(t_grid, [gt_density.score_density_value(t) for t in t_grid])
    # plt.plot(t_grid, [reciprocal_pdf_func(t) for t in t_grid])
    #
    # plt.show()

    # -----------------------------------
    # import matplotlib.pyplot as plt
    #
    # db_order = 3
    # resolution = 3
    # w_estimator = WaveletEstimator(db_order, resolution, edge_strategy='zero_extension')
    #
    # g = mt.Generators()
    # mean_func = g.sample_periodic_func2
    #
    # sigma = 0.1
    # points_nb = 1000
    # path_values, path_coordinates = g.generate_gaussian_white_noise_path(
    #         std=sigma, mean_func=mean_func, points_nb=points_nb, random_seed=919191)
    # trajectory = interp1d(path_coordinates, path_values, kind='linear')
    #
    # t_grid = np.linspace(0, 1, 1000)
    # plt.plot(t_grid, [mean_func(t) for t in t_grid], color='red')
    # plt.scatter(path_coordinates, path_values, alpha=0.5)
    #
    # w_estimator.fit(trajectory)
    # plt.plot(t_grid, [w_estimator.score_trajectory_mean(t) for t in t_grid], color='blue')
    #
    # plt.show()

    # -----------------------------------
    # import matplotlib.pyplot as plt
    # g = mt.Generators()
    # mean_func = g.sample_nonperiodic_func2
    # folded_mean_func = get_folded_function(mean_func)
    # folded_mean_func(0.5)
    #
    # t_grid = np.linspace(-1, 2, 1000)
    # plt.plot(t_grid, [folded_mean_func(t) for t in t_grid], color='gray')
    #
    # plt.plot(np.linspace(0, 1, 1000), [mean_func(t) for t in np.linspace(0, 1, 1000)], color='red')
    #
    # plt.show()

    # -----------------------------------
    # import matplotlib.pyplot as plt
    #
    # db_order = 3
    # resolution = 3
    # w_estimator = WaveletEstimator(db_order, resolution, edge_strategy='folding_extension')
    #
    # g = mt.Generators()
    # mean_func = g.sample_periodic_func2
    #
    # sigma = 0.1
    # points_nb = 1000
    # path_values, path_coordinates = g.generate_gaussian_white_noise_path(
    #         std=sigma, mean_func=mean_func, points_nb=points_nb, random_seed=919191)
    # trajectory = interp1d(path_coordinates, path_values, kind='linear')
    #
    # t_grid1 = np.linspace(-1, 2, 1000)
    # folded_trajectory = get_folded_function(trajectory)
    #
    # t_grid = np.linspace(0, 1, 1000)
    # plt.plot(t_grid, [mean_func(t) for t in t_grid], color='red')
    # plt.scatter(path_coordinates, path_values, alpha=0.5)
    #
    # w_estimator.fit(trajectory)
    # plt.plot(t_grid, [w_estimator.score_trajectory_mean(t) for t in t_grid], color='blue')
    # plt.show()

    # -----------------------------------
    # import matplotlib.pyplot as plt
    #
    # db_order = 3
    # resolution = 3
    # w_estimator = WaveletEstimator(db_order, resolution, edge_strategy='boundaries_correction')
    #
    # g = mt.Generators()
    # mean_func = g.sample_periodic_func2
    #
    # sigma = 0.1
    # points_nb = 1000
    # path_values, path_coordinates = g.generate_gaussian_white_noise_path(
    #         std=sigma, mean_func=mean_func, points_nb=points_nb, random_seed=919191)
    # trajectory = interp1d(path_coordinates, path_values, kind='linear')
    # w_estimator.fit(trajectory)
    #
    # t_grid = np.linspace(0, 1, 100)
    #
    # plt.plot(t_grid, [mean_func(t) for t in t_grid], color='red')
    # plt.scatter(path_coordinates, path_values, alpha=0.5)
    # plt.plot(t_grid, [w_estimator.score_trajectory_mean(t) for t in t_grid], color='blue')
    # plt.show()

    # -----------------------------------
    # import matplotlib.pyplot as plt
    #
    # db_order = 3
    # resolution = 3
    # w_estimator = WaveletEstimator(db_order, resolution, edge_strategy='boundaries_correction')
    #
    # g = mt.Generators()
    # mean_func = g.sample_nonperiodic_func2
    #
    # sigma = 0.1
    # points_nb = 1000
    # path_values, path_coordinates = g.generate_gaussian_white_noise_path(
    #         std=sigma, mean_func=mean_func, points_nb=points_nb, random_seed=919191)
    # trajectory = interp1d(path_coordinates, path_values, kind='linear')
    # w_estimator.fit(trajectory)
    #
    # t_grid = np.linspace(0, 1, 100)
    #
    # plt.plot(t_grid, [mean_func(t) for t in t_grid], color='red')
    # plt.scatter(path_coordinates, path_values, alpha=0.5)
    # plt.plot(t_grid, [w_estimator.score_trajectory_mean(t) for t in t_grid], color='blue')
    # plt.show()

    pass
