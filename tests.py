from estimators import *

from scipy.stats import reciprocal, beta, truncnorm
import matplotlib.pyplot as plt
import time
import os

class Generators(object):
    """"Class for generation of all examples and simulations"""

    @staticmethod
    def generate_gaussian_white_noise_trajectory(std, mean_func=None, points_nb=100, random_seed=None, **kwargs):
        if random_seed is not None:
            np.random.seed(random_seed)
        path_coordinates = list(np.linspace(0, 1, points_nb))
        noise = np.random.normal(loc=0.0, scale=std, size=points_nb)
        if mean_func is not None:
            mean_values = np.array([mean_func(t) for t in path_coordinates])
        else:
            mean_values = np.zeros(points_nb)
        path_values = list(mean_values + noise)
        trajectory = interp1d(path_coordinates, path_values, **kwargs)
        return trajectory, (path_values, path_coordinates)

    @staticmethod
    def periodic_func1(t):
        return 0.25 * np.sin(4 * np.pi * t) + 1 / 3 * np.cos(4 * np.pi * t) - 1 / 2 * np.sin(2 * np.pi * t)

    @staticmethod
    def periodic_func2(t):
        return 0.5 * np.cos(4 * t * np.pi + np.pi) + 0.75 * np.sin(2 * np.pi * t)

    @staticmethod
    def nonperiodic_func1(t):
        return 0.5 * np.cos(4 * (t ** 2) * np.pi + np.pi / 2) + 0.5 * np.exp(t ** 2)

    @staticmethod
    def nonperiodic_func2(t):
        return 0.5 * np.cos(4 * (t ** 2) * np.pi + np.pi / 2 + 0.5) + 0.8 * np.exp(t ** 2 + 0.5)

    @staticmethod
    def generate_reciprocal_simulation(a, b, samples_nb, random_seed=None):
        if not (random_seed is None):
            np.random.seed(random_seed)
        simulation_samples = list(reciprocal(a, b).rvs(samples_nb))
        return simulation_samples

    @staticmethod
    def get_reciprocal_pdf_function(a, b):
        return reciprocal(a, b).pdf

    @staticmethod
    def generate_beta_simulation(a, b, samples_nb, random_seed=None):
        if not (random_seed is None):
            np.random.seed(random_seed)
        simulation_samples = list(beta(a, b).rvs(samples_nb))
        return simulation_samples

    @staticmethod
    def get_beta_pdf_function(a, b):
        return beta(a, b).pdf

    @staticmethod
    def generate_truncnorm_simulation(a, b, mean, std, samples_nb, random_seed=None):
        if not (random_seed is None):
            np.random.seed(random_seed)
        corrected_a, corrected_b = (a - mean) / std, (b - mean) / std
        simulation_samples = list(truncnorm(corrected_a, corrected_b, mean, std).rvs(samples_nb))
        return simulation_samples

    @staticmethod
    def get_truncnorm_pdf_function(a, b, mean, std):
        corrected_a, corrected_b = (a - mean) / std, (b - mean) / std
        return truncnorm(corrected_a, corrected_b, mean, std).pdf

    @staticmethod
    def generate_bimodal_truncnorm_simulation(a, b, mean1, std1, mean2, std2, samples_nb, random_seed=None):
        samples_nb1 = int(samples_nb / 2)
        samples_nb2 = samples_nb - samples_nb1

        if not (random_seed is None):
            np.random.seed(random_seed)
        corrected_a1, corrected_b1 = (a - mean1) / std1, (b - mean1) / std1
        simulation_samples1 = list(truncnorm(corrected_a1, corrected_b1, mean1, std1).rvs(samples_nb1))

        if not (random_seed is None):
            np.random.seed(random_seed + 1)
        corrected_a2, corrected_b2 = (a - mean2) / std2, (b - mean2) / std2
        simulation_samples2 = list(truncnorm(corrected_a2, corrected_b2, mean2, std2).rvs(samples_nb2))

        simulation_samples = simulation_samples1 + simulation_samples2
        if not (random_seed is None):
            np.random.seed(random_seed + 2)
        np.random.shuffle(simulation_samples)

        return simulation_samples

    @staticmethod
    def get_bimodal_truncnorm_pdf_function(a, b, mean1, std1, mean2, std2):
        corrected_a1, corrected_b1 = (a - mean1) / std1, (b - mean1) / std1
        truncnorm_pdf1 = truncnorm(corrected_a1, corrected_b1, mean1, std1).pdf
        corrected_a2, corrected_b2 = (a - mean2) / std2, (b - mean2) / std2
        truncnorm_pdf2 = truncnorm(corrected_a2, corrected_b2, mean2, std2).pdf

        def bimodal_truncnorm_pdf(t):
            return (truncnorm_pdf1(t) + truncnorm_pdf2(t)) / 2

        return bimodal_truncnorm_pdf


def visualize_reciprocal_pdf_estimators(out_path=None):
    g = Generators()
    samples_nb = 10000
    pdf_name, pdf_func, simulation_samples, (pdf_supp_min, pdf_supp_max) = (
            'reciprocal',
            g.get_reciprocal_pdf_function(0.1, 1.1),
            g.generate_reciprocal_simulation(0.1, 1.1, samples_nb=samples_nb, random_seed=5441111),
            (0.1, 1.1),
        )

    t_grid = np.linspace(pdf_supp_min, pdf_supp_max, 1000)
    estimators = {}

    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=4, edge_strategy='folding_extension'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('pinsker', dict(L=1.0, beta=0.5, reflected_extension=True))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('wavelet', dict(db_order=2, resolution=3, edge_strategy='boundaries_correction'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=2, res.=3$); \nboundaries correction'] = [
        gt_density.score_density_value(t) for t in t_grid]

    fig, ax = plt.subplots()
    ax.plot(t_grid, [pdf_func(t) for t in t_grid], color='red',
            label='Reciprocal PDF (a=0.1, b=1.1)')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=2, res.=3$); \nboundaries correction'],
            label='Wavelet estimator ($N=2, res.=3$); \nboundaries correction', ls=':', lw=2, color='green')

    ax.plot(t_grid, estimators['Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension'],
            label='Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension', ls='--', lw=1, color='black')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'],
            label='Wavelet estimator ($N=3, res.=4$); \nfolding', color='blue')

    ax.legend()

    if out_path is not None:
        plt.savefig(os.path.join(out_path, '{}_reciprocal_pdf_estimators.png'.format(
            time.strftime('%Y%m%d_%H%M%S'))),
                    quality=100, dpi=500)
    else:
        plt.show()


def visualize_beta_pdf_estimators(out_path=None):
    g = Generators()
    samples_nb = 10000
    pdf_name, pdf_func, simulation_samples, (pdf_supp_min, pdf_supp_max) = (
            'beta',
            g.get_beta_pdf_function(2, 2),
            g.generate_beta_simulation(2, 2, samples_nb=samples_nb, random_seed=2352555),
            (0.0, 1.0),
        )

    t_grid = np.linspace(pdf_supp_min, pdf_supp_max, 1000)
    estimators = {}

    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=4, edge_strategy='folding_extension'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('pinsker', dict(L=1.0, beta=1.0, reflected_extension=False))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Pinsker ($L=1, \\beta=1.0$)'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=4, edge_strategy='boundaries_correction'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=4$); \nboundaries correction'] = [
        gt_density.score_density_value(t) for t in t_grid]

    fig, ax = plt.subplots()
    ax.plot(t_grid, [pdf_func(t) for t in t_grid], color='red',
            label='Beta PDF (a=2, b=2)')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=4$); \nboundaries correction'],
            label='Wavelet estimator ($N=3, res.=4$); \nboundaries correction', ls=':', lw=2, color='green')

    ax.plot(t_grid, estimators['Pinsker ($L=1, \\beta=1.0$)'],
            label='Pinsker ($L=1, \\beta=1.0$)', ls='--', lw=1, color='black')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'],
            label='Wavelet estimator ($N=3, res.=4$); \nfolding', color='blue')

    ax.legend()

    if out_path is not None:
        plt.savefig(os.path.join(out_path, '{}_beta_pdf_estimators.png'.format(
            time.strftime('%Y%m%d_%H%M%S'))),
                    quality=100, dpi=500)
    else:
        plt.show()


def visualize_truncnorm_pdf_estimators(out_path=None):
    g = Generators()
    samples_nb = 10000
    pdf_name, pdf_func, simulation_samples, (pdf_supp_min, pdf_supp_max) = (
            'truncnorm',
            g.get_truncnorm_pdf_function(0, 1, mean=0.8, std=0.3),
            g.generate_truncnorm_simulation(0, 1, mean=0.8, std=0.3, samples_nb=samples_nb, random_seed=11203220),
            (0.0, 1.0),
        )

    t_grid = np.linspace(pdf_supp_min, pdf_supp_max, 1000)
    estimators = {}

    estimation_algorithm, params = ('wavelet', dict(db_order=4, resolution=3, edge_strategy='folding_extension'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=4, res.=3$); \nfolding'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('pinsker', dict(L=1.0, beta=1.0, reflected_extension=True))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Pinsker ($L=1, \\beta=1.0$) \nafter reflected extension'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=3, edge_strategy='boundaries_correction'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=3$); \nboundaries correction'] = [
        gt_density.score_density_value(t) for t in t_grid]

    fig, ax = plt.subplots()
    ax.plot(t_grid, [pdf_func(t) for t in t_grid], color='red',
            label='Truncated normal PDF \n($a = 0, b = 1, \mu = 0.8, \sigma=0.3$)')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=3$); \nboundaries correction'],
            label='Wavelet estimator \n($N=3, res.=3$); \nboundaries correction', ls=':', lw=2, color='green')

    ax.plot(t_grid, estimators['Pinsker ($L=1, \\beta=1.0$) \nafter reflected extension'],
            label='Pinsker ($L=1, \\beta=1.0$) \nafter reflected extension', ls='--', lw=1, color='black')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=4, res.=3$); \nfolding'],
            label='Wavelet estimator \n($N=4, res.=3$); folding', color='blue')

    ax.legend()

    if out_path is not None:
        plt.savefig(os.path.join(out_path, '{}_truncnorm_pdf_estimators.png'.format(
            time.strftime('%Y%m%d_%H%M%S'))),
                    quality=100, dpi=500)
    else:
        plt.show()


def visualize_bimodal_pdf_estimators(out_path=None):
    g = Generators()
    samples_nb = 10000
    pdf_name, pdf_func, simulation_samples, (pdf_supp_min, pdf_supp_max) = (
            'bimodal_truncnorm',
            g.get_bimodal_truncnorm_pdf_function(
                0, 1, mean1=0.7, std1=0.15, mean2=0.35, std2=0.1),
            g.generate_bimodal_truncnorm_simulation(
                0, 1, mean1=0.7, std1=0.15, mean2=0.35, std2=0.1, samples_nb=samples_nb, random_seed=435331),
            (0.0, 1.0),
        )

    t_grid = np.linspace(pdf_supp_min, pdf_supp_max, 1000)
    estimators = {}

    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=4, edge_strategy='folding_extension'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('pinsker', dict(L=1.0, beta=0.5, reflected_extension=True))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension'] = [
        gt_density.score_density_value(t) for t in t_grid]
    # ---
    estimation_algorithm, params = ('wavelet', dict(db_order=3, resolution=4, edge_strategy='boundaries_correction'))
    gt_density = GaussianTrajectoryDensity(
        estimation_algorithm=estimation_algorithm, gamma=1,
        pdf_supp_min=pdf_supp_min, pdf_supp_max=pdf_supp_max)
    gt_density.fit(simulation_samples, **params)
    estimators['Wavelet estimator ($N=3, res.=4$); \nboundaries correction'] = [
        gt_density.score_density_value(t) for t in t_grid]

    fig, ax = plt.subplots()
    ax.plot(t_grid, [pdf_func(t) for t in t_grid], color='red',
            label='Bimodal PDF')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=4$); \nboundaries correction'],
            label='Wavelet estimator \n($N=3, res.=4$); \nboundaries correction', ls=':', lw=2, color='green')

    ax.plot(t_grid, estimators['Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension'],
            label='Pinsker ($L=1, \\beta=0.5$) \nafter reflected extension', ls='--', lw=1, color='black')

    ax.plot(t_grid, estimators['Wavelet estimator ($N=3, res.=4$); \nfolding'],
            label='Wavelet estimator \n($N=3, res.=4$); \nfolding', color='blue')

    ax.legend()

    if out_path is not None:
        plt.savefig(os.path.join(out_path, '{}_bimodal_pdf_estimators.png'.format(
            time.strftime('%Y%m%d_%H%M%S'))),
                    quality=100, dpi=500)
    else:
        plt.show()


def test_estimators(out_path=None):
    visualize_reciprocal_pdf_estimators(out_path)
    visualize_beta_pdf_estimators(out_path)
    visualize_truncnorm_pdf_estimators(out_path)
    visualize_bimodal_pdf_estimators(out_path)


if __name__ == '__main__':
    test_estimators()
