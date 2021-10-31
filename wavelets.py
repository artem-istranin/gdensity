# ==============================
# Daubechies scaling functions and wavelets
# ==============================

import scipy.signal
import numpy as np
from numpy import sqrt as sqrt


def db_scaling_function_and_wavelet(db_order, level=10, method='binary_interpolation'):
    """
        Generate values of Daubechies scaling function and wavelet in corresponding t-grid coordinates.

        :param db_order: int
            Order of Daubechies scaling function and wavelet;
        :param level: int
             Interpolation level. Function returns values in interval [0, 2 * db_order - 1] with step 1 / (2 ** level);
        :param method: str
            The function supports the following methods:
                1) 'iterative_approximation': iterative method, based on Fixed-point theorem;
                2) 'binary_interpolation': interpolation of the exact values on dyadic rationals t-grid;

        :returns
            phi: list
                List of Daubechies scaling function in corresponding t-grid coordinates;
            psi: list
                List of Daubechies wavelet in corresponding t-grid coordinates;
            t_grid: list
                List of t-grid coordinates.
    """
    if not isinstance(db_order, int) or db_order < 2:
        raise ValueError('Parameter `db_order` must be positive integer.')
    if not isinstance(level, int) or level < 0:
        raise ValueError('Parameter `level` must be non-negative integer.')

    if method == 'iterative_approximation':
        # t-grid coordinates:
        _t_step = 1 / (2 ** level)  # after x iterations, g is a piecewise constant function with step 1 / (2 ** x)
        t_grid = list(np.arange(_t_step, 2 * db_order - 1, _t_step))  # compact support [0, 2 * db_order - 1]

        # Scaling function:
        coefficients = scipy.signal.daub(db_order)  # coefficients vector
        g_list = []

        def g_0(t):
            return int(0 < t <= 1)
        g_list.append(g_0)

        def next_g(last_g):
            def _res(t):
                _output_sum = 0
                for k in range(len(coefficients)):
                    if 0 < 2 * t - k < 2 * db_order - 1:
                        _output_sum += sqrt(2) * coefficients[k] * last_g(2 * t - k)
                return _output_sum
            return _res

        for r in range(1, level + 1):
            g_list.append(next_g(last_g=g_list[-1]))
        scaling_function_approx = g_list[-1]
        phi = [scaling_function_approx(t) for t in t_grid]

        # Wavelet:
        wavelet_coefficients = [(-1) ** (k + 1) * coefficients[-k] for k in range(1, len(coefficients) + 1)]

        def wavelet_approx(t):
            _output_sum = 0
            for k in range(len(wavelet_coefficients)):
                if 0 < 2 * t - k < 2 * db_order - 1:
                    _output_sum += sqrt(2) * wavelet_coefficients[k] * scaling_function_approx(2 * t - k)
            return _output_sum
        psi = [wavelet_approx(t) for t in t_grid]

        t_grid = [0] + t_grid + [2 * db_order - 1]
        phi = [0] + phi + [0]
        psi = [0] + psi + [0]
    elif method == 'binary_interpolation':
        # t-grid coordinates:
        t_grid_collection = []

        # Scaling function in integers [0, 2 * db_order - 1]:
        phi_collection = []
        coefficients = scipy.signal.daub(db_order)  # coefficients vector

        b_matrix = []
        for j in range(len(coefficients)):
            curr_row = []
            for k in range(len(coefficients)):
                coefficient_ind = 2 * j - k
                if 0 <= coefficient_ind <= len(coefficients) - 1:
                    curr_row.append(sqrt(2) * coefficients[coefficient_ind])
                else:
                    curr_row.append(0)
            b_matrix.append(curr_row)

        # form system of linear equations:
        for i in range(len(b_matrix)):
            b_matrix[i][i] = b_matrix[i][i] - 1
        b_matrix = np.array(b_matrix)
        # phi equals 0 in t = 0 and t= 2 * db_order - 1 because phi is continuous
        # and has compact support [0, 2 * db_order - 1]:
        b_matrix = b_matrix[1: -1]
        b_matrix = b_matrix.T[1: -1].T
        # adding constraint:
        b_matrix = b_matrix.tolist() + [[1.] * (len(coefficients) - 2)]
        # least-squares solution to a linear matrix equation with constraint:
        phi_integers = np.linalg.lstsq(b_matrix, [0] * (len(b_matrix) - 1) + [1], rcond=None)[0].tolist()

        t_grid_collection.append(list(range(1, 2 * db_order - 1)))
        phi_collection.append(phi_integers)

        # Wavelet:
        wavelet_coefficients = [(-1) ** (k + 1) * coefficients[-k] for k in range(1, len(coefficients) + 1)]
        psi_collection = []

        psi_integers = []
        for t in range(1, 2 * db_order - 1):
            _psi_value = 0
            for k in range(len(wavelet_coefficients)):
                if 0 < 2 * t - k < 2 * db_order - 1:
                    _ref_ind = int(2 * t - k - 1)
                    _psi_value += sqrt(2) * wavelet_coefficients[k] * phi_collection[-1][_ref_ind]
            psi_integers.append(_psi_value)
        psi_collection.append(psi_integers)

        # Recursion:
        for curr_level in range(1, level + 1):
            curr_slice = 1 / (2 ** curr_level)
            curr_t_grid = list(np.arange(curr_slice, 2 * db_order - 1, curr_slice * 2))

            curr_phi_values = []
            curr_psi_values = []
            for t in curr_t_grid:
                _phi_value = 0
                _psi_value = 0
                for k in range(len(coefficients)):
                    if 0 < 2 * t - k < 2 * db_order - 1:

                        # find corresponding index of phi in the previous level:
                        _ref_slice = curr_slice * 2
                        if _ref_slice == 1:
                            _ref_ind = int(2 * t - k - 1)
                        else:
                            _ref_ind = int(
                                ((2 * t - k) / _ref_slice) - (((2 * t - k) - _ref_slice) / (2 * _ref_slice)) - 1)

                        _phi_value += sqrt(2) * coefficients[k] * phi_collection[-1][_ref_ind]
                        _psi_value += sqrt(2) * wavelet_coefficients[k] * phi_collection[-1][_ref_ind]
                curr_phi_values.append(_phi_value)
                curr_psi_values.append(_psi_value)

            t_grid_collection.append(curr_t_grid)
            phi_collection.append(curr_phi_values)
            psi_collection.append(curr_psi_values)

        t_grid = np.concatenate(t_grid_collection)
        sorted_indices = np.argsort(t_grid)
        t_grid = [0] + t_grid[sorted_indices].tolist() + [2 * db_order - 1]
        phi = [0] + np.concatenate(phi_collection)[sorted_indices].tolist() + [0]
        psi = [0] + np.concatenate(psi_collection)[sorted_indices].tolist() + [0]
    else:
        raise NotImplementedError('{} method is not developed'.format(method))
    return phi, psi, t_grid


_ScalingFunc_LeftEdgeCoefficients = {
    'db2':
    [
        # k = 0:
        [0.6033325119E+00, 0.6908955318E+00, -0.3983129977E+00],
        # k = 1:
        [0.3751746045E-01, 0.4573276599E+00, 0.8500881025E+00, 0.2238203570E+00,
         -0.1292227434E+00]
    ],
    # 'db4':
    # [
    #     # k = 0:
    #     [0.9097539258E+00, 0.4041658894E+00, 0.8904031866E-01, -0.1198419201E-01,
    #      -0.3042908414E-01],
    #     # k = 1:
    #     [-0.2728514077E+00, 0.5090815232E+00, 0.6236424434E+00, 0.4628400863E+00,
    #      0.2467476417E+00, -0.1766953329E-01, -0.4517364549E-01],
    #     # k = 2:
    #     [0.1261179286E+00, -0.2308557268E+00, -0.5279923525E-01, 0.2192651713E+00,
    #      0.4634807211E+00, 0.7001197140E+00, 0.4120325790E+00, -0.2622276250E-01,
    #      -0.6704069413E-01],
    #     # k = 3:
    #     [-0.2907980427E-01, 0.5992807229E-01, 0.6176427778E-02, -0.4021099904E-01,
    #      -0.3952587013E-01, -0.5259906257E-01, 0.3289494480E+00, 0.7966378967E+00,
    #      0.4901130336E+00, -0.2943287768E-01, -0.7524762313E-01]
    # ],
}

_Wavelets_LeftEdgeCoefficients = {
    'db2':
    [
        # k = 0:
        [-0.7965435169E+00, 0.5463927140E+00, -0.2587922483E+00],
        # k = 1:
        [0.1003722456E-01, 0.1223510431E+00, 0.2274281117E+00, -0.8366029212E+00,
         0.4830129218E+00]
    ],
    # 'db4':
    # [
    #     # k = 0:
    #     [-0.7573970762E-01, 0.3254391718E+00, -0.6843490805E+00, 0.6200442107E+00,
    #      -0.1885851398E+00],
    #     # k = 1:
    #     [0.1665959592E+00, -0.4847843089E+00, 0.3564635425E+00, 0.4839896156E+00,
    #      -0.6057543651E+00, 0.3451833289E-01, 0.8824901639E-01],
    #     # k = 2:
    #     [0.2082535326E+00, -0.4018227932E+00, -0.6872148762E-01, 0.3302135113E+00,
    #      0.5580212962E+00, -0.5994974134E+00, -0.6909199198E-01, 0.2785356997E-01,
    #      0.7120999037E-01],
    #     # k = 3:
    #     [0.6548500701E-01, -0.1349524295E+00, -0.1390873929E-01, 0.9055141946E-01,
    #      0.8900857304E-01, 0.3733444476E+00, -0.8404653708E+00, 0.3156849362E+00,
    #      0.1202976509E+00, -0.1307020280E-01, -0.3341507090E-01]
    # ],
}

_ScalingFunc_RightEdgeCoefficients = {
    'db2':
    [
        # k = -2:
        [0.4431490496E+00, 0.7675566693E+00, 0.3749553316E+00, 0.1901514184E+00,
         -0.1942334074E+00],
        # k = -1:
        [0.2303890438E+00, 0.4348969980E+00, 0.8705087534E+00]
    ],
    # 'db4':
    # [
    #     # k = -4:
    #     [0.3221027840E-01, -0.1259895190E-01, -0.9910804055E-01, 0.2977111011E+00,
    #      0.8039495996E+00, 0.4977920821E+00, -0.3023588481E-01, -0.6765916174E-01,
    #      -0.1770918425E-01, 0.1913244129E-01, -0.6775603652E-02],
    #     # k = -3:
    #     [0.3214874197E-01, -0.1257488211E-01, -0.1027663536E+00, 0.2986473346E+00,
    #      0.8164119742E+00, 0.4606168537E+00, 0.2921367950E-01, -0.1390716006E+00,
    #      0.1290078289E-01],
    #     # k = -2:
    #     [0.4126840881E-01, -0.1614201190E-01, -0.1581338944E+00, 0.3937758157E+00,
    #      0.7540005084E+00, 0.4488001781E+00, -0.2191626469E+00],
    #     # k = -1:
    #     [0.6437934569E-01, -0.2519180851E-01, 0.5947771124E-01, 0.3919142810E+00,
    #      0.9154705188E+00]
    # ],
}

_Wavelets_RightEdgeCoefficients = {
    'db2':
    [
        # k = -2:
        [0.2315575950E+00, 0.4010695194E+00, -0.7175799994E+00, -0.3639069596E+00,
         0.3717189665E+00],
        # k = -1:
        [-0.5398225007E+00, 0.8014229620E+00, -0.2575129195E+00]
    ],
    # 'db4':
    # [
    #     # k = -4:
    #     [0.7577116678E-01, -0.2963766018E-01, -0.4976470524E+00, 0.8037936841E+00,
    #      -0.2977899841E+00, -0.9920191439E-01, 0.1285325684E-01, 0.2876186983E-01,
    #      0.7528163799E-02, -0.8133189531E-02, 0.2880305124E-02],
    #     # k = -3:
    #     [0.7575680778E-01, -0.2963204370E-01, -0.4968427065E+00, 0.8033636317E+00,
    #      -0.3015815012E+00, -0.9504834999E-01, 0.1388737145E-01, 0.3062032365E-01,
    #      0.4581959340E-02],
    #     # k = -2:
    #     [-0.6951519362E-01, 0.2719065540E-01, 0.4499340827E+00, -0.6735345752E+00,
    #      0.6811856626E-01, 0.5092867484E+00, -0.2726273506E+00],
    #     # k = -1:
    #     [-0.9924194741E-01, 0.4050309541E+00, -0.6495297603E+00, 0.6040677868E+00,
    #      -0.1982779906E+00]
    # ],
}


VALID_BC_WAVELETS_ORDER = [2]  # TODO: append higher orders


def bc_db_scaling_functions_and_wavelets(db_order, level=10, **kwargs):
    """"
        Generate values of boundary corrected Daubechies scaling function and wavelet
        in corresponding t-grid coordinates.

        :param db_order: int
            Order of Daubechies scaling function and wavelet;
        :param level: int
             Interpolation level. Function returns values in the support interval with step 1 / (2 ** level);

        :returns
            bc_db_collection: dict
                Dictionary with keys 'left' and 'right' for left and right edge scaling functions and wavelets
                respectively. The values of this keys are list of tuples [..., (phi_k, psi_k, t_grid_k), ...],
                where
                    phi_k: list
                        List of k-th boundary-corrected Daubechies scaling function in
                        corresponding t-grid coordinates;
                    psi_k: list
                        List of k-th boundary-corrected Daubechies wavelet in
                        corresponding t-grid coordinates;
                    t_grid_k: list
                        List of corresponding t-grid coordinates for k-th boundary-corrected
                        scaling functions and wavelets.
                Left edge functions are indexed for k = 0, ..., db_order - 1, right edge functions for
                k = -1, ..., -db_order in accordance with the original article A. Cohen, I. Daubechies, and P. Vial.
                “Wavelets on the Interval and Fast Wavelet Transforms”. 1993.
    """
    if not isinstance(db_order, int) or db_order < 2:
        raise ValueError('Parameter `db_order` must be positive integer.')
    if db_order not in VALID_BC_WAVELETS_ORDER:
        raise NotImplementedError('Boundary-corrected wavelets construction '
                                  'of order {} is not yet developed'.format(db_order))
    if not isinstance(level, int) or level < 0:
        raise ValueError('Parameter `level` must be non-negative integer.')

    phi, psi, x_grid_for_phi_psi = db_scaling_function_and_wavelet(db_order=db_order, level=level, **kwargs)

    def p(x_value):
        # return phi for x_value, where x_value is referencing to the translated version of phi
        # with supp [-db_order + 1, db_order]
        x_value += db_order - 1
        if not(0 <= x_value <= 2 * db_order - 1):
            return 0
        _slice = 1 / (2 ** level)
        if _slice == 1:
            _ind = int(x_value)
        else:
            _ind = int(x_value / _slice)
        return phi[_ind]

    bc_db_collection = dict(left=None, right=None)

    # 1. Left boundary
    bc_left_scaling_functions = []  # list of tuples (<phi_collection>, <x_grid_collection>)
    left_scaling_func_coefficients = _ScalingFunc_LeftEdgeCoefficients['db{}'.format(db_order)]

    # Left scaling function in integers:
    equations_nb = sum([db_order + k + 1 for k in range(0, (db_order - 1) + 1)])
    variables_nb = equations_nb
    a = np.zeros((equations_nb, variables_nb))
    b = np.zeros(equations_nb)

    eq_counter = 0
    for k in range(0, (db_order - 1) + 1):
        bc_left_scaling_functions.append(([[]], [[]]))  # new tuple for current k
        for int_x in range(0, (db_order + k) + 1):
            bc_left_scaling_functions[-1][1][-1].append(int_x)  # <x_grid_collection> update
            for q in range(0, (db_order - 1) + 1):
                if 0 <= 2 * int_x <= db_order + q:
                    var_index = sum([db_order + g + 1 for g in range(0, (db_order - 1) + 1) if g < q]) + int(2 * int_x)
                    a[eq_counter][var_index] = sqrt(2) * left_scaling_func_coefficients[k][q]
            a[eq_counter][eq_counter] -= 1
            b[eq_counter] = sum([-sqrt(2) * left_scaling_func_coefficients[k][m] * p(2 * int_x - m) for
                                 m in range(db_order, (db_order + 2 * k) + 1)])
            eq_counter += 1

    bc_left_phi_integers = np.linalg.solve(a, b)
    eq_counter = 0
    for k in range(0, (db_order - 1) + 1):
        for int_x in range(0, (db_order + k) + 1):
            bc_left_scaling_functions[k][0][-1].append(bc_left_phi_integers[eq_counter])  # <phi_collection> update
            eq_counter += 1

    # Recursion:
    for curr_level in range(1, level + 1):

        for k in range(0, (db_order - 1) + 1):
            bc_left_scaling_functions[k][0].append([])  # new empty list for new phi values
            bc_left_scaling_functions[k][1].append([])  # new empty list for new x values

        for k in range(0, (db_order - 1) + 1):
            curr_slice = 1 / (2 ** curr_level)
            curr_x_grid = list(np.arange(curr_slice, db_order + k, curr_slice * 2))
            for x in curr_x_grid:
                bc_left_scaling_functions[k][1][-1].append(x)

                left_bc_phi_value = 0
                for q in range(0, (db_order - 1) + 1):

                    # find corresponding index of phi_l(2x):
                    if not (0 <= 2 * x <= db_order + q):
                        left_l_2x = 0
                    else:
                        _ref_slice = curr_slice * 2
                        if _ref_slice == 1:
                            _ref_ind = int(2 * x)
                        else:
                            _ref_ind = int(
                                ((2 * x) / _ref_slice) - (((2 * x) - _ref_slice) / (2 * _ref_slice)) - 1)
                        left_l_2x = bc_left_scaling_functions[q][0][-2][_ref_ind]

                    left_bc_phi_value += sqrt(2) * left_scaling_func_coefficients[k][q] * left_l_2x

                left_bc_phi_value += sqrt(2) * sum([left_scaling_func_coefficients[k][m] * p(2 * x - m) for
                                                    m in range(db_order, (db_order + 2 * k) + 1)])

                bc_left_scaling_functions[k][0][-1].append(left_bc_phi_value)

    for k in range(0, (db_order - 1) + 1):
        x_grid_collection_k = bc_left_scaling_functions[k][1]
        x_grid_collection_k = np.concatenate(x_grid_collection_k)
        sorted_indices = np.argsort(x_grid_collection_k)
        t_grid_k = list(x_grid_collection_k[sorted_indices])

        left_phi_collection_k = bc_left_scaling_functions[k][0]
        left_phi_k = list(np.concatenate(left_phi_collection_k)[sorted_indices])

        bc_left_scaling_functions[k] = (left_phi_k, t_grid_k)

    # Left wavelets:
    bc_left_wavelets = []  # list of tuples (<psi_collection>, <x_grid_collection>)
    left_wavelets_coefficients = _Wavelets_LeftEdgeCoefficients['db{}'.format(db_order)]

    for k in range(0, (db_order - 1) + 1):
        bc_left_wavelets.append(([], []))  # new tuple for current k

        top_slice = 1 / (2 ** level)
        top_x_grid = list(np.arange(0, db_order + k + top_slice, top_slice))
        for x in top_x_grid:
            bc_left_wavelets[k][1].append(x)

            left_bc_psi_value = 0
            for q in range(0, (db_order - 1) + 1):

                # find corresponding phi_l(2x):
                if not (0 <= 2 * x <= db_order + q):
                    left_l_2x = 0
                else:
                    _ref_slice = 1 / (2 ** level)
                    if _ref_slice == 1:
                        ind = int(2 * x)
                    else:
                        ind = int((2 * x) / _ref_slice)
                    left_l_2x = bc_left_scaling_functions[q][0][ind]

                left_bc_psi_value += sqrt(2) * left_wavelets_coefficients[k][q] * left_l_2x

            left_bc_psi_value += sqrt(2) * sum([left_wavelets_coefficients[k][m] * p(2 * x - m) for
                                                m in range(db_order, (db_order + 2 * k) + 1)])

            bc_left_wavelets[k][0].append(left_bc_psi_value)

    # Left scaling functions and wavelets concatenation:
    res_left_scaling_functions_and_wavelets = []
    for k in range(0, (db_order - 1) + 1):
        res_left_scaling_functions_and_wavelets.append(
            (bc_left_scaling_functions[k][0], bc_left_wavelets[k][0], bc_left_scaling_functions[k][1])
        )
    bc_db_collection['left'] = res_left_scaling_functions_and_wavelets

    # 2. Right boundary
    bc_right_scaling_functions = []  # list of tuples (<phi_collection>, <x_grid_collection>)
    right_scaling_func_coefficients = _ScalingFunc_RightEdgeCoefficients['db{}'.format(db_order)]

    # Left scaling function in integers:
    equations_nb = sum([abs(k - db_order + 1) + 1 for k in range(-db_order, (-1) + 1)])
    variables_nb = equations_nb
    a = np.zeros((equations_nb, variables_nb))
    b = np.zeros(equations_nb)

    eq_counter = 0
    for k in range(-db_order, (-1) + 1):
        bc_right_scaling_functions.append(([[]], [[]]))  # new tuple for current k
        for int_x in range(k - db_order + 1, 0 + 1):
            bc_right_scaling_functions[-1][1][-1].append(int_x)  # <x_grid_collection> update
            for q in range(-db_order, (-1) + 1):
                if q - db_order + 1 <= 2 * int_x <= 0:
                    var_index = sum([abs(g - db_order + 1) + 1 for g in range(-db_order, (-1) + 1) if g < q]) + \
                                (abs(q - db_order + 1) - int(abs(2 * int_x)))
                    a[eq_counter][var_index] = sqrt(2) * right_scaling_func_coefficients[k][q]
            a[eq_counter][eq_counter] -= 1
            b[eq_counter] = sum([-sqrt(2) * right_scaling_func_coefficients[k][m] * p(2 * int_x - m) for
                                 m in range(-db_order + 2 * k + 1, (-db_order - 1) + 1)])
            eq_counter += 1

    bc_right_phi_integers = np.linalg.solve(a, b)
    eq_counter = 0
    for k in range(-db_order, (-1) + 1):
        for int_x in range(k - db_order + 1, 0 + 1):
            bc_right_scaling_functions[k][0][-1].append(bc_right_phi_integers[eq_counter])  # <phi_collection> update
            eq_counter += 1

    # Recursion:
    for curr_level in range(1, level + 1):

        for k in range(-db_order, (-1) + 1):
            bc_right_scaling_functions[k][0].append([])  # new empty list for new phi values
            bc_right_scaling_functions[k][1].append([])  # new empty list for new x values

        for k in range(-db_order, (-1) + 1):
            curr_slice = 1 / (2 ** curr_level)
            curr_x_grid = list(np.arange(k - db_order + 1 + curr_slice, 0, curr_slice * 2))
            for x in curr_x_grid:
                bc_right_scaling_functions[k][1][-1].append(x)

                right_bc_phi_value = 0
                for q in range(-db_order, (-1) + 1):

                    # find corresponding index of phi_l(2x):
                    if not (q - db_order + 1 <= 2 * x <= 0):
                        right_l_2x = 0
                    else:
                        _ref_slice = curr_slice * 2
                        if _ref_slice == 1:
                            _ref_ind = int(2 * x) - 1
                        else:
                            _ref_ind = -int(
                                ((2 * abs(x)) / _ref_slice) - (((2 * abs(x)) - _ref_slice) / (2 * _ref_slice)) - 1) - 1
                        right_l_2x = bc_right_scaling_functions[q][0][-2][_ref_ind]

                    right_bc_phi_value += sqrt(2) * right_scaling_func_coefficients[k][q] * right_l_2x

                right_bc_phi_value += sum([sqrt(2) * right_scaling_func_coefficients[k][m] * p(2 * x - m) for
                                           m in range(-db_order + 2 * k + 1, (-db_order - 1) + 1)])

                bc_right_scaling_functions[k][0][-1].append(right_bc_phi_value)

    for k in range(-db_order, (-1) + 1):
        x_grid_collection_k = bc_right_scaling_functions[k][1]
        x_grid_collection_k = np.concatenate(x_grid_collection_k)
        sorted_indices = np.argsort(x_grid_collection_k)
        t_grid_k = list(x_grid_collection_k[sorted_indices])

        right_phi_collection_k = bc_right_scaling_functions[k][0]
        right_phi_k = list(np.concatenate(right_phi_collection_k)[sorted_indices])

        bc_right_scaling_functions[k] = (right_phi_k, t_grid_k)

    # Right wavelets:
    bc_right_wavelets = []  # list of tuples (<psi_collection>, <x_grid_collection>)
    right_wavelets_coefficients = _Wavelets_RightEdgeCoefficients['db{}'.format(db_order)]

    for k in range(-db_order, (-1) + 1):
        bc_right_wavelets.append(([], []))  # new tuple for current k

    for k in range(-db_order, (-1) + 1):
        top_slice = 1 / (2 ** level)
        top_x_grid = list(np.arange(k - db_order + 1, 0 + top_slice, top_slice))
        for x in top_x_grid:
            bc_right_wavelets[k][1].append(x)

            right_bc_psi_value = 0
            for q in range(-db_order, (-1) + 1):

                # find corresponding phi_l(2x):
                if not (q - db_order + 1 <= 2 * x <= 0):
                    right_l_2x = 0
                else:
                    _ref_slice = 1 / (2 ** level)
                    if _ref_slice == 1:
                        ind = int(2 * x) - 1
                    else:
                        ind = -int((2 * abs(x)) / _ref_slice) - 1
                    right_l_2x = bc_right_scaling_functions[q][0][ind]

                right_bc_psi_value += sqrt(2) * right_wavelets_coefficients[k][q] * right_l_2x

            right_bc_psi_value += sum([sqrt(2) * right_wavelets_coefficients[k][m] * p(2 * x - m) for
                                       m in range(-db_order + 2 * k + 1, (-db_order - 1) + 1)])

            bc_right_wavelets[k][0].append(right_bc_psi_value)

    # Right scaling functions and wavelets concatenation:
    res_right_scaling_functions_and_wavelets = []
    for k in range(-db_order, (-1) + 1):
        res_right_scaling_functions_and_wavelets.append(
            (bc_right_scaling_functions[k][0], bc_right_wavelets[k][0], bc_right_scaling_functions[k][1])
        )
    bc_db_collection['right'] = res_right_scaling_functions_and_wavelets

    return bc_db_collection
