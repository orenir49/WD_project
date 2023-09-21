import numpy as np

from uncertainties import unumpy as unp, ufloat
from uncertainties import correlated_values_norm, correlation_matrix


# =============================================================================
#                Sample multicatiate distribution with correlation
# =============================================================================
def multivar_sample(mu, sigma, corr, n):
    cov = corr*(sigma[:, None] * sigma[None, :])
    # l = spla.cholesky(cov)
    # z = np.random.normal(size=(n, mu.shape[0]))
    # return z.dot(l) + mu
    return np.random.multivariate_normal(mu, cov, size=n)


# =============================================================================
#                       Calculate the geometric coefficients
# =============================================================================
def geometric_elements(thiele_innes,
                       thiele_innes_errors=np.zeros(4),
                       corr_matrix=np.identity(4)):
    """
    For the given set of orbital parameters by Gaia, this function calculates
    the standard geometrical elements (a, omega, Omega, and i). If the error
    estimates and covariance matrix are prodived, the error estimates on the
    calculated parameters are returned as well.

    Input: thiele_innes: Thiele Innes parameters [A,B,F,G] in milli-arcsec
           thiele_innes_errors : Corresponding errors.
           corr_matrix : Corresponding  4X4 correlation matrix.

  Output: geometric_parameters : orbital elements [a_mas, omega_deg, OMEGA_deg, i_deg]
            geometric_parameters_errors : Corresponding errors.
    """
    # Read the coefficients and assign the correlation matrix.
    # Create correlated quantities
    A, B, F, G = correlated_values_norm([(thiele_innes[0], thiele_innes_errors[0]),
                                         (thiele_innes[1], thiele_innes_errors[1]),
                                         (thiele_innes[2], thiele_innes_errors[2]),
                                         (thiele_innes[3], thiele_innes_errors[3])],
                                        corr_matrix)

    # This in an intermediate step in the formulae...
    p = (A ** 2 + B ** 2 + G ** 2 + F ** 2) / 2.
    q = A * G - B * F

    # Calculate the angular semimajor axis (already in mas)
    a_mas = unp.sqrt(p + unp.sqrt(p ** 2 - q ** 2))

    # Calculate the inclination and convert from radians to degrees
    i_deg = unp.arccos(q / (a_mas ** 2.)) * (180 / np.pi)

    # Calculate omega and Omega, then convert from radians to degrees
    omega_deg = 0.5 * (unp.arctan2(B - F, A + G) + unp.arctan2(-B - F, A - G)) * (180 / np.pi)
    OMEGA_deg = 0.5 * (unp.arctan2(B - F, A + G) - unp.arctan2(-B - F, A - G)) * (180 / np.pi)

    # Extract expectancy values and standard deviations
    geo_pars = np.array([unp.nominal_values(a_mas),
                         unp.nominal_values(omega_deg),
                         unp.nominal_values(OMEGA_deg),
                         unp.nominal_values(i_deg)])

    geo_pars_error = np.array([unp.std_devs(a_mas),
                              unp.std_devs(omega_deg),
                              unp.std_devs(OMEGA_deg),
                              unp.std_devs(i_deg)])

    return geo_pars, geo_pars_error


# =============================================================================
#                       Generate the correlation matrix
# =============================================================================
def make_corr_matrix(input_table, pars=None):
    """
    INPUT:
    input_table nss_two_body_orbit table.
    pars : list
            list of parameters for the corresponding solution of the desired
              target, in the same order as they appear in the Gaia table.
      """
    if pars is None:
        pars = get_par_list()

    # read the correlation vector
    corr_vec = input_table['corr_vec']
    # set the number of parameters in the table
    n_pars = len(pars)
    # define the correlation matrix.
    corr_mat = np.ones([n_pars, n_pars], dtype=float)

    # Read the matrix (lower triangle)
    ind = 0
    for i in range(n_pars):
        for j in range(i):
            corr_mat[j][i] = corr_vec[ind]
            corr_mat[i][j] = corr_vec[ind]
            ind += 1

    return corr_mat


# =============================================================================
#                       Read the data from the NSS table
# =============================================================================
def get_nss_data(input_table, source_id):

    target_idx = np.argwhere(input_table['source_id'] == source_id)[0][0]
    pars = get_par_list(input_table['nss_solution_type'][target_idx])
    corr_mat = make_corr_matrix(input_table[target_idx], pars=pars)

    mu, std = np.zeros(len(pars)), np.zeros(len(pars))
    for i, par in enumerate(pars):
        try:
            mu[i] = input_table[par][target_idx]
            std[i] = input_table[par + '_error'][target_idx]
        except KeyError:
            mu[i], std[i] = np.nan, np.nan

    nan_idxs = np.argwhere(np.isnan(corr_mat))
    corr_mat[nan_idxs[:, 0], nan_idxs[:, 1]] = 0.0

    return mu, std, corr_mat


# =============================================================================
#                  Get the parameter list for teh solution type
# =============================================================================
def get_par_list(solution_type=None):
    if (solution_type is None) or (solution_type=='Orbital'):
        return ('ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes',
                'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes',
                'eccentricity', 'period', 't_periastron')

    elif (solution_type=='OrbitalAlternative') or (solution_type=='OrbitalAlternativeValidated') \
            or (solution_type=='OrbitalTargetedSearch') or (solution_type=='OrbitalTargetedSearchValidated'):
        return ('ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes',
                'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes',
                'period', 'eccentricity', 't_periastron')

    elif solution_type=='AstroSpectroSB1':
        return ('ra', 'dec', 'parallax', 'pmra', 'pmdec', 'a_thiele_innes',
                'b_thiele_innes', 'f_thiele_innes', 'g_thiele_innes',
                'c_thiele_innes', 'h_thiele_innes', 'center_of_mass_velocity',
                'eccentricity', 'period', 't_periastron')


# =============================================================================
#                       A function to q from class-III AMRF
# =============================================================================
def calc_q(A):
    '''
    ***
    '''
    y = A**3

    # Calculate q minimum
    h = (y/2 + (y**2)/3 + (y**3)/27
         + np.sqrt(3)/18*y*np.sqrt(4*y+27))**(1/3)

    q = np.array(h + (2*y/3 + (y**2)/9)/h + y/3)

    return q



# =============================================================================
#                      calc AMRF
# =============================================================================
def calc_AMRF(par_in, par_in_errors, m1, m1_error, corr_matrix, bit_index=8191):
    """
    For the given set of orbital parameters by Gaia, this function calculates
    the standard geometrical elements (a, omega, Omega, and i). If the error
    estimates and covariance matrix are prodived, the error estimates on the
    calculated parameters are returned as well.

    Input: thiele_innes: Thiele Innes parameters [A,B,F,G] in milli-arcsec
           thiele_innes_errors : Corresponding errors.
           corr_matrix : Corresponding  4X4 correlation matrix.

  Output: class-III probability via monte carlo
    """
    # Read the coefficients and assign the correlation matrix.
    # Create correlated quantities. If the error is nan we assume 1e-6...
    par_in_errors[np.isnan(par_in_errors)] = 1e-6
    par_list = correlated_values_norm([(par_in[i], par_in_errors[i]) for i in np.arange(len(par_in))], corr_matrix)
    key_list = bit_index_map(bit_index)

    par = {key_list[i]: par_list[i] for i in np.arange(len(key_list))}
    par['mass'] = ufloat(m1, m1_error)

    # Add the G Thiele-Innes parameter if needed.
    if (bit_index == 8179) | (bit_index == 65435):
        G = -par['A']*par['F']/par['B']
    else:
        G = par['G']

    # This in an intermediate step in the formulae...
    p = (par['A'] ** 2 + par['B'] ** 2 + G ** 2 + par['F'] ** 2) / 2.
    q = par['A'] * G - par['B'] * par['F']

    # Calculate the angular semimajor axis (already in mas)
    a_mas = unp.sqrt(p + unp.sqrt(p ** 2 - q ** 2))

    # Calculate the inclination and convert from radians to degrees
    i_deg = unp.arccos(q / (a_mas ** 2.)) * (180 / np.pi)

    # Calculate the AMRF
    try:
        AMRF = a_mas / par['parallax'] * par['mass'] ** (-1 / 3) * (par['P']/ 365.25) ** (-2 / 3)

        # Calculate AMRF q
        y = AMRF ** 3
        h = (y/2 + (y**2)/3 + (y**3)/27
             + np.sqrt(3)/18*y*unp.sqrt(4*y+27))**(1/3)
        q = h + (2*y/3 + (y**2)/9)/h + y/3

        # Calculate AMRF secondary mass
        m2 = q*m1

        # Extract expectancy values and standard deviations
        pars = np.array([unp.nominal_values(AMRF),
                         unp.nominal_values(q),
                         unp.nominal_values(m1),
                         unp.nominal_values(m2),
                         unp.nominal_values(a_mas),
                        unp.nominal_values(i_deg)])

        pars_error = np.array([unp.std_devs(AMRF),
                               unp.std_devs(q),
                               unp.std_devs(m1),
                               unp.std_devs(m2),
                               unp.std_devs(a_mas),
                               unp.std_devs(i_deg)])

    except:
        # Extract expectancy values and standard deviations
        pars = np.array([np.nan,
                         np.nan,
                         unp.nominal_values(m1),
                         np.nan,
                         unp.nominal_values(a_mas),
                         unp.nominal_values(i_deg)])

        pars_error = np.array([np.nan,
                               np.nan,
                               unp.std_devs(m1),
                               np.nan,
                               unp.std_devs(a_mas),
                               unp.std_devs(i_deg)])

    return pars, pars_error


# =============================================================================
#                       calculate the AMRF & its error
# =============================================================================
def classIII_prob(triple_limit, par_in, par_in_errors,
                  m1, m1_error, corr_matrix, bit_index=8191, n=1e2, factor=1.0):
    """
    For the given set of orbital parameters by Gaia, this function calculates
    the standard geometrical elements (a, omega, Omega, and i). If the error
    estimates and covariance matrix are prodived, the error estimates on the
    calculated parameters are returned as well.

    Input: thiele_innes: Thiele Innes parameters [A,B,F,G] in milli-arcsec
           thiele_innes_errors : Corresponding errors.
           corr_matrix : Corresponding  4X4 correlation matrix.

  Output: physical and geometrical parameters
    """

    detections = 0
    no_detections = 0
    par_in_errors[np.isnan(par_in_errors)] = 1e-6
    vecs = multivar_sample(par_in, par_in_errors, corr_matrix, int(n))
    key_list = bit_index_map(bit_index)

    for vec in vecs:
        par = {key_list[i]: vec[i] for i in np.arange(len(key_list))}
        par['mass'] = m1_error*np.random.randn() + m1

        # Add the G Thiele-Innes parameter if needed.
        if (bit_index == 8179) | (bit_index == 65435):
            par['G'] = -par['A'] * par['F'] / par['B']

        # This in an intermediate step in the formulae...
        p = (par['A'] ** 2 + par['B'] ** 2 + par['G'] ** 2 + par['F'] ** 2) / 2.
        q = par['A'] * par['G'] - par['B'] * par['F']

        # Calculate the semimajor axis (already in mas)
        a_mas = np.sqrt(p + np.sqrt(p ** 2 - q ** 2))

        # Calculate the AMRF
        AMRF = a_mas / par['parallax'] * par['mass'] ** (-1 / 3) * (par['P']/ 365.25) ** (-2 / 3)

        try:
            if 0 < par['e'] < 1:
                if AMRF > triple_limit(par['mass'])*factor:
                    detections += 1
        except KeyError:
            pass

    return detections/n #(no_detections + detections)



# =============================================================================
#                       calculate the RV motion
# =============================================================================
def RV_motion(par_in, par_in_errors,
              m1, m1_error, corr_matrix, bit_index=8191, n=1e2):
    """
    For the given set of orbital parameters by Gaia, calculates the relevant RV parameters

    Input: thiele_innes: Thiele Innes parameters [A,B,F,G] in milli-arcsec
           thiele_innes_errors : Corresponding errors.
           corr_matrix : Corresponding  4X4 correlation matrix.

  Output: physical and geometrical parameters [K1, e, t0, omega, period]
    """

    detections = 0
    par_in_errors[np.isnan(par_in_errors)] = 1e-6
    vecs = multivar_sample(par_in, par_in_errors, corr_matrix, int(n))
    key_list = bit_index_map(bit_index)

    # Our parameters for the output are:
    # K1, e, t0, omega, period
    vec_out = np.full((int(n),5), np.nan)
    for ind, vec in enumerate(vecs):
        par = {key_list[i]: vec[i] for i in np.arange(len(key_list))}
        par['mass'] = m1_error*np.random.randn() + m1

        # Add the G Thiele-Innes parameter if needed.
        if (bit_index == 8179) | (bit_index == 65435):
            par['G'] = -par['A'] * par['F'] / par['B']

        # This in an intermediate step in the formulae...
        p = (par['A'] ** 2 + par['B'] ** 2 + par['G'] ** 2 + par['F'] ** 2) / 2.
        q = par['A'] * par['G'] - par['B'] * par['F']

        # Calculate the semimajor axis (already in mas)
        a_mas  = np.sqrt(p + np.sqrt(p ** 2 - q ** 2))
        plx    = par['parallax']
        a1     = a_mas/plx    # This is in AU
        mass   = par['mass']  # This is in solar mass
        period = par['P']/ 365.25 # This is in years


        # Calculate the inclination and convert from radians to degrees
        i_rad = np.arccos(q / (a_mas ** 2.))

        # Calculate omega and Omega, then convert from radians to degrees
        omega_rad = 0.5 * (np.arctan2(par['B'] - par['F'], par['A'] + par['G']) + \
                           unp.arctan2(-par['B'] - par['F'], par['A'] - par['G']))

        # Read the eccentricity
        e, t0 = par['e'], par['T']

        # calculate K (lets double check this! :) )
        K1 = 4.74372*(2*np.pi*a1/period)/np.sqrt(1-e**2)*np.sin(i_rad)

        # return an entry in the output
        vec_out[ind, :] = [K1, e, t0, omega_rad, period]

    return  vec_out



def bit_index_map(bit_index):
    if bit_index==8191:
        return ['ra','dec','parallax','pmra','pmdec','A','B','F','G', 'e','P', 'T']
    elif bit_index==8179:
        return ['ra','dec','parallax','pmra','pmdec','A','B','F','P', 'T']
    elif bit_index==65535:
        return ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'A', 'B', 'F', 'G', 'C', 'H', 'gamma','e', 'P', 'T']
    elif bit_index==65435:
        return ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'A', 'B', 'F', 'H', 'gamma', 'P', 'T']
    else:
        return None
