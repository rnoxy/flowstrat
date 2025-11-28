import numpy as np
import torch


from scipy.stats import norm
from scipy.stats.distributions import chi2


def func_estimate(points):
    r = 0.05
    # sigma = 0.25
    # S0 = 100
    K = 100

    return np.exp(-r) * np.maximum(points - K, 0)


def func_estimate2(points):
    # Recal, we aim at estimating Ef(X)
    # This is function f.
    # For points p1,p2,..,pn it returns [f(p1),...,f(pn)]

    # return np.sqrt(np.sum(points**2, axis=1))
    return np.sum(points**2, axis=1)

    # earlier ver, where points were explicetely 2d
    # return np.log(np.abs(x/y))
    # return (x+2)/(0.01+y**2)

    # return 1.0*(x*y<0.2)

    # wycena opcji, do GBM
    # K=0.2
    # r=0.05
    # return np.exp(-r)*np.maximum(x+y-K,0)


# def european_option_exact_value(S0, ):


def func_estimate_european_option(x):
    K = 100
    r = 0.05
    return np.exp(-r) * np.maximum(x - K, 0)


def inverse_rayleigh(t):
    return np.sqrt(-2 * np.log(1 - t))


def simulate_univariate_strata(m, vecR):
    # store all used randomness in U_all, theta_all (in case one wants later to use the same)
    U_all = []
    theta_all = []

    # vecR = R_1, R_2,,, R_m

    # simulate N(0,1)
    N1_rvs = []
    #    N2_rvs = []

    # info about strata number
    Y_i_strata = []

    for i in np.arange(m):
        ni = vecR[i]

        # randomness
        U1 = np.random.rand(ni)

        # store randomness
        U_all = np.concatenate((U_all, U1))

        # store info about stratas
        Y_i_strata = np.concatenate((Y_i_strata, i * np.ones(ni).astype(int))).astype(
            int
        )

        V = i / m + 1 / m * U1
        # Rr = inverse_rayleigh(V)

        N1_rvs = np.concatenate((N1_rvs, norm.ppf(V)))

    return np.stack((N1_rvs)).T, Y_i_strata, U_all, theta_all


def simulate_multivariate_strata(nr_strata, vecR, dim):
    m = nr_strata

    N_rvs_all = np.zeros((1, dim))  # first row zeros => remove later
    Y_i_strata = []

    for i in np.arange(m):
        ni = vecR[i]
        N_rvs = torch.normal(0, 1, size=(ni, dim))

        N_rvs_normalized = torch.nn.functional.normalize(N_rvs, p=2.0, dim=1)

        U1 = np.random.rand(ni)

        V = i / m + 1 / m * U1
        # chi2 with d=dim degrees of freedeom at V
        Rr = chi2.ppf(V, df=dim)
        Rr = torch.tensor(Rr)
        N_rvs_strata = (
            Rr[:, None] * N_rvs_normalized
        )  # wierze, ze to mnozy cale wiersze
        N_rvs_all = np.concatenate((N_rvs_all, N_rvs_strata))

        Y_i_strata = np.concatenate((Y_i_strata, i * np.ones(ni).astype(int))).astype(
            int
        )

    N_rvs_all = N_rvs_all[1:, :]

    return N_rvs_all, Y_i_strata


def simulate_bivariate_strata(m, vecR, U1_all, U2_all):
    N1_rvs = []
    N2_rvs = []

    # info about strata number
    Y_i_strata = []

    vecR_appended0 = np.append(0, np.cumsum(vecR))

    for i in np.arange(m):
        ni = vecR[i]
        #
        # # randomness
        # U1 = np.random.rand(ni);
        # U2 = np.random.rand(ni);

        U1 = U1_all[vecR_appended0[i] : vecR_appended0[i + 1]]
        U2 = U2_all[vecR_appended0[i] : vecR_appended0[i + 1]]

        theta = U2 * 2 * np.pi

        # store randomness
        # U_all = np.concatenate((U_all, U1))
        #
        # theta_all = np.concatenate((theta_all, theta))

        # store info about stratasf
        Y_i_strata = np.concatenate((Y_i_strata, i * np.ones(ni).astype(int))).astype(
            int
        )

        V = i / m + 1 / m * U1
        Rr = inverse_rayleigh(V)

        N1_rvs = np.concatenate((N1_rvs, Rr * np.sin(theta)))
        N2_rvs = np.concatenate((N2_rvs, Rr * np.cos(theta)))

    return np.stack((N1_rvs, N2_rvs)).T, Y_i_strata, 0, 0


def simulate_bivariate_strata_old(m, vecR):
    # store all used randomness in U_all, theta_all (in case one wants later to use the same)
    U_all = []
    theta_all = []

    # vecR = R_1, R_2,,, R_m

    # simulate N(0,1)
    N1_rvs = []
    N2_rvs = []

    # info about strata number
    Y_i_strata = []

    for i in np.arange(m):
        ni = vecR[i]

        # randomness
        U1 = np.random.rand(ni)
        U2 = np.random.rand(ni)

        theta = U2 * 2 * np.pi

        # store randomness
        U_all = np.concatenate((U_all, U1))

        theta_all = np.concatenate((theta_all, theta))

        # store info about stratas
        Y_i_strata = np.concatenate((Y_i_strata, i * np.ones(ni).astype(int))).astype(
            int
        )

        V = i / m + 1 / m * U1
        Rr = inverse_rayleigh(V)

        N1_rvs = np.concatenate((N1_rvs, Rr * np.sin(theta)))
        N2_rvs = np.concatenate((N2_rvs, Rr * np.cos(theta)))

    return np.stack((N1_rvs, N2_rvs)).T, Y_i_strata, U_all, theta_all


def compute_vecR(weights, R):
    vecR = (weights * R).astype(int)
    vecR[vecR == 0] = 1
    # adjust last strata, so that sum = R_pilot
    vecR[len(vecR) - 1] = R - np.sum(vecR[:-1])
    return vecR


def show_results_conf_intervals(Y_val_text, Y_val, Y_val_var_text, Y_val_var):
    print(Y_val_text, " =  \t ", Y_val)
    print(Y_val_var_text, " =\t ", Y_val_var)
    print(
        "Prob( true value is in [",
        Y_val - np.round(1.96 * np.sqrt(Y_val_var), 5),
        " , ",
        Y_val + np.round(1.96 * np.sqrt(Y_val_var), 5),
        "]) = 0.95",
    )


def stratified_estimate(
    model, N_rvs, Y_opt_i_strata, nr_strata, vecR, weights, obs_mean
):
    samples_stratum = []

    for stratum in np.arange(nr_strata):
        print("Flow: sampling ", vecR[stratum], " points in strata ", stratum, " ...")

        N_rvs_strata = N_rvs[Y_opt_i_strata == stratum]

        z_sampled = torch.tensor(N_rvs_strata)
        z_sampled = z_sampled.type(torch.FloatTensor)

        sample = model(z_sampled, None, reverse=True).cpu().detach().numpy()

        samples_stratum.append(sample)

    STR_stds = np.zeros(nr_strata)
    STR_variances = np.zeros(nr_strata)
    STR_means = np.zeros(nr_strata)

    for stratum in np.arange(nr_strata):
        points_stratum = samples_stratum[stratum]
        # x_str = samples_stratum[stratum][:, 0]
        # y_str = samples_stratum[stratum][:, 1]

        Y_i_str = func_estimate(points_stratum + obs_mean)  # x_str, y_str)

        # Y_i_str = x_str / y_str
        # Y_i_str = x_str / (1+y_str**2)

        STR_stds[stratum] = np.std(Y_i_str)
        STR_means[stratum] = np.mean(Y_i_str)
        STR_variances[stratum] = np.var(Y_i_str, ddof=1)
        # STR_variances[stratum] = np.var(Y_i_str)

    Y_STR = np.sum(weights * STR_means)
    Y_STR_var = np.sum(weights**2 * STR_variances / vecR)

    return Y_STR, Y_STR_var, STR_stds


def stratified_estimate_old(model, N_rvs, Y_opt_i_strata, nr_strata, vecR, weights):
    samples_stratum = []

    for stratum in np.arange(nr_strata):
        print("Flow: sampling ", vecR[stratum], " points in strata ", stratum, " ...")

        N_rvs_strata = N_rvs[Y_opt_i_strata == stratum]

        z_sampled = torch.tensor(N_rvs_strata)
        z_sampled = z_sampled.type(torch.FloatTensor)

        sample = model(z_sampled, None, reverse=True).cpu().detach().numpy()

        samples_stratum.append(sample)

    STR_stds = np.zeros(nr_strata)
    STR_variances = np.zeros(nr_strata)
    STR_means = np.zeros(nr_strata)

    for stratum in np.arange(nr_strata):
        x_str = samples_stratum[stratum][:, 0]
        y_str = samples_stratum[stratum][:, 1]

        Y_i_str = func_estimate(x_str, y_str)

        # Y_i_str = x_str / y_str
        # Y_i_str = x_str / (1+y_str**2)

        STR_stds[stratum] = np.std(Y_i_str)
        STR_means[stratum] = np.mean(Y_i_str)
        STR_variances[stratum] = np.var(Y_i_str, ddof=1)
        # STR_variances[stratum] = np.var(Y_i_str)

    Y_STR = np.sum(weights * STR_means)
    Y_STR_var = np.sum(weights**2 * STR_variances / vecR)

    return Y_STR, Y_STR_var, STR_stds
