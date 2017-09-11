import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import glob


def get_regression_fun(x, y):
    (slope, intercept, r_value,
     p_value, std_err) = linregress(x, y)
    return (slope * np.unique(x) + intercept,
            slope, r_value, p_value)


def eval_vs_length(fn, scale=True):

    results = np.loadtxt(fn, dtype=str, delimiter='\t')

    results = results[:, 1:].astype(float)
    x = results[:, -1]
    x_u = np.unique(x)
    mse = results[:, 0]
    R2 = results[:, 1]
    r = results[:, 2]

    if scale:
        mse *= np.tanh(x.max() / x) / np.tanh(1)
        R2 *= np.tanh(x / x.max()) / np.tanh(1)
        r *= np.tanh(x / x.max()) / np.tanh(1)

    mse_reg, mse_slope, mse_R2, mse_pvalue = get_regression_fun(x, mse)
    R2_reg, R2_slope, R2_R2, R2_pvalue = get_regression_fun(x, R2)
    r_reg, r_slope, r_R2, r_pvalue = get_regression_fun(x, r)
    print mse_slope, R2_slope
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.scatter(x, mse)
    ax1.plot(x_u, mse_reg, c='k')
    ax1.set_title(
        'MSE (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(mse_pvalue, mse_R2**2))
    ax2.scatter(x, R2)
    ax2.plot(x_u, R2_reg, c='k')
    ax2.set_title(
        '$R^2$ (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(R2_pvalue, R2_R2**2))
    ax3.scatter(x, r)
    ax3.plot(x_u, r_reg, c='k')
    ax3.set_title(
        '$r$ (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(r_pvalue, r_R2**2))
    plt.tight_layout()
    ax3.set_xlabel('Onsets per piece')
    ax2.set_ylabel('$R^2$')
    ax1.set_ylabel('MSE')
    ax3.set_ylabel('$r$')
    plt.show()

    return results

if __name__ == '__main__':

    results_dir = '../results/vel_selection_onsets_diff/'

    all_result_fns = glob.glob(os.path.join(results_dir, 'results_rnn*.txt'))

    results = []
    for fn in all_result_fns:
        res = np.loadtxt(fn, dtype=str, delimiter='\t')[:, 1:].astype(float)

        results.append(res)

    results = np.vstack(results)

    scale = True
    x = results[:, -1]
    x_u = np.unique(x)
    mse = results[:, 0]
    R2 = results[:, 1]
    r = results[:, 2]

    if scale:
        mse *= np.tanh(x.max() / x) / np.tanh(1)
        R2 *= np.tanh(x / x.max()) / np.tanh(1)
        r *= np.tanh(x / x.max()) / np.tanh(1)

    mse_reg, mse_slope, mse_R2, mse_pvalue = get_regression_fun(x, mse)
    R2_reg, R2_slope, R2_R2, R2_pvalue = get_regression_fun(x, R2)
    r_reg, r_slope, r_R2, r_pvalue = get_regression_fun(x, r)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.scatter(x, mse)
    ax1.plot(x_u, mse_reg, c='k')
    ax1.set_title(
        'MSE (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(mse_pvalue, mse_R2**2))
    ax2.scatter(x, R2)
    ax2.plot(x_u, R2_reg, c='k')
    ax2.set_title(
        '$R^2$ (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(R2_pvalue, R2_R2**2))
    ax3.scatter(x, r)
    ax3.plot(x_u, r_reg, c='k')
    ax3.set_title(
        '$r$ (p-value:{0:.3f}, $R^2$:{1:.3f})'.format(r_pvalue, r_R2**2))
    plt.tight_layout()
    ax3.set_xlabel('Onsets per piece')
    ax2.set_ylabel('$R^2$')
    ax1.set_ylabel('MSE')
    ax3.set_ylabel('$r$')
    plt.show()
