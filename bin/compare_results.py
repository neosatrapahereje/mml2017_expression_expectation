import numpy as np
import os
import glob
import re
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def flatten(l):
    return [item for sublist in l for item in sublist]


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def corr(predictions, targets):
    """
    Pearsons correlation
    """
    return np.corrcoef(predictions.flatten(), targets.flatten())[1, 0]


def R2(predictions, targets):
    """
    Coefficient of determination
    """
    SS_error = np.sum((predictions - targets) ** 2, 0)
    mean_y = np.mean(targets, 0)
    SS_y = np.sum((targets - mean_y) ** 2, 0)
    return 1 - SS_error / SS_y

pred_fn_pttrn = re.compile(r'preds_([0-9]*).txt')


def main():

    import argparse

    parser = argparse.ArgumentParser('Join and compare results')

    parser.add_argument('results_dir')

    args = parser.parse_args()

    results_dir = args.results_dir

    out_file = os.path.join(results_dir, 'results.txt')

    experiment_dirs = []

    for d in glob.glob(os.path.join(results_dir, '*')):
        if os.path.isdir(d):
            experiment_dirs.append(d)

    exp_eval = []

    evaluations = []
    exp_names = []
    piece_lengths = []
    for e_dir in experiment_dirs:

        evaluation = []
        e_name = os.path.basename(e_dir)

        result_fns = glob.glob(os.path.join(e_dir, 'fold*', 'preds*'))
        p_names = []
        for i, fn in enumerate(result_fns):

            f_dir = os.path.dirname(fn)
            t_pieces = np.loadtxt(
                os.path.join(f_dir, 'pieces_test.txt'),
                dtype=str).reshape(-1)

            p_idx = int(pred_fn_pttrn.search(fn).group(1))

            p_name = t_pieces[p_idx]

            res = np.loadtxt(fn)
            t = res[:, 0]
            y = res[:, 1]
            p_length = len(res)
            piece_lengths.append(p_length)

            evaluation.append(
                (mse(y, t), R2(y, t), corr(y, t)))
            p_names.append(p_name)

        evaluation = np.array(evaluation)

        evaluations.append(evaluation)
        p_names = np.array(p_names)
        # p_names = p_names[p_names.argsort()]
        # evaluation = evaluation[p_names.argsort()]
        with open(out_file.replace('.txt', '_' + e_name + '.txt'), 'w') as f:
            for p, e, p_l in zip(p_names, evaluation, piece_lengths):
                e_str = '\t'.join(
                    ['{0:.4f}'.format(i) for i in e] + ['{0}'.format(p_l)])
                out_str = '{0}\t{1}\n'.format(p, e_str)
                f.write(out_str)

        exp_eval.append((e_name, evaluation.mean(0)))
        exp_names.append(e_name)

    with open(out_file, 'w') as f:

        out_str = 'Experiment\tMSE\tR2\tr\n'
        for e in exp_eval:
            e_str = '\t'.join(['{0:.3f}'.format(i) for i in e[1]])
            out_str += '{0}\t{1}\n'.format(e[0], e_str)

        f.write(out_str)

    print out_str

    # significance test
    s_out_string = 'metric\tstatistic\tpvalue\n'
    tukey_groups = np.array([len(evaluation) * [e]
                             for e in exp_names]).flatten()

    f_mse = ['MSE', stats.f_oneway(*[e[:, 0] for e in evaluations])]
    f_R2 = ['R2', stats.f_oneway(*[e[:, 1] for e in evaluations])]
    f_r = ['r', stats.f_oneway(*[e[:, 2] for e in evaluations])]

    for ft in [f_mse, f_R2, f_r]:

        s_out_string += ('\t'.join([ft[0], '{0:.2f}'.format(ft[1].statistic),
                                    '{0:.2e}'.format(ft[1].pvalue)]) + '\n')

    s_out_string += '\n\nMSE\n---------\n'

    tukey_mse = pairwise_tukeyhsd(
        endog=np.array([e[:, 0] for e in evaluations]).flatten(),     # Data
        groups=tukey_groups,   # Groups
        alpha=0.01)          # Significance level

    s_out_string += tukey_mse.summary().as_text()

    s_out_string += '\n\nR2\n---------\n'

    tukey_mse = pairwise_tukeyhsd(
        endog=np.array([e[:, 1] for e in evaluations]).flatten(),     # Data
        groups=tukey_groups,   # Groups
        alpha=0.05)          # Significance level

    s_out_string += tukey_mse.summary().as_text()

    s_out_string += '\n\nr\n---------\n'

    tukey_mse = pairwise_tukeyhsd(
        endog=np.array([e[:, 2] for e in evaluations]).flatten(),     # Data
        groups=tukey_groups,   # Groups
        alpha=0.05)          # Significance level

    s_out_string += tukey_mse.summary().as_text()

    print s_out_string

    with open(out_file.replace('.txt', 'sig.txt'), 'w') as f:
        f.write(s_out_string)


if __name__ == '__main__':

    main()
    # experiment_fn = '../results/compare_viewpoints'

    # results_fn = glob.glob(os.path.join(experiment_fn, '*', 'results.txt'))

    # results = []
    # names = []
    # for fn in results_fn:
    #     e_name = os.path.basename(os.path.dirname(fn))
    #     res = np.loadtxt(fn, skiprows=1, dtype=str)

    #     results.append(res)
    #     names.append(e_name)
    # results = np.array(results)
    # results = results[:, 1:].astype(float)
    # names = np.array(names)

    # sort_idx = results[:, 0].argsort()
    # results = results[sort_idx]
    # names = names[sort_idx]

    # out_str = 'Experiment\tMSE\tR2\tr\n'
    # for n, r in zip(names, results):

    #     e_str = '\t'.join(['{0:.4f}'.format(i) for i in r])
    #     out_str += '{0}\t{1}\n'.format(n, e_str)

    # with open(os.path.join(experiment_fn, 'results.txt'), 'w') as f:
    #     f.write(out_str)

    # print out_str
