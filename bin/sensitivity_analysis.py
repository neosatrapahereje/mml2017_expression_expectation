import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import argparse
import os

from run_experiment import load_pyc_bz, load_data
from architectures import model_architectures
import copy
import logging

import nn_models.custom_layers as c_layers
from nn_models.utils import sliding_window
from lasagne import layers
from parse_data import (ONSETWISE_TARGETS, INPUTS, PRETTY_INPUT_NAMES,
                        SCORE_FEATURES, EXPECTANCY_FEATURES)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
floatX = theano.config.floatX

ORDERED_FEATURES = SCORE_FEATURES + EXPECTANCY_FEATURES


def get_nonlinearity(name):
    try:
        return getattr(lasagne.nonlinearities, name)
    except AttributeError:
        LOGGER.error('unknown nonlinearity "{}"'.format(name))
        raise


def test():

    input_dim = 5
    n_hidden = 4
    output_dim = 1
    gradient_steps = 10

    l_in = lasagne.layers.InputLayer((None, None, input_dim))
    b_size, seqlen, _ = l_in.input_var.shape
    l_r_f = lasagne.layers.RecurrentLayer(l_in, n_hidden,
                                          nonlinearity=None, b=None)

    l_r_f.W_in_to_hid.set_value(
        2 * np.ones_like(l_r_f.W_in_to_hid.get_value()))
    l_r_f.W_hid_to_hid.set_value(
        0.8 * np.eye(n_hidden).astype(np.float32))
    l_r_b = lasagne.layers.RecurrentLayer(l_in, n_hidden, backwards=True,
                                          nonlinearity=None, b=None)
    l_r_b.W_in_to_hid.set_value(
        2 * np.ones_like(l_r_b.W_in_to_hid.get_value()))
    l_r_b.W_hid_to_hid.set_value(
        0.8 * np.eye(n_hidden).astype(np.float32))

    l_r = lasagne.layers.ElemwiseSumLayer([l_r_f, l_r_b])

    l_re = lasagne.layers.ReshapeLayer(l_r, (-1, n_hidden))
    l_d = lasagne.layers.DenseLayer(l_re, output_dim,
                                    nonlinearity=None)
    l_out = lasagne.layers.ReshapeLayer(l_d, (b_size, seqlen, output_dim))

    y = lasagne.layers.get_output(l_out)

    y_slice = y[:, gradient_steps - 1, :]

    grad = T.grad(T.mean(y_slice), wrt=l_in.input_var)
    y_fun = theano.function([l_in.input_var], [y, y_slice])
    grad_fun = theano.function([l_in.input_var], grad)

    X = np.random.rand(15, 2 * gradient_steps, input_dim).astype(np.float32)

    Y, Ys = y_fun(X)

    grads = np.array([grad_fun(x[np.newaxis])[0] for x in X])

    plt.matshow(np.mean(grads, 0).T)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute sensitivity analysis')

    parser.add_argument('config', help='Config file')
    parser.add_argument('info', help='Info file')
    parser.add_argument('params', help='Params')
    parser.add_argument('pieces')
    parser.add_argument('--outdir', default='/tmp')
    parser.add_argument('--n-frames', type=int, default=500)

    args = parser.parse_args()

    info = json.load(open(args.info))
    config = json.load(open(args.config))

    inputs = np.array([INPUTS.index(i) for i in info['input_names']])
    targets = np.array([ONSETWISE_TARGETS.index(i)
                        for i in info['output_names']])

    data = load_data(info['data_dir'], info['m_idyom_dir'],
                     inputs=inputs,
                     targets=targets)

    params = load_pyc_bz(args.params)

    model_config = config['architecture']

    kwargs = copy.deepcopy(model_config)

    model_arch = kwargs.pop('model')
    gradient_steps = kwargs.pop('gradient_steps')

    build_architecture = getattr(model_architectures, model_arch)

    if 'layer_type' in kwargs:
        try:
            kwargs['layer_type'] = getattr(layers,
                                           kwargs['layer_type'])
        except:
            LOGGER.info('Using custom {0}'.format(kwargs['layer_type']))
            kwargs['layer_type'] = getattr(c_layers,
                                           kwargs['layer_type'])

    for k in kwargs.keys():
        if 'nonlinearity' in k:
            print 'Getting {0}'.format(k)
            if isinstance(kwargs[k], (list, tuple)):
                kwargs[k] = [
                    get_nonlinearity(nonlin_name)
                    for nonlin_name in kwargs[k]]
            else:
                kwargs[k] = get_nonlinearity(kwargs[k])
        if 'objective' in k:
            print 'Getting {0}'.format(k)
            kwargs[k] = getattr(lasagne.objectives, kwargs[k])

    kwargs['input_dim'] = data['n_features']
    kwargs['output_dim'] = data['output_dim']

    rnn = build_architecture(**kwargs)
    lasagne.layers.set_all_param_values(rnn['l_out'], params)

    y_slice = rnn['predictions'][:, gradient_steps - 1, :]

    n_output = T.sum(y_slice)

    y_slice_fun = theano.function([rnn['l_in'].input_var], y_slice)
    grad = T.grad(n_output, wrt=rnn['l_in'].input_var)
    n_fun = theano.function([rnn['l_in'].input_var], n_output)
    grad_fun = theano.function([rnn['l_in'].input_var], grad)

    pieces_test = np.loadtxt(args.pieces, dtype=str)
    piece_idx = np.array([list(data['pieces']).index(p) for p in pieces_test])

    X_s = []
    for X in data['X'][piece_idx]:
        x_s = sliding_window(
            X, ws=(2 * gradient_steps, X.shape[-1]), ss=(1, 1))
        X_s.append(x_s)

    X_s = np.vstack(X_s)

    idx = np.arange(len(X_s))

    np.random.shuffle(idx)

    sensitivity = np.array([grad_fun(x[np.newaxis])[0]
                            for x in X_s[idx[:args.n_frames]]])

    
    out_results = dict(
        inputs=X_s[idx[:args.n_frames]],
        sensitivity=sensitivity)
    np.savez_compressed(
        out_results_fn, **out_results)

    sensitivity = sensitivity.mean(0)

    yticklabels = []
    y_idxs = []

    for i in ORDERED_FEATURES:

        if i in data['input_names']:
            y_idxs.append(int(np.where(data['input_names'] == i)))
            yticklabels.append(PRETTY_INPUT_NAMES[i])

    y_idxs = np.array(y_idxs)

    # Reorder data
    sensitivity = sensitivity[:, y_idxs]

    vmax = abs(sensitivity).max()
    vmin = - vmax

    fig, ax = plt.subplots()
    ax.matshow(sensitivity.T,
               interpolation='nearest',
               origin='lower', aspect='auto',
               cmap='bwr',
               vmax=vmax, vmin=vmin)
    # cbar = fig.colorbar(ax)
    ax.set_yticks(range(len(data['input_names'])))
    ax.set_yticklabels(data['input_names'])
    ax.set_xticks(range(2 * gradient_steps))
    ax.set_xticklabels(
        ['t - {0}'.format((gradient_steps - 1) - i) for i in range(gradient_steps - 1)] +
        ['t'] +
        ['t + {0}'.format(i + 1) for i in range(gradient_steps)], rotation=90)
    ax.set_xlim([(gradient_steps - 1) - 10, (gradient_steps - 1) + 10])
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'sensitivity.pdf'))
    plt.clf()
    plt.close()
