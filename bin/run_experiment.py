import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, KFold
import logging
import theano
import json
import lasagne
import copy

import cPickle
import bz2

from cPickle import UnpicklingError

from nn_models.models import RNN, FFNN
from compare_results import mse, corr, R2
from parse_data import load_piece, INPUTS, ONSETWISE_TARGETS, C_INPUTS, M_INPUTS

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

floatX = theano.config.floatX

RANDOM_SEED = 1984


def load_pyc_bz(fn):
    """
    Load a bzipped pickle file

    Parameters
    ----------
    fn : str
        Path to the pickle file.

    Returns
    -------
    object
        Pickled object
    """
    return cPickle.load(bz2.BZ2File(fn, 'r'))


def save_pyc_bz(d, fn):
    """
    Save an object to a bzipped pickle file

    Parameters
    ----------
    d : object
        Object to be pickled
    fn : str
        Path to the pickle file.

    """
    cPickle.dump(d, bz2.BZ2File(fn, 'w'), cPickle.HIGHEST_PROTOCOL)


def get_from_cache_or_compute(cache_fn, func, args=(), kwargs={}, refresh_cache=False):
    """
    If `cache_fn` exists, return the unpickled contents of that file
    (the cache file is treated as a bzipped pickle file). If this
    fails, compute `func`(*`args`), pickle the result to `cache_fn`,
    and return the result.

    Parameters
    ----------

    func : function
        function to compute

    args : tuple
        argument for which to evaluate `func`

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    refresh_cache : boolean
        if True, ignore the cache file, compute function, and store the result
        in the cache file

    Returns
    -------

    object

        the result of `func`(*`args`)
    """

    result = None
    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                result = load_pyc_bz(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(('The file {0} exists, but cannot be unpickled. '
                              'Is it readable? Is this a pickle file?'
                              '').format(cache_fn))
                raise e

    if result is None:
        result = func(*args, **kwargs)
        if cache_fn is not None:
            save_pyc_bz(result, cache_fn)
    return result


def load_data(data_dir, idyom_dir, inputs=None,
              targets=[0], data_type='combined', diff=False,
              interpolate=False,
              c_idyom_model='onsets_vintcc_ltm'):
    """
    Load data files

    Parameters
    ----------
    data_dir : str
        Directory containing the data.
    idyom_dir : str
        Directory containing the expectancy features computed
        with IDyOM.
    inputs : list
        List specifying the indices of the inputs.

    """
    # Get name of the pieces
    pieces = [os.path.basename(p).replace('.txt', '')
              for p in glob.glob(os.path.join(data_dir, '*.txt'))]

    # Load time signatures
    timesigs = np.loadtxt('data/mozart_timesigs.txt', dtype=str)

    # Get idyom model
    idyom_model = list(os.path.split(idyom_dir))
    if '' in idyom_model:
        idyom_model.remove('')
    idyom_model = idyom_model[-1]

    
    if data_type == 'combined':
        features = INPUTS
    elif data_type == 'm':
        features = M_INPUTS
    elif data_type == 'c':
        features = C_INPUTS

    input_names = features
    if inputs is not None:
        input_names = [input_names[i] for i in inputs]

    LOGGER.info('Using {0}'.format(' '.join(input_names)))

    if diff:
        LOGGER.info('Using differential targets')
        targets = [t + 2 for t in targets]
    output_names = [ONSETWISE_TARGETS[i] for i in targets]

    LOGGER.info('Predicting: {0}'.format(' '.join(output_names)))

    feature_idx = np.array([features.index(i) for i in input_names])
    target_idx = np.array([ONSETWISE_TARGETS.index(i) for i in output_names])

    X = []
    Y = []

    for piece in pieces:
        # Load pieces
        piece_dict = load_piece(piece, idyom_model,
                                interpolate=interpolate,
                                timesigs=timesigs,
                                c_idyom_model=c_idyom_model)
        # Get selected features
        x = piece_dict['{0}_data'.format(data_type)][:, feature_idx]
        # Get selected targets
        y = piece_dict['{0}_targets'.format(data_type)][:, target_idx]

        LOGGER.info('Normalizing data')
        y = (y - y.mean(0, keepdims=True)) / (y.std(0, keepdims=True))

        X.append(x.astype(floatX))
        Y.append(y.astype(floatX))

    X = np.array(X)
    Y = np.array(Y)
    pieces = np.array(pieces)

    return dict(X=X,
                Y=Y,
                n_features=len(input_names),
                output_dim=len(output_names),
                input_names=input_names,
                output_names=output_names,
                n_pieces=len(pieces),
                pieces=pieces)


def build_model(model_config, data, out_dir):
    import architectures.model_architectures as architectures
    import nn_models.custom_layers as c_layers

    kwargs = copy.deepcopy(model_config)

    model_arch = kwargs.pop('model')
    build_architecture = getattr(architectures,
                                 model_arch)

    if 'layer_type' in kwargs:
        try:
            kwargs['layer_type'] = getattr(lasagne.layers,
                                           kwargs['layer_type'])
        except:
            LOGGER.info('Using custom {0}'.format(kwargs['layer_type']))
            kwargs['layer_type'] = getattr(c_layers,
                                           kwargs['layer_type'])

    if 'nonlinearity' in kwargs:
        def get_nonlinearity(name):
            try:
                return getattr(lasagne.nonlinearities, name)
            except AttributeError:
                LOGGER.error('unknown nonlinearity "{}"'.format(name))
                raise

        if isinstance(kwargs['nonlinearity'], (list, tuple)):
            kwargs['nonlinearity'] = [
                get_nonlinearity(nonlin_name)
                for nonlin_name in kwargs['nonlinearity']]
        else:
            kwargs['nonlinearity'] = get_nonlinearity(
                kwargs['nonlinearity'])

    kwargs['input_dim'] = data['n_features']
    kwargs['output_dim'] = data['output_dim']

    model_kwargs = build_architecture(**kwargs)
    model_kwargs['input_names'] = data['input_names']
    model_kwargs['output_names'] = data['output_names']
    model_kwargs['out_dir'] = out_dir

    model_type = model_kwargs.pop('model_type')

    if model_type == 'RNN':
        nn = RNN(**model_kwargs)
    elif model_type == 'FFNN':
        nn = FFNN(**model_kwargs)
    return nn


def build_args_train(model, X_train, X_valid,
                     y_train, y_valid,
                     keep_training=False,
                     **kwargs):

    if isinstance(model, (RNN, FFNN)):
        t_kwargs = dict(X_train=X_train,
                        X_valid=X_valid,
                        y_train=y_train,
                        y_valid=y_valid,
                        n_epochs=kwargs['n_epochs'],
                        batch_size=kwargs['batch_size'],
                        max_epochs_from_best=kwargs['max_epochs_from_best'],
                        keep_training=keep_training)

    return t_kwargs


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('Train a model')

    parser.add_argument('data_dir', help='Path to the data')
    parser.add_argument('idyom_dir', help='Path to the IDyOM results')
    parser.add_argument('keyword', help='Keyword')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--kf', help='K folds',
                        default=None)
    parser.add_argument('--inputs',
                        help="index of the inputs (default are all)",
                        nargs='+', type=np.int)
    parser.add_argument("--config", help="Configuration file for the model",
                        default=None)
    parser.add_argument("--data-type", help="m, c or combined",
                        default="combined")
    parser.add_argument('--targets',
                        help="index of the targets (default is 0)",
                        nargs='+', type=np.int,
                        default=[0])
    parser.add_argument('--diff',
                        help='differentiate target',
                        action='store_true', default=False)
    parser.add_argument('--interpolate',
                        help='interpolate expectation curves',
                        action='store_true', default=False)
    parser.add_argument('--c-idyom-model',
                        help='Path to the IDyOM results for chords',
                        default='onsets_vintcc_ltm')

    args = parser.parse_args()

    if args.config:
        model_config = json.load(open(args.config))
    else:
        model_config = dict(
            train=dict(
                    n_epochs=5000,
                    batch_size=100,
                    max_epochs_from_best=100
            ),
                architecture=dict(
                    model='l1lstm_l2d',
                    layer_type='MILSTMLayer',
                    nonlinearity='tanh',
                    n_hidden=5,
                    learning_rate=1e-3,
                    gradient_steps=100,
                    grad_clipping=2,
                    wl2=1e-3,
                    wl1=0,
                    bidirectional=True,
                    skip_connection=True,
        )
        )

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    out_dir = os.path.join(args.out_dir, args.keyword)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

    data = load_data(data_dir=args.data_dir,
                     idyom_dir=args.idyom_dir,
                     inputs=args.inputs,
                     targets=args.targets,
                     data_type=args.data_type,
                     diff=args.diff,
                     interpolate=args.interpolate,
                     c_idyom_model=args.c_idyom_model)

    if args.kf is None:
        n_folds = data['n_pieces']

    else:
        n_folds = int(args.kf)

    experiment_info = dict(
        input_names=list(data['input_names']),
        output_names=list(data['output_names']),
        data_type=args.data_type,
        interpolate=args.interpolate,
        diff=args.diff,
        c_idyom_model=args.c_idyom_model,
        data_dir=args.data_dir,
        m_idyom_dir=args.idyom_dir)

    with open(os.path.join(out_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=4)

    # Divide data for KFold or LOO cross validation
    if n_folds > 1:
        fold_iter = KFold(n_splits=n_folds,
                          random_state=RANDOM_SEED,
                          shuffle=True).split(
            data['X'])

    else:
        # Do not do a cross validation
        fold_iter = iter([[range(data['n_pieces']), []]])

    for ii, (train_val_idx, test_idx) in enumerate(fold_iter):

        LOGGER.info('Fold {0}/{1}'.format(ii + 1, n_folds))
        # Test dataset
        X_test = data['X'][test_idx]
        y_test = data['Y'][test_idx]
        pieces_test = data['pieces'][test_idx]

        # Training and validation sets
        (X_train, X_valid,
         y_train, y_valid,
         pieces_train, pieces_valid) = train_test_split(
             data['X'][train_val_idx], data['Y'][train_val_idx],
             data['pieces'][train_val_idx],
             test_size=0.20, random_state=RANDOM_SEED)

        # Make output dirs for each fold
        out_dir_fold = os.path.join(out_dir, 'fold_{0}'.format(ii))
        if not os.path.exists(out_dir_fold):
            os.mkdir(out_dir_fold)

        np.savetxt(
            os.path.join(out_dir_fold, 'pieces_train.txt'),
            pieces_train, fmt='%s')
        np.savetxt(
            os.path.join(out_dir_fold, 'pieces_valid.txt'),
            pieces_valid, fmt='%s')

        if len(test_idx) > 0:
            np.savetxt(
                os.path.join(out_dir_fold, 'pieces_test.txt'),
                pieces_test, fmt='%s')

        model = build_model(
            model_config['architecture'], data, out_dir_fold)

        if ii == 0:
            # All folds have the same initial params
            init_params = model.get_params()
        else:
            model.set_params(init_params)

        train_args = build_args_train(
            model, X_train, X_valid, y_train, y_valid,
                                      **model_config['train'])

        params_fn = os.path.join(out_dir_fold, 'params.pyc.bz')

        params = get_from_cache_or_compute(
            params_fn,
            model.fit,
            kwargs=train_args)
        model.set_params(params)

        predictions = []
        evaluation = []
        if len(test_idx) > 0:

            for i, (x, y) in enumerate(zip(X_test, y_test)):
                preds = model.predict(x).reshape(-1)
                predictions.append(np.column_stack((y.reshape(-1), preds)))
                np.savetxt(
                    os.path.join(out_dir_fold, 'preds_{0}.txt'.format(i)), predictions[-1])
                evaluation.append(
                    (mse(predictions[-1][:, 1], predictions[-1][:, 0]),
                    R2(predictions[-1][:, 1], predictions[-1][:, 0]),
                    corr(predictions[-1][:, 1], predictions[-1][:, 0])))

                print "MSE {0:.3f}\tR2 {1:.3f}\tr {2:.3f}".format(
                    evaluation[-1][0], evaluation[-1][1], evaluation[-1][2])

            predictions = np.vstack(predictions)
            print "MSE {0:.3f}\tR2 {1:.3f}\tr {2:.3f}".format(
                mse(predictions[:, 1], predictions[:, 0]),
                R2(predictions[:, 1], predictions[:, 0]),
                corr(predictions[:, 1], predictions[:, 0]))
            evaluation = np.array(evaluation)
            mean_results = evaluation.mean(0)
            print mean_results
            np.savetxt(os.path.join(out_dir_fold, 'results.txt'),
                       evaluation,
                       fmt='%.3f',
                       delimiter='\t',
                       header='\t'.join(['MSE {0:.4f}'.format(mean_results[0]),
                                         'R2 {0:.4f}'.format(mean_results[1]),
                                         'r {0:.4f}'.format(mean_results[2])]))
