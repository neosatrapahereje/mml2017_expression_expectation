import numpy as np
import theano
import theano.tensor as T
import lasagne
import logging
from itertools import cycle
import os
from utils import (delete_if_exists,
                   write_append)
import cPickle
import bz2

from batch_provider import (BatchProvider,
                            ConcatBatchProvider,
                            EMBatchProvider,
                            EMConcatBatchProvider)

from mixture_density_networks import (_get_output_indices,
                                      _get_mixture_parameters)
from utils import ensure_list

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('visualize').setLevel(logging.INFO)
# Set random state for all random initializations in Lasagne
lasagne.random.set_rng(np.random.RandomState(1984))

REGRESSION = 0
BINARY_CLASSIFICATION = 1
CATEGORICAL_CLASSIFICATION = 2
CONVOLUTIONAL_AUTOENCODER = 3


def save_pyc_bz(d, fn):
    # Copied from extra.utils.os_utils
    cPickle.dump(d, bz2.BZ2File(fn, 'w'), cPickle.HIGHEST_PROTOCOL)


class RegressionNeuralNetwork(object):

    """
    Base class for training regression neural networks
    """

    def __init__(self, l_in, l_out,
                 train_loss,
                 valid_loss,
                 target,
                 input_vars=None,
                 updates=None,
                 predictions=None,
                 input_names=None,
                 output_names=None,
                 output_dim=None,
                 n_components=None,
                 input_type=None,
                 out_dir='/tmp',
                 visualize=False,
                 random_state=np.random.RandomState(1984)):

        # Input layer
        self.l_in = l_in
        # Output layer
        self.l_out = l_out
        # Target
        self.target = target
        # Input variables
        self.input_vars = input_vars

        if predictions is None:
            predictions = lasagne.layers.get_output(
                self.l_out, determistic=True)

        # task type (for consistency with old code)
        self.task_type = REGRESSION

        # Set name of the input features
        self.input_names = input_names
        # Set name of the output features
        self.output_names = output_names
        # Set description of the input
        self.input_type = input_type
        # Set path to save output files
        self.out_dir = out_dir

        # Random number generator
        self.prng = random_state

        # Initialize stats
        self.target_mean = None
        self.target_std = None

        # Initialize as model as a FFNN
        self.is_rnn = False

        # For Mixture Densities
        self.n_components = n_components
        self.output_dim = output_dim
        self._setup_graphs(train_loss, valid_loss,
                           predictions, target, updates)

    def _setup_graphs(self, stochastic_loss, deterministic_loss,
                      predictions, target, updates):

        if self.input_vars is None:
            input_vars_loss = [self.l_in.input_var, target]
            input_vars_predict = [self.l_in.input_var]
        else:
            input_vars_loss = ensure_list(self.input_vars) + [target]
            input_vars_predict = ensure_list(self.input_vars)

        if isinstance(updates, (list, tuple)):
            # train_fun is a list for all updates
            self.train_fun = [theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=up) for up in updates]
        else:
            self.train_fun = theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=updates)

        self.valid_fun = theano.function(
            inputs=input_vars_loss,
            outputs=ensure_list(deterministic_loss))

        if any(p is None for p in ensure_list(predictions)):
            preds = []

            for fun in ensure_list(predictions):
                if fun is None:
                    preds.append(lambda x: None)
                else:
                    preds.append(theano.function(
                        inputs=input_vars_predict,
                        outputs=fun))

            self.predict_fun = lambda x: [pr(x) for pr in preds]
        else:
            self.predict_fun = theano.function(
                inputs=input_vars_predict,
                outputs=ensure_list(predictions))

    def fit(self, *args, **kwargs):
        raise NotImplementedError("RegressionNeuralNetwork does not "
                                  "implement the fit method")

    def get_params(self):
        """
        Get the parameters of the network.

        Returns
        -------
        iterable of arrays
            A list with the values of all parameters that specify the
            neural network.
        """
        return lasagne.layers.get_all_param_values(self.l_out)

    def set_params(self, params):
        """
        Set the parameters of the neural network

        Parameters
        ----------
        params : iterable of arrays
            An iterable that contains the values of all parameters that
            specify the neural network. Each parameters must be of the
            same size and dtype as the original parameters.
        """
        return lasagne.layers.set_all_param_values(self.l_out, params)

    def set_stats(self, stats):
        self.target_mean = stats[0]
        self.target_std = stats[1]

    def get_stats(self):
        return [self.target_mean, self.target_std]

    def predict(self, X):
        """
        Compute predictions of the neural network for the given input.

        Parameters
        ----------
        X : array
            Input array for the neural network.

        Returns
        -------
        array
            Predictions of the neural network.
        """
        # Reshape data for RNNs
        do_reshape = self.is_rnn and len(X.shape) < 3
        if do_reshape:
            X = X[np.newaxis]

        predictions = self.predict_fun(X)[0]
        if do_reshape:
            return predictions[0]
        else:
            return predictions

    def get_param_dict(self):
        params = lasagne.layers.get_all_params(self.l_out)
        self.param_dict = dict([(p.name, p) for p in params])

    def set_param_dict(self, param_dict):
        if not hasattr(self, 'param_dict'):
            self.get_param_dict()

        for p in param_dict:
            if p in self.param_dict:
                print 'Using {0}'.format(p)
                self.param_dict[p].set_value(param_dict[p].get_value())
            else:
                print '{0} not in model'.format(p)


class FFNN(RegressionNeuralNetwork):

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs
        self._build_model()

    def _build_model(self):
        super(FFNN, self).__init__(
            *self.args, **self.kwargs)
        self.is_rnn = False

    def __getstate__(self):
        """
        Get current state of the object for pickling.

        Returns
        -------
        dict
            A dictionary containing all parameters to fully reconstruct
            an instance of the class.
        """
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats(),
        )
        return state

    def __setstate__(self, state):
        """
        Set a state for unpickling the object.

        Parameters
        ----------
        state : dict
            A pickled dictionary containing the state to reconstruct the
            :class:`RegressionNeuralNetwork` instance. Valid states are
            generated with `__getstate__`.
        """
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_model()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            n_epochs=100, batch_size=10,
            max_epochs_from_best=10,
            keep_training=False):

        # Make output files and initialize fit parameters
        (train_batch_provider, valid_batch_provider,
         start_epoch, best_loss, best_params,
         total_train_instances,
         train_loss_fn, valid_loss_fn, valid_loss_2_fn,
         validate, best_epoch) = _fit_init(
            self, ConcatBatchProvider, X_train, y_train,
            X_valid, y_valid, keep_training)

        train_batch_provider._concatenate_instances()

        valid_loss = np.array(
            [0 for o in self.valid_fun.outputs])

        try:
            for epoch in xrange(start_epoch, n_epochs):

                train_results = []
                for X_t, y_t in train_batch_provider.iter_pieces_batch(
                        batch_size, shuffle=True, mode='valid'):
                    train_results.append(self.train_fun(X_t, y_t))

                train_loss = np.mean(train_results, axis=0)

                named_train_results = zip([o.variable.name for o in
                                           self.train_fun.outputs],
                                          train_loss)

                write_append(train_loss_fn, epoch, train_loss[0])

                if validate and (
                        np.mod(epoch - start_epoch, 5) == 0 or
                        epoch == n_epochs - 1):
                    valid_results = []
                    for i, (X_v, y_v) in enumerate(
                            valid_batch_provider.iter_pieces()):
                        valid_results.append(self.valid_fun(X_v, y_v))

                    valid_loss = np.mean(valid_results, axis=0)

                    write_append(
                        valid_loss_fn, epoch, valid_loss[0])
                    write_append(
                        valid_loss_2_fn, epoch, valid_loss[-1])

                named_valid_results = zip([o.variable.name for o in
                                           self.valid_fun.outputs],
                                          valid_loss)
                LOGGER.info(
                    ("Epoch: {0}/{3}, "
                     "train: {1}, "
                     "validate: {2} ")
                    .format(epoch,
                            '; '.join(
                                '{0} ={1: .3f}'.format(k, v) for k, v in
                                named_train_results),
                            '; '.join(
                                '{0} ={1: .3f}'.format(k, v) for k, v in
                                named_valid_results),
                            n_epochs))

                params = self.get_params()

                # Early stopping
                if validate:
                    es_loss = valid_loss[0]

                else:
                    es_loss = train_loss[0]

                if es_loss < best_loss:
                    best_params = params
                    best_loss = es_loss
                    best_epoch = epoch
                # Make a backup every 100 epochs (Astrud is sometimes
                # unreliable)
                if np.mod(epoch - start_epoch, 100) == 0:
                    LOGGER.info('Backing parameters up!')
                    save_pyc_bz(best_params,
                                os.path.join(self.out_dir, 'backup_params.pyc.bz'))

                early_stop = (
                    epoch > (best_epoch + max_epochs_from_best))

                if early_stop:
                    break

        except KeyboardInterrupt:
            LOGGER.info('Training interrupted')

        if best_loss < np.inf:
            LOGGER.info(
                'Reloading best parameters (epoch = {0}, {2} loss = {1:.3f})'
                .format(best_epoch + 1, best_loss,
                        'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()


def _fit_init(model, batch_provider,
              X_train, y_train, X_valid,
              y_valid, keep_training, em_train=False):

    start_epoch = 0
    best_epoch = 0
    best_loss = np.inf
    best_params = model.get_params()
    validate = X_valid is not None and y_valid is not None
    epoch_idx = 0

    if em_train:
        start_cycle = 0
        best_cycle = 0
        epoch_idx = 1
        cycle_idx = 0

    train_loss_fn = os.path.join(model.out_dir, 'train_loss.txt')

    if not keep_training:
        delete_if_exists(train_loss_fn)

    if validate:
        valid_loss_fn = os.path.join(model.out_dir, 'valid_loss_reg.txt')
        valid_loss_2_fn = os.path.join(model.out_dir, 'valid_loss.txt')
        if not keep_training:
            delete_if_exists(valid_loss_fn)
            delete_if_exists(valid_loss_2_fn)

        else:
            if os.path.exists(valid_loss_fn):
                valid_loss_old = np.loadtxt(valid_loss_fn)
                best_loss_idx = np.argmin(valid_loss_old[:, -1])
                best_loss = valid_loss_old[best_loss_idx, -1]
                best_epoch = valid_loss_old[best_loss_idx, epoch_idx]
                start_epoch = int(best_epoch) + 1

                if em_train:
                    best_cycle = valid_loss_old[best_loss_idx, cycle_idx]
                    start_cycle = int(best_cycle) + 1

            elif os.path.exists(train_loss_fn):
                train_loss_old = np.loadtxt(train_loss_fn)
                best_loss_idx = np.argmin(train_loss_old[:, -1])
                best_loss = train_loss_old[best_loss_idx, -1]
                best_epoch = train_loss_old[best_loss_idx, epoch_idx]
                start_epoch = int(best_epoch) + 1

                if em_train:
                    best_cycle = train_loss_old[best_loss_idx, cycle_idx]
                    start_cycle = int(best_cycle) + 1

    train_batch_provider = batch_provider(theano.config.floatX)
    train_batch_provider.store_data(X_train, y_train)

    if validate:
        valid_batch_provider = batch_provider(theano.config.floatX)
        valid_batch_provider.store_data(X_valid, y_valid)
    else:
        valid_batch_provider = None

    total_train_instances = np.sum(len(x) for x in X_train)

    if em_train:
        return (train_batch_provider, valid_batch_provider,
                start_epoch, best_loss, best_params,
                total_train_instances,
                train_loss_fn, valid_loss_fn, valid_loss_2_fn,
                validate, best_epoch, start_cycle, best_cycle)
    else:
        return (train_batch_provider, valid_batch_provider,
                start_epoch, best_loss, best_params,
                total_train_instances,
                train_loss_fn, valid_loss_fn, valid_loss_2_fn,
                validate, best_epoch)


class RNN(RegressionNeuralNetwork):

    def __init__(self, *args, **kwargs):

        self.args = args

        if 'gradient_steps' in kwargs:
            self.gradient_steps = kwargs.pop('gradient_steps')

        self.kwargs = kwargs

        self._build_model()

    def _build_model(self):
        super(RNN, self).__init__(
            *self.args, **self.kwargs)
        self.is_rnn = True

    def __getstate__(self):
        """
        Get current state of the object for pickling.

        Returns
        -------
        dict
            A dictionary containing all parameters to fully reconstruct
            an instance of the class.
        """
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats(),
            gradient_steps=self.gradient_steps
        )
        return state

    def __setstate__(self, state):
        """
        Set a state for unpickling the object.

        Parameters
        ----------
        state : dict
            A pickled dictionary containing the state to reconstruct the
            :class:`RegressionNeuralNetwork` instance. Valid states are
            generated with `__getstate__`.
        """
        self.gradient_steps = state['gradient_steps']
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_model()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            n_epochs=100, batch_size=10,
            max_epochs_from_best=10,
            keep_training=False):

        if self.gradient_steps < 1:
            seq_length = 1
        else:
            seq_length = 2 * self.gradient_steps

        start_epoch = 0
        best_epoch = 0
        best_loss = np.inf
        best_params = self.get_params()
        validate = X_valid is not None and y_valid is not None

        train_loss_fn = os.path.join(self.out_dir, 'train_loss.txt')

        if not keep_training:
            delete_if_exists(train_loss_fn)

        if validate:
            valid_loss_fn = os.path.join(self.out_dir, 'valid_loss_reg.txt')
            valid_loss_2_fn = os.path.join(self.out_dir, 'valid_loss.txt')
            if not keep_training:
                delete_if_exists(valid_loss_fn)
                delete_if_exists(valid_loss_2_fn)

            else:
                if os.path.exists(valid_loss_fn):
                    valid_loss_old = np.loadtxt(valid_loss_fn)
                    best_loss_idx = np.argmin(valid_loss_old[:, 1])
                    best_loss = valid_loss_old[best_loss_idx, 1]
                    best_epoch = valid_loss_old[best_loss_idx, 0]
                    start_epoch = int(best_epoch) + 1

                elif os.path.exists(train_loss_fn):
                    train_loss_old = np.loadtxt(train_loss_fn)
                    best_loss_idx = np.argmin(train_loss_old[:, 1])
                    best_loss = train_loss_old[best_loss_idx, 1]
                    best_epoch = train_loss_old[best_loss_idx, 0]
                    start_epoch = int(best_epoch) + 1

        else:
            named_valid_results = ()

        train_batch_provider = BatchProvider(theano.config.floatX)
        train_batch_provider.store_data(X_train, y_train)

        valid_batch_provider = BatchProvider(theano.config.floatX)
        valid_batch_provider.store_data(X_valid, y_valid)

        total_train_instances = np.sum(len(x) for x in X_train)
        n_train_batches_per_epoch = max(
            1, 2 * total_train_instances / (batch_size * seq_length))

        LOGGER.info('Batch size: {}; Batches per epoch: {}'
                    .format(batch_size, n_train_batches_per_epoch))

        # variables to hold data batches; reusing rather than recreating the
        # arrays saves (a little bit of) time
        X_t = train_batch_provider.make_X_batch_array(batch_size, seq_length)
        y_t = train_batch_provider.make_Y_batch_array(batch_size, seq_length)

        if isinstance(self.train_fun, (tuple, list)):
            train_mode_selector = ParameterUpdate(start_epoch, n_epochs)
        else:
            train_mode_selector = SimpleParameterUpdate()

        # Initialize valid loss (in case there is no validation set)
        valid_loss = np.array([0 for o in self.valid_fun.outputs])

        try:
            for epoch in xrange(start_epoch,
                                n_epochs):

                # train_results = []
                mode = train_mode_selector.select_mode(epoch)
                LOGGER.info('Training {0} params'.format(mode))
                train_loss, named_train_results = _train_loss(
                    self, train_batch_provider, batch_size, seq_length,
                    X_t, y_t, train_loss_fn, epoch,
                    n_train_batches_per_epoch, mode=mode)

                if validate and (
                        np.mod(epoch - start_epoch, 5) == 0 or
                        epoch == n_epochs - 1):
                    valid_results = []
                    for i, (X_v, y_v) in enumerate(
                            valid_batch_provider.iter_pieces()):
                        valid_results.append(self.valid_fun(X_v, y_v))

                    valid_loss = np.nanmean(valid_results, axis=0)

                    write_append(
                        valid_loss_fn, epoch, valid_loss[0])
                    write_append(
                        valid_loss_2_fn, epoch, valid_loss[-1])

                named_valid_results = zip([o.variable.name for o in
                                           self.valid_fun.outputs],
                                          valid_loss)
                LOGGER.info(
                    ("Epoch: {0}/{3}, "
                     "train: {1}, "
                     "validate: {2} ")
                    .format(epoch,
                            '; '.join(
                                '{0} ={1: .3f}'.format(k, v) for k, v in
                                named_train_results),
                            '; '.join(
                                '{0} ={1: .3f}'.format(k, v) for k, v in
                                named_valid_results),
                            n_epochs))

                params = self.get_params()

                # Early stopping
                if validate:
                    es_loss = valid_loss[0]

                else:
                    es_loss = train_loss[0]

                if es_loss < best_loss:
                    best_params = params
                    best_loss = es_loss
                    best_epoch = epoch
                # Make a backup every 100 epochs (Astrud is sometimes
                # unreliable)
                if np.mod(epoch - start_epoch, 100) == 0:
                    LOGGER.info('Backing parameters up!')
                    save_pyc_bz(best_params,
                                os.path.join(self.out_dir, 'backup_params.pyc.bz'))

                early_stop = (
                    epoch > (best_epoch + max_epochs_from_best))

                if early_stop:
                    break

        except KeyboardInterrupt:
            print('Training interrupted')

        if best_loss < np.inf:
            print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                  .format(best_epoch + 1, best_loss,
                          'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()


class MDRNN(RNN):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._build_mdn()

    def __getstate__(self):
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats(),
            gradient_steps=self.gradient_steps
        )
        return state

    def __setstate__(self, state):
        self.gradient_steps = state['gradient_steps']
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_mdn()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def _build_mdn(self):
        super(MDRNN, self).__init__(*self.args, **self.kwargs)
        if self.output_dim is not None and self.n_components is not None:
            (self.means_idx, self.cov_diag_idx,
             self.cov_non_diag_idx,
             self.mixtures_idx) = _get_output_indices(
                self.output_dim, self.n_components)


class MDFFNN(FFNN):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._build_mdn()

    def __getstate__(self):
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats()
        )
        return state

    def __setstate__(self, state):

        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_mdn()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def _build_mdn(self):
        super(MDFFNN, self).__init__(*self.args, **self.kwargs)
        if self.output_dim is not None and self.n_components is not None:
            (self.means_idx, self.cov_diag_idx,
             self.cov_non_diag_idx,
             self.mixtures_idx) = _get_output_indices(
                self.output_dim, self.n_components)


class MDRNN_EM(RNN):

    def __init__(self, *args, **kwargs):

        self.args = args

        if 'rho' in kwargs:
            self.rho = kwargs.pop('rho')
        if 'dummy_rho' in kwargs:
            self.dummy_rho = kwargs.pop('dummy_rho')

        self.kwargs = kwargs
        self._build_mdn()

    def __getstate__(self):
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats(),
            gradient_steps=self.gradient_steps,
            rho=self.rho,
            dummy_rho=self.dummy_rho
        )
        return state

    def __setstate__(self, state):
        self.gradient_steps = state['gradient_steps']
        self.rho = state['rho']
        self.dummy_rho = state['dummy_rho']
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_mdn()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def _build_mdn(self):
        super(MDRNN_EM, self).__init__(*self.args, **self.kwargs)
        if self.output_dim is not None and self.n_components is not None:
            (self.means_idx, self.cov_diag_idx,
             self.cov_non_diag_idx,
             self.mixtures_idx) = _get_output_indices(
                self.output_dim, self.n_components)

    def _setup_graphs(self, stochastic_loss, deterministic_loss,
                      predictions, target, updates):

        # Add responsabilities to the input variables for
        # computing the loss
        input_vars_loss = [self.l_in.input_var, target,
                           self.dummy_rho]
        input_vars_predict = [self.l_in.input_var]
        input_vars_rho = [self.l_in.input_var, target]

        if isinstance(updates, (list, tuple)):
            # train_fun is a list for all updates
            self.train_fun = [theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=up) for up in updates]
        else:
            self.train_fun = theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=updates)

        self.valid_fun = theano.function(
            inputs=input_vars_loss,
            outputs=ensure_list(deterministic_loss))

        if any(p is None for p in ensure_list(predictions)):
            preds = []

            for fun in ensure_list(predictions):
                if fun is None:
                    preds.append(lambda x: None)
                else:
                    preds.append(theano.function(
                        inputs=input_vars_predict,
                        outputs=fun))

            self.predict_fun = lambda x: [pr(x) for pr in preds]
        else:
            self.predict_fun = theano.function(
                inputs=input_vars_predict,
                outputs=ensure_list(predictions))

        # compile function for evaluating the responsabilities
        # warn on_unused_input in case of a single component
        self.rho_fun = theano.function(input_vars_rho, self.rho,
                                       on_unused_input='warn')

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            n_epochs=100, em_cycles=10,
            batch_size=10,
            max_epochs_from_best=10,
            keep_training=False):

        if self.gradient_steps < 1:
            seq_length = 1
        else:
            seq_length = 2 * self.gradient_steps

        # Make output files and initialize fit parameters
        (train_batch_provider, valid_batch_provider,
         start_epoch, best_loss, best_params,
         total_train_instances,
         train_loss_fn, valid_loss_fn, valid_loss_2_fn,
         validate, best_epoch,
         start_cycle, best_cycle) = _fit_init(
            self, EMBatchProvider, X_train, y_train,
             X_valid, y_valid, keep_training, em_train=True)

        # initialize
        train_batch_provider.rho_fun = self.rho_fun

        train_batch_provider.update_rho()
        if valid_batch_provider is not None:
            valid_batch_provider.rho_fun = self.rho_fun
            valid_batch_provider.update_rho()

        n_train_batches_per_epoch = max(
            1, 2 * total_train_instances / (batch_size * seq_length))

        LOGGER.info('Batch size: {}; Batches per epoch: {}'
                    .format(batch_size, n_train_batches_per_epoch))

        # variables to hold data batches; reusing rather than recreating the
        # arrays saves (a little bit of) time
        X_t = train_batch_provider.make_X_batch_array(
            batch_size, seq_length)
        y_t = train_batch_provider.make_Y_batch_array(
            batch_size, seq_length)
        rho_t = train_batch_provider.make_rho_batch_array(
            batch_size, seq_length)

        if isinstance(self.train_fun, (tuple, list)):
            train_mode_selector = ParameterUpdate(start_epoch, n_epochs)
        else:
            train_mode_selector = SimpleParameterUpdate()

        # Initialize valid loss (in case there is no validation set)
        valid_loss = np.array([0 for o in self.valid_fun.outputs])

        try:
            for e_cycle in xrange(start_cycle, em_cycles):

                # Update responsabilities each cycle
                train_batch_provider.update_rho()

                if validate:
                    valid_batch_provider.update_rho()

                for epoch in xrange(start_epoch,
                                    n_epochs):

                    # train_results = []
                    mode = train_mode_selector.select_mode(epoch)
                    LOGGER.info('Training {0} params'.format(mode))
                    train_loss, named_train_results = _train_loss(
                        self, train_batch_provider, batch_size,
                        seq_length,
                        X_t, y_t, train_loss_fn, epoch,
                        n_train_batches_per_epoch, mode=mode,
                        rho_t=rho_t)

                    if validate and (
                            np.mod(epoch - start_epoch, 5) == 0 or
                            epoch == n_epochs - 1):
                        valid_results = []
                        for i, (X_v, y_v, rho_v) in enumerate(
                                valid_batch_provider.iter_pieces()):
                            valid_results.append(
                                self.valid_fun(X_v, y_v, rho_v))

                        valid_loss = np.nanmean(valid_results, axis=0)

                        write_append(
                            valid_loss_fn, [e_cycle, epoch], valid_loss[0])
                        write_append(
                            valid_loss_2_fn, [e_cycle, epoch], valid_loss[-1])

                    named_valid_results = zip([o.variable.name for o in
                                               self.valid_fun.outputs],
                                              valid_loss)
                    LOGGER.info(
                        ("Cycle: {4}/{5}, "
                         "Epoch: {0}/{3}, "
                         "train: {1}, "
                         "validate: {2} ")
                        .format(epoch,
                                '; '.join(
                                    '{0} ={1: .3f}'.format(k, v) for k, v in
                                    named_train_results),
                                '; '.join(
                                    '{0} ={1: .3f}'.format(k, v) for k, v in
                                    named_valid_results),
                                n_epochs,
                                e_cycle, em_cycles))

                    params = self.get_params()

                    # Early stopping
                    if validate:
                        es_loss = valid_loss[0]

                    else:
                        es_loss = train_loss[0]

                    if es_loss < best_loss:
                        best_params = params
                        best_loss = es_loss
                        best_epoch = epoch
                    # Make a backup every 100 epochs (Astrud is sometimes
                    # unreliable)
                    if np.mod(epoch - start_epoch, 100) == 0:
                        save_pyc_bz(best_params,
                                    os.path.join(self.out_dir, 'backup_params.pyc.bz'))

                    early_stop = (
                        epoch > (best_epoch + max_epochs_from_best))

                    if early_stop:
                        break

                # Set best barams for every cycle
                if best_loss < np.inf:
                    print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                          .format(best_epoch + 1, best_loss,
                                  'validation' if validate else 'training'))
                    self.set_params(best_params)

                # Reinitialize start_epoch and best_loss after every cycle
                start_epoch = 0
                best_loss = np.inf
                best_epoch = 0

        except KeyboardInterrupt:
            print('Training interrupted')

        if best_loss < np.inf:
            print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                  .format(best_epoch + 1, best_loss,
                          'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()


class MDFFNN_EM(FFNN):

    def __init__(self, *args, **kwargs):

        self.args = args

        if 'rho' in kwargs:
            self.rho = kwargs.pop('rho')
        if 'dummy_rho' in kwargs:
            self.dummy_rho = kwargs.pop('dummy_rho')

        self.kwargs = kwargs
        self._build_mdn()

    def __getstate__(self):
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            stats=self.get_stats(),
            rho=self.rho,
            dummy_rho=self.dummy_rho
        )
        return state

    def __setstate__(self, state):
        self.rho = state['rho']
        self.dummy_rho = state['dummy_rho']
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_mdn()
        self.set_params(state['params'])
        self.set_stats(state['stats'])

    def _build_mdn(self):
        super(MDFFNN_EM, self).__init__(*self.args, **self.kwargs)
        if self.output_dim is not None and self.n_components is not None:
            (self.means_idx, self.cov_diag_idx,
             self.cov_non_diag_idx,
             self.mixtures_idx) = _get_output_indices(
                self.output_dim, self.n_components)

    def _setup_graphs(self, stochastic_loss, deterministic_loss,
                      predictions, target, updates):

        # Add responsabilities to the input variables for
        # computing the loss
        input_vars_loss = [self.l_in.input_var, target,
                           self.dummy_rho]
        input_vars_predict = [self.l_in.input_var]
        input_vars_rho = [self.l_in.input_var, target]

        if isinstance(updates, (list, tuple)):
            # train_fun is a list for all updates
            self.train_fun = [theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=up) for up in updates]
        else:
            self.train_fun = theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=updates)

        self.valid_fun = theano.function(
            inputs=input_vars_loss,
            outputs=ensure_list(deterministic_loss))

        if any(p is None for p in ensure_list(predictions)):
            preds = []

            for fun in ensure_list(predictions):
                if fun is None:
                    preds.append(lambda x: None)
                else:
                    preds.append(theano.function(
                        inputs=input_vars_predict,
                        outputs=fun))

            self.predict_fun = lambda x: [pr(x) for pr in preds]
        else:
            self.predict_fun = theano.function(
                inputs=input_vars_predict,
                outputs=ensure_list(predictions))

        # compile function for evaluating the responsabilities
        # warn on_unused_input in case of a single component
        self.rho_fun = theano.function(input_vars_rho, self.rho,
                                       on_unused_input='warn')

    def fit(self, X_train, y_train,
            X_valid=None, y_valid=None,
            n_epochs=100, em_cycles=10,
            batch_size=10,
            max_epochs_from_best=10,
            keep_training=False):

        # Make output files and initialize fit parameters
        (train_batch_provider, valid_batch_provider,
         start_epoch, best_loss, best_params,
         total_train_instances,
         train_loss_fn, valid_loss_fn, valid_loss_2_fn,
         validate, best_epoch,
         start_cycle, best_cycle) = _fit_init(
            self, EMConcatBatchProvider, X_train, y_train,
             X_valid, y_valid, keep_training, em_train=True)

        # initialize
        train_batch_provider.rho_fun = self.rho_fun

        if valid_batch_provider is not None:
            valid_batch_provider.rho_fun = self.rho_fun

        train_batch_provider._concatenate_instances()

        valid_loss = np.array(
            [0 for o in self.valid_fun.outputs])

        try:
            for e_cycle in xrange(start_cycle, em_cycles):

                # Update responsabilities each cycle
                train_batch_provider.update_rho()

                if validate:
                    valid_batch_provider.update_rho()

                for epoch in xrange(start_epoch,
                                    n_epochs):

                    train_results = []
                    for X_t, y_t, rho_t in train_batch_provider.iter_pieces_batch(
                            batch_size, shuffle=True, mode='valid'):
                        train_results.append(self.train_fun(X_t, y_t, rho_t))

                    train_loss = np.mean(train_results, axis=0)

                    named_train_results = zip([o.variable.name for o in
                                               self.train_fun.outputs],
                                              train_loss)

                    write_append(train_loss_fn, epoch, train_loss[0])

                    if validate and (
                            np.mod(epoch - start_epoch, 5) == 0 or
                            epoch == n_epochs - 1):
                        valid_results = []
                        for i, (X_v, y_v, rho_v) in enumerate(
                                valid_batch_provider.iter_pieces()):
                            valid_results.append(
                                self.valid_fun(X_v, y_v, rho_v))

                        valid_loss = np.nanmean(valid_results, axis=0)

                        write_append(
                            valid_loss_fn, [e_cycle, epoch], valid_loss[0])
                        write_append(
                            valid_loss_2_fn, [e_cycle, epoch], valid_loss[-1])

                    named_valid_results = zip([o.variable.name for o in
                                               self.valid_fun.outputs],
                                              valid_loss)
                    LOGGER.info(
                        ("Cycle: {4}/{5}, "
                         "Epoch: {0}/{3}, "
                         "train: {1}, "
                         "validate: {2} ")
                        .format(epoch,
                                '; '.join(
                                    '{0} ={1: .3f}'.format(k, v) for k, v in
                                    named_train_results),
                                '; '.join(
                                    '{0} ={1: .3f}'.format(k, v) for k, v in
                                    named_valid_results),
                                n_epochs,
                                e_cycle, em_cycles))

                    params = self.get_params()

                    # Early stopping
                    if validate:
                        es_loss = valid_loss[0]

                    else:
                        es_loss = train_loss[0]

                    if es_loss < best_loss:
                        best_params = params
                        best_loss = es_loss
                        best_epoch = epoch

                    early_stop = (
                        epoch > (best_epoch + max_epochs_from_best))

                    if early_stop:
                        break

                # Set best barams for every cycle
                if best_loss < np.inf:
                    print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                          .format(best_epoch + 1, best_loss,
                                  'validation' if validate else 'training'))
                    self.set_params(best_params)

                # Reinitialize start_epoch and best_loss after every cycle
                start_epoch = 0
                best_loss = np.inf
                best_epoch = 0

        except KeyboardInterrupt:
            print('Training interrupted')

        if best_loss < np.inf:
            print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                  .format(best_epoch + 1, best_loss,
                          'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()


class MDN_aggregator(object):

    def __init__(self, MDN_model, aggregation="mean"):
        self.m = MDN_model

        if hasattr(MDN_model, 'aggregation'):
            self.aggregation = MDN_model.aggregation
        else:
            self.aggregation = aggregation

    @property
    def input_names(self):
        return self.m.input_names

    @property
    def output_names(self):
        return self.m.output_names

    @property
    def input_type(self):
        return self.m.input_type

    @property
    def is_rnn(self):
        return self.m.is_rnn

    @property
    def n_components(self):
        return self.m.n_components

    @property
    def output_dim(self):
        return self.m.output_dim

    @property
    def target_mean(self):
        return self.m.target_mean

    @property
    def target_std(self):
        return self.m.target_std

    def _get_aggregations(self, predictions):

        assert self.aggregation in ('mean', 'max_component')
        ndim = predictions.ndim

        # (mixtures, means,
        #  covariances_diagonal,
        #  covariances_nondiagonal) = self._get_mixture_parameters(predictions)
        mixture_parameters = self._get_mixture_parameters(predictions)
        mixtures = mixture_parameters['mixtures']
        means = mixture_parameters['means']
        if mixtures is None:
            result = means[0]
        else:
            if self.aggregation == 'mean':
                if ndim == 2:
                    result = sum([mixtures[:, [i]] * means[i]
                                  for i in range(self.n_components)])
                elif ndim == 3:
                    result = sum([mixtures[:, :, [i]] * means[i]
                                  for i in range(self.n_components)])

            if self.aggregation == 'max_component':
                idx = mixtures.argmax(-1)
                result = []
                for ts, pr in enumerate(idx):
                    if ndim == 2:
                        result.append(means[pr][ts, :])
                    elif ndim == 3:
                        piece = []
                        for t, ix in enumerate(pr):
                            piece.append(means[ix][ts, t, :])
                        result.append(np.vstack(piece))
                if ndim == 2:
                    result = np.vstack(result)
                else:
                    result = np.array(result)
        return result

    def predict(self, X):
        """
        Compute predictions of the neural network for the given input.

        Parameters
        ----------
        X : array
            Input array for the neural network.

        Returns
        -------
        array
            Predictions of the neural network.
        """
        if X[0].ndim == 2:
            predictions = [self.m.predict(x) for x in X]
            results = [self._get_aggregations(x)
                       for x in predictions]
            results = np.array(results)
        elif X[0].ndim == 1:
            predictions = self.m.predict(X)
            results = self._get_aggregations(predictions)
        return results

    def _get_mixture_parameters(self, predictions, return_cov_matrices=False,
                                return_precision_matrices=False,
                                return_cholesky_matrices=False):
        """Legacy convenience method for getting the parameters of a GMM
        """
        return _get_mixture_parameters(
            predictions=predictions,
            n_components=self.n_components,
            means_idx=self.m.means_idx,
            cov_diag_idx=self.m.cov_diag_idx,
            cov_non_diag_idx=self.m.cov_non_diag_idx,
            mixtures_idx=self.m.mixtures_idx,
            return_cov_matrices=return_cov_matrices,
            return_precision_matrices=return_precision_matrices,
            return_cholesky_matrices=return_cholesky_matrices)

    def get_mixture_parameters(self, X, return_precision_matrices=False,
                               return_cholesky_matrices=False):
        """Get the parameters of the GMM given an input
        """

        if X[0].ndim == 2:

            predictions = [self.m.predict(x) for x in X]
            mps = [
                self._get_mixture_parameters(
                    x, return_cov_matrices=True,
                    return_precision_matrices=return_precision_matrices,
                    return_cholesky_matrices=return_cholesky_matrices)
                for x in predictions]

            mixtures = []
            means = []
            covariance_matrices = []
            precision_matrices = []
            cholesky_matrices = []

            for ps in mps:
                mixtures.append(ps['mixtures'])
                means.append(ps['means'])
                covariance_matrices.append(ps['covariance_matices'])
                precision_matrices.append(ps['precision_matrices'])
                cholesky_matrices.append(ps['cholesky_matrices'])
        elif X[0].ndim == 1:
            predictions = self.m.predict(X)
            mps = self._get_mixture_parameters(
                predictions,
                return_cov_matrices=True,
                return_precision_matrices=return_precision_matrices,
                return_cholesky_matrices=return_cholesky_matrices)
            mixtures = mps['mixtures']
            means = mps['means']
            covariance_matrices = mps['covariance_matrices']
            precision_matrices = mps['precision_matrices']
            cholesky_matrices = mps['cholesky_matrices']

        # TODO: This way of returning the outputs seems unnecesarily convoluted
        # Perhaps it would be better to change it everywere as a dict?
        return dict(
            mixtures=mixtures,
            means=means,
            covariance_matrices=covariance_matrices,
            precision_matrices=precision_matrices,
            cholesky_matrices=cholesky_matrices)


def _train_loss(model, batch_provider, batch_size, seq_length,
                X_t, y_t, train_loss_fn, epoch,
                batches_per_epoch, mode='valid', rho_t=None,
                e_cycle=None):

    train_results = []

    if rho_t is None:
        inputs = [X_t, y_t]
    else:
        inputs = [X_t, y_t, rho_t]

    # Select training function
    if mode == 'valid':
        get_batch = batch_provider.get_batch_valid
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[0]
        else:
            train_fun = model.train_fun
    elif mode == 'init':
        get_batch = batch_provider.get_batch_start
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[1]
        else:
            train_fun = model.train_fun
    elif mode == 'end':
        get_batch = batch_provider.get_batch_end
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[2]
        else:
            train_fun = model.train_fun

    elif mode == 'full':
        get_batch = batch_provider.get_batch_full
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[3]
        else:
            train_fun = model.train_fun

    for i in range(batches_per_epoch):

        get_batch(*([batch_size, seq_length] + inputs))
        train_results.append(train_fun(*inputs))

    train_loss = np.mean(train_results, axis=0)
    if any(np.isnan(train_loss)):
        LOGGER.debug('Warning! NaN in loss. '
                     'The following results cannot be trusted!')
        # Replace NaNs with an absurdly large number (to know that
        # something went horribly wrong)
        train_loss[np.where(np.isnan(train_loss))] = np.finfo(np.float32).max

    if e_cycle is None:
        write_append(train_loss_fn, epoch, train_loss[0])
    else:
        write_append(train_loss_fn, [e_cycle, epoch], train_loss[0])
    named_train_results = zip([o.variable.name for o in
                               train_fun.outputs],
                              train_loss)

    return train_loss, named_train_results


class SimpleParameterUpdate(object):

    def select_mode(self, epoch, crit=None):
        return 'valid'


class ParameterUpdate(object):

    def __init__(self, start_epoch, n_epochs):

        self.start_epoch = start_epoch
        self.n_epochs = n_epochs
        self.modes = cycle(['valid', 'init', 'end'])
        self.crit = 200
        self.crits = cycle([200, 25, 25])

    def select_mode(self, epoch, crit=None):

        # epochs = n_epochs - start_epoch
        if crit is None:
            crit = self.crit
        if np.mod(epoch - self.start_epoch, crit) == 0:
            self.mode = next(self.modes)
            self.crit = next(self.crits)

        return self.mode
