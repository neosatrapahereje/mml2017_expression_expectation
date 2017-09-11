"""
Diverse array and file handling utilities.
"""

import numpy as np
import os
import logging
import theano.tensor as T
import lasagne

print(__name__)
LOGGER = logging.getLogger(__name__)


def norm_shape(shape):
    '''Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Taken from http://www.johnvinyard.com/blog/?p=268

    Parameters
    ----------
    shape : int or a tuple of ints

    Returns
    -------
    tuple of ints
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''Return a sliding window over a in any number of dimensions.
    Taken from http://www.johnvinyard.com/blog/?p=268

    Parameters
    ----------
        a  : an n-dimensional numpy array
        ws : int or tuple
            Represents the size of each dimension of the window
        ss : an int (a is 1D) or tuple (a is 2D or greater)
            Representing the amount to slide the window in each dimension.
            If not specified, it defaults to ws.
        flatten : bool (default: True)
            If True, all slices are flattened, otherwise, there is an
            extra dimension for each dimension of the input.

    Returns
    -------
    numpy array
        An array containing each n-dimensional window from a
    '''
    from numpy.lib.stride_tricks import as_strided as ast
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy
    # arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(
            ('a.shape, ws and ss must all have the same length. '
             'They were %s' % str(ls)))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(
            'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will
    # be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides
    # multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one
    # more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


def delete_if_exists(fn):
    """Delete file if exists
    """
    try:
        os.unlink(fn)
    except OSError:
        pass


def write_append(fn, epoch, value, fmt='{0} {1:.8f}\n'):
    """Append value to a file
    """
    with open(fn, 'a') as f:
        f.write(fmt.format(epoch, value))


def stable_softmax(x):
    """Numerically stable softmax function
    """
    return ((1. - np.finfo(np.float32).resolution) * T.nnet.softmax(x) +
            np.finfo(np.float32).resolution)


def mean_nan_smasher(x, axis=0):
    """Computes the mean of an array, and replaces NaN with the largest
    32 bit number possible.
    """
    result = np.nanmean(x, axis=axis)
    # result = np.nansum(x, axis=axis)
    if np.isnan(result).any():
        LOGGER.warn(('WARNING: a wild NaN appeared. '
                     'This result should not be trusted'))
        nan_idx = np.where(np.isnan(result))
        result[nan_idx] = np.finfo(np.float32).max
    return result


def save_predictions_to_file(fn, y, y_hat):
    """Save targets and predictions to a file.
    """
    out_array = np.hstack([y.reshape(-1, 1), y_hat.reshape(-1, 1)])
    np.savetxt(fn,
               out_array,
               fmt='%f', delimiter='\t',
               header='\t'.join(['y', 'y_hat']))


def ensure_list(input):
    if not isinstance(input, (list, tuple)):
        return [input]
    else:
        return input


def get_all_params(l_out, recurrent_layers_f=None, recurrent_layers_b=None):

    train_params = lasagne.layers.get_all_params(l_out, trainable=True)
    all_params = lasagne.layers.get_all_params(l_out)

    if recurrent_layers_f is not None:
        init_params = _get_init_params(recurrent_layers_f)
    else:
        init_params = None

    if recurrent_layers_b is not None:
        end_params = _get_init_params(recurrent_layers_b)
    else:
        end_params = None

    return (train_params, all_params, init_params, end_params)


def _get_init_params(layers):
    init_params = []
    for l in ensure_list(layers):
        for p in l.get_params():
            if 'init' in p.name:
                init_params.append(p)
    return init_params
