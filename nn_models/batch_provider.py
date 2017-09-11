#!/usr/bin/env python

import numpy as np
import argparse
import pdb
import logging


class BatchProvider(object):

    """A class to load data from files and serve it in batches

    """

    def __init__(self, dtype):
        self.data = []
        self.sizes = []
        self.dtype = dtype

    def store_data(self, X, Y):
        """
        Store sequence data from which batches are to be provided. `X` and `Y` must
        be lists of equal length.

        Parameters
        ----------

        X : list
            List of input sequences (ndarray)

        Y : list
            List of output sequences (ndarray)
        """

        assert len(X) == len(Y)

        x_dim, y_dim = None, None
        for x, y in zip(X, Y):

            if not np.all(x_dim == x.shape[1:]):
                if x_dim is None:
                    x_dim = x.shape[1:]
                else:
                    raise Exception('Cannot deal with variable input shapes')

            if not np.all(y_dim == y.shape[1:]):
                if y_dim is None:
                    y_dim = y.shape[1:]
                else:
                    raise Exception('Cannot deal with variable output shapes')

            self.data.append((x, y))
            self.sizes.append(len(x))

        self.x_dim = x_dim
        self.y_dim = y_dim

        self._cs = np.r_[0, np.cumsum(self.sizes)]

    def make_X_batch_array(self, batch_size, segment_length):
        return np.empty([batch_size, segment_length] + list(self.x_dim), dtype=self.dtype)

    def make_Y_batch_array(self, batch_size, segment_length):
        return np.empty([batch_size, segment_length] + list(self.y_dim), dtype=self.dtype)

    def iter_pieces(self):
        for x, y in self.data:
            yield (x[np.newaxis,:,:].astype(self.dtype, copy=False),
                   y[np.newaxis,:,:].astype(self.dtype, copy=False))

    # to be deprecated
    def iter_pieces_batch(self, stepsize, segment_length):
        for x, y in self.data:
            batch_size = (len(x) - segment_length / 2) // stepsize

            if batch_size < 1:
                continue

            xn = np.empty([batch_size, segment_length] + list(x.shape[1:]),
                          dtype=self.dtype)
            yn = np.empty([batch_size, segment_length] + list(y.shape[1:]),
                          dtype=self.dtype)
            for i in range(0, batch_size):
                start = i * stepsize
                end = start + segment_length
                xn[i] = x[start: end]
                yn[i] = y[start: end]

            yield xn, yn

    def get_batch_full(self, batch_size, segment_length,
                       batch_X=None, batch_Y=None):
        """
        Return a batch from the stored data. The segments in the batch may start
        before the start of a data sequence. In this case, they are zero-padded
        on the left, up to the start of the sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_full, batch_size,
                               segment_length, batch_X, batch_Y)

    def get_batch_valid(self, batch_size, segment_length,
                        batch_X=None, batch_Y=None):
        """
        Return a batch from the stored data. Other than for `get_batch_full`, the
        segments in the batch are always the subseqeuence of a data sequence. No
        zero-padding will take place. Note that this implies that data from
        sequences shorter than `segment_length` will never appear in the
        returned batches.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data

        batch_Y : ndarray, optional
            An array for storing the X batch data


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_valid, batch_size,
                               segment_length, batch_X, batch_Y)

    def get_batch_start(self, batch_size, segment_length,
                        batch_X=None, batch_Y=None):
        """
        Return a batch from the stored data. This function returns only segments
        starting at the beginning of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_start, batch_size,
                               segment_length, batch_X, batch_Y)

    def get_batch_end(self, batch_size, segment_length,
                      batch_X=None, batch_Y=None):
        """
        Return a batch from the stored data. This function returns only segments
        ending at the end of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_end, batch_size,
                               segment_length, batch_X, batch_Y)

    def _get_batch(self, segment_producer, batch_size, segment_length,
                   batch_X=None, batch_Y=None):

        if batch_X is None:
            batch_X = self.make_X_batch_array(batch_size, segment_length)
        if batch_Y is None:
            batch_Y = self.make_Y_batch_array(batch_size, segment_length)

        for i, (piece, segment_end) in enumerate(segment_producer(batch_size,
                                                                  segment_length)):
            X, Y = self.data[piece]

            start = segment_end - segment_length
            start_trimmed = max(0, start)

            batch_X[i, - (segment_end - start_trimmed):] = X[
                start_trimmed: segment_end]
            batch_Y[i, - (segment_end - start_trimmed):] = Y[
                start_trimmed: segment_end]

            if start < 0:
                batch_X[i, :- (segment_end - start_trimmed)] = 0
                batch_Y[i, :- (segment_end - start_trimmed)] = 0

        return batch_X, batch_Y

    def _select_segments_start(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))
        return np.column_stack(
            (piece_idx, np.ones(k, dtype=np.int) * segment_size))

    def _select_segments_end(self, k, segment_size):
        sizes = np.array(self.sizes)
        available_idx = sizes - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        return np.column_stack((piece_idx, sizes[piece_idx]))

    def _select_segments_valid(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size + 1
        valid = np.where(available_idx > 0)[0]
        cum_idx = np.cumsum(available_idx[valid])

        try:
            segment_starts = np.random.randint(0, cum_idx[-1], k)
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        piece_idx = np.searchsorted(cum_idx - 1, segment_starts, side='left')
        index_within_piece = segment_starts - np.r_[0, cum_idx[:-1]][piece_idx]

        return np.column_stack(
            (valid[piece_idx], index_within_piece + segment_size))

    def _select_segments_full(self, k, segment_size):
        total_instances = self._cs[-1]
        segment_ends = np.random.randint(1, total_instances + 1, k)
        piece_idx = np.searchsorted(self._cs[1:], segment_ends, side='left')
        index_within_piece = segment_ends - self._cs[piece_idx]

        return np.column_stack((piece_idx, index_within_piece))


class ConcatBatchProvider(BatchProvider):

    def __init__(self, dtype, random_seed=1984):
        self.prng = np.random.RandomState(random_seed)
        super(ConcatBatchProvider, self).__init__(dtype)

    def iter_pieces(self):

        for x, y in self.data:
            yield (x.astype(self.dtype, copy=False),
                   y.astype(self.dtype, copy=False))

    def _concatenate_instances(self):

        X = []
        Y = []

        for x, y in self.data:
            X.append(x)
            Y.append(y)

        self.datac = (np.vstack(X), np.vstack(Y))

    def iter_pieces_batch(self, batch_size, shuffle=True, mode='valid'):

        idx = np.arange(self._cs[-1])

        if shuffle:
            self.prng.shuffle(idx)

        if mode == 'valid':
            n_batches = len(idx) // batch_size
        elif mode == 'full':
            n_batches = int(np.ceil(len(idx) / float(batch_size)))

        if n_batches == 0:
            batch_size = self._cs[-1]
            n_batches = 1

        for batch_num in xrange(n_batches):
            batch_slice = slice(batch_size * batch_num,
                                min(len(idx), batch_size * (batch_num + 1)))

            yield self.datac[0][batch_slice], self.datac[1][batch_slice]


class EMBatchProvider(BatchProvider):

    def __init__(self, dtype, rho_fun=None, random_seed=1984):
        self.prng = np.random.RandomState(random_seed)
        # Make sure to initialize rho_fun
        self.rho_fun = rho_fun
        super(EMBatchProvider, self).__init__(dtype)

    def make_rho_batch_array(self, batch_size, segment_length):
        # The size of the rho batch array is (n_components,
        # batch_size, sequence_length)
        return np.empty([self.rho_dim, batch_size, segment_length],
                        dtype=self.dtype)

    def update_rho(self):
        # Save the value of the responsabilities
        # The output of of rho_fun  is
        # n_components, batch_size (1 for a single sequence),
        # sequence_length, and we want to store it
        # as batch_size, segment_length, n_components
        self.all_rho = [
            self.rho_fun(x[np.newaxis,:],
                         y[np.newaxis,:])
            for x, y in self.data]

        # flatten first dimenstion of array (to have a 3D object)
        # self.all_rho = [r.reshape(r.shape[1], r.shape[2])
        #                 for r in self.all_rho]
        self.rho_dim = self.all_rho[0].shape[0]

    def iter_pieces(self):
        for (x, y), r in zip(self.data, self.all_rho):
            yield (x[np.newaxis,:,:].astype(self.dtype,
                                              copy=False),
                   y[np.newaxis,:,:].astype(self.dtype,
                                              copy=False),
                   r.astype(self.dtype, copy=False))

    def _get_batch(self, segment_producer, batch_size,
                   segment_length,
                   batch_X=None, batch_Y=None,
                   batch_rho=None):

        if batch_X is None:
            batch_X = self.make_X_batch_array(batch_size,
                                              segment_length)
        if batch_Y is None:
            batch_Y = self.make_Y_batch_array(batch_size,
                                              segment_length)
        if batch_rho is None:
            batch_rho = self.make_rho_batch_array(batch_size,
                                                  segment_length)

        for i, (piece, segment_end) in enumerate(segment_producer(batch_size,
                                                                  segment_length)):
            X, Y = self.data[piece]
            R = self.all_rho[piece]

            start = segment_end - segment_length
            start_trimmed = max(0, start)

            batch_X[i, - (segment_end - start_trimmed):] = X[
                start_trimmed: segment_end]
            batch_Y[i, - (segment_end - start_trimmed):] = Y[
                start_trimmed: segment_end]

            batch_rho[:, i, - (segment_end - start_trimmed):] = R[
                :, 0, start_trimmed: segment_end]

            if start < 0:
                batch_X[i, :- (segment_end - start_trimmed)] = 0
                batch_Y[i, :- (segment_end - start_trimmed)] = 0
                batch_rho[i, :- (segment_end - start_trimmed)] = 1

        return batch_X, batch_Y, batch_rho

    def get_batch_full(self, batch_size, segment_length,
                       batch_X=None, batch_Y=None,
                       batch_rho=None):
        """
        Return a batch from the stored data. The segments in the batch may start
        before the start of a data sequence. In this case, they are zero-padded
        on the left, up to the start of the sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_full,
                               batch_size,
                               segment_length, batch_X,
                               batch_Y, batch_rho)

    def get_batch_valid(self, batch_size, segment_length,
                        batch_X=None, batch_Y=None,
                        batch_rho=None):
        """
        Return a batch from the stored data. Other than for `get_batch_full`, the
        segments in the batch are always the subseqeuence of a data sequence. No
        zero-padding will take place. Note that this implies that data from
        sequences shorter than `segment_length` will never appear in the
        returned batches.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data

        batch_Y : ndarray, optional
            An array for storing the X batch data


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_valid,
                               batch_size,
                               segment_length, batch_X,
                               batch_Y, batch_rho)

    def get_batch_start(self, batch_size, segment_length,
                        batch_X=None, batch_Y=None,
                        batch_rho=None):
        """
        Return a batch from the stored data. This function returns only segments
        starting at the beginning of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_start,
                               batch_size,
                               segment_length, batch_X,
                               batch_Y, batch_rho)

    def get_batch_end(self, batch_size, segment_length,
                      batch_X=None, batch_Y=None,
                      batch_rho=None):
        """
        Return a batch from the stored data. This function returns only segments
        ending at the end of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_X : ndarray, optional
            An array for storing the X batch data in

        batch_Y : ndarray, optional
            An array for storing the X batch data in


        Returns
        -------

        tuple
            A tuple with two ndarrays, containing the X and Y data for the batch
        """
        return self._get_batch(self._select_segments_end,
                               batch_size,
                               segment_length, batch_X,
                               batch_Y, batch_rho)


class EMConcatBatchProvider(BatchProvider):

    def __init__(self, dtype, rho_fun=None, random_seed=1984):

        self.prng = np.random.RandomState(random_seed)
        # Make sure to initialize rho_fun
        self.rho_fun = rho_fun
        super(EMConcatBatchProvider, self).__init__(dtype)

    def update_rho(self):

        self.all_rho = [self.rho_fun(x, y) for x, y in self.data]

        self.all_rhoc = np.vstack([r.transpose(1, 0) for r in self.all_rho])

    def _concatenate_instances(self):
        X = []
        Y = []

        for x, y in self.data:
            X.append(x)
            Y.append(y)

        self.datac = (np.vstack(X), np.vstack(Y))

    def iter_pieces(self):

        for (x, y), r in zip(self.data, self.all_rho):
            yield (x.astype(self.dtype, copy=False),
                   y.astype(self.dtype, copy=False),
                   r)

    def iter_pieces_batch(self, batch_size, shuffle=True,
                          mode='valid'):

        idx = np.arange(self._cs[-1])

        if shuffle:
            self.prng.shuffle(idx)

        if mode == 'valid':
            n_batches = len(idx) // batch_size
        elif mode == 'full':
            n_batches = int(np.ceil(len(idx) / float(batch_size)))

        if n_batches == 0:
            batch_size = self._cs[-1]
            n_batches = 1

        for batch_num in xrange(n_batches):
            batch_slice = slice(batch_size * batch_num,
                                min(len(idx), batch_size * (batch_num + 1)))

            yield (self.datac[0][batch_slice],
                   self.datac[1][batch_slice],
                   self.all_rhoc[batch_slice].transpose(1, 0))


def test():
    n_inputs = 2
    n_outputs = 2
    n_pieces = 3
    min_piece_len = 5
    max_piece_len = 10

    # create some data
    piece_lens = np.random.randint(min_piece_len, max_piece_len + 1, n_pieces)
    # X = [np.random.random((n_instances, n_inputs))
    #      for n_instances in piece_lens]
    X = [np.column_stack((np.ones(n_instances) * i, np.arange(n_instances))).astype(np.int)
         for i, n_instances in enumerate(piece_lens)]
    Y = [np.random.random((n_instances, n_outputs))
         for n_instances in piece_lens]

    bp = BatchProvider('float32')
    # store the data in the batch provider
    bp.store_data(X, Y)
    for x in X:
        print(x)

    batch_size = 10
    seq_len = 5
    # print('batch')
    # for p, s in bp._select_segments_valid(batch_size, seq_len):
    #     print(X[p][s-seq_len:s,:])
    # print('end batch')
    # if True:
    #     return
    # batch_size = 10
    # seq_len = 2

    # get data batches

    X, Y = bp.get_batch_end(batch_size, seq_len)

    for i, x in enumerate(X):
        print('sequence {}'.format(i))
        print(x)
        print('')


def test_concatbatch():
    logging.basicConfig(level=logging.DEBUG)
    # test()
    n_inputs = 2
    n_outputs = 2
    n_pieces = 3
    min_piece_len = 5
    max_piece_len = 10

    # create some data
    piece_lens = np.random.randint(min_piece_len, max_piece_len + 1, n_pieces)
    # X = [np.random.random((n_instances, n_inputs))
    #      for n_instances in piece_lens]
    X = [np.column_stack((np.ones(n_instances) * i,
                          np.arange(n_instances))).astype(np.int)
         for i, n_instances in enumerate(piece_lens)]
    Y = [np.random.random((n_instances, n_outputs))
         for n_instances in piece_lens]

    bp = ConcatBatchProvider(np.float32)
    # store the data in the batch provider
    bp.store_data(X, Y)
    bp._concatenate_instances()
    for x in X:
        print 'x'
        print(x)

    batch_size = 10

    for x, y in bp.iter_pieces_batch(batch_size, False):
        print 'batch'
        print x

    for x, y in bp.iter_pieces():
        print 'pieces'
        print x


class RecurrentBatchProvider(object):
    """A class to load data from files and serve it in batches
    """
    def __init__(self, dtype=np.float32):
        self.data = []
        self.sizes = []
        self.dtype = dtype

    def store_data(self, *args):
        
        if not all([len(x) == len(args[0]) for x in args]):
            raise Exception('The length of each array must be the same')

        self.n_inputs = len(args)

        dims = [None] * len(args)
        for arrays in zip(*args):

            for i, array in enumerate(arrays):
                if not np.all(dims[i] == array.shape[1:]):
                    if dims[i] is None:
                        dims[i] = array.shape[1:]
                    else:
                        raise Exception(
                            'Cannot deal with variable output shapes')

            self.data.append(arrays)
            self.sizes.append(len(arrays[0]))

        self.dims = dims
        self._cs = np.r_[0, np.cumsum(self.sizes)]

    def _make_batch_array(self, batch_size, segment_length, dim):
        return np.empty([batch_size, segment_length] + list(dim), dtype=self.dtype)

    def make_batch_arrays(self, batch_size, segment_length):
        return [self._make_batch_array(batch_size, segment_length, dim)
                for dim in self.dims]

    def iter_pieces(self):
        for arrays in self.data:
            yield (array[np.newaxis,:,:].astype(self.dtype, copy=False)
                   for array in arrays)

    def _get_batch(self, segment_producer, batch_size, segment_length,
                   batch_arrays=None):

        if batch_arrays is None:
            batch_arrays = self.make_batch_arrays(batch_size, segment_length)
        else:
            # Check that the number of given arrays is the same as the number of
            # inputs
            if len(batch_arrays) != self.n_inputs:
                raise Exception(('Different number of arrays provided: {0} given '
                                 'but {1} expected').format(len(batch_arrays),
                                                            self.n_inputs))

        for i, (piece, segment_end) in enumerate(segment_producer(batch_size,
                                                                  segment_length)):
            arrays = self.data[piece]

            start = segment_end - segment_length
            start_trimmed = max(0, start)

            for batch_a, array in zip(batch_arrays, arrays):
                batch_a[i, - (segment_end - start_trimmed):] = array[
                    start_trimmed: segment_end]
                    

            if start < 0:
                for batch_a in batch_arrays:
                    batch_a[i, :- (segment_end - start_trimmed)] = 0

        return batch_arrays

    def _select_segments_start(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))
        return np.column_stack(
            (piece_idx, np.ones(k, dtype=np.int) * segment_size))

    def _select_segments_end(self, k, segment_size):
        sizes = np.array(self.sizes)
        available_idx = sizes - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        return np.column_stack((piece_idx, sizes[piece_idx]))

    def _select_segments_valid(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size + 1
        valid = np.where(available_idx > 0)[0]
        cum_idx = np.cumsum(available_idx[valid])

        try:
            segment_starts = np.random.randint(0, cum_idx[-1], k)
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        piece_idx = np.searchsorted(cum_idx - 1, segment_starts, side='left')
        index_within_piece = segment_starts - np.r_[0, cum_idx[:-1]][piece_idx]

        return np.column_stack(
            # (valid[piece_idx], index_within_piece))
            (valid[piece_idx], index_within_piece + segment_size))

    def _select_segments_full(self, k, segment_size):
        total_instances = self._cs[-1]
        segment_ends = np.random.randint(1, total_instances + 1, k)
        piece_idx = np.searchsorted(self._cs[1:], segment_ends, side='left')
        index_within_piece = segment_ends - self._cs[piece_idx]

        return np.column_stack((piece_idx, index_within_piece))

    def get_batch_full(self, batch_size, segment_length,
                       batch_arrays=None):
        """
        Return a batch from the stored data. The segments in the batch may start
        before the start of a data sequence. In this case, they are zero-padded
        on the left, up to the start of the sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences
            
        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_full, batch_size,
                               segment_length, batch_arrays)

    def get_batch_valid(self, batch_size, segment_length,
                        batch_arrays=None):
        """
        Return a batch from the stored data. Other than for `get_batch_full`, the
        segments in the batch are always the subseqeuence of a data sequence. No
        zero-padding will take place. Note that this implies that data from
        sequences shorter than `segment_length` will never appear in the
        returned batches.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences
        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_valid, batch_size,
                               segment_length, batch_arrays)

    def get_batch_start(self, batch_size, segment_length,
                        batch_arrays=None):
        """
        Return a batch from the stored data. This function returns only segments
        starting at the beginning of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_start, batch_size,
                               segment_length, batch_arrays)

    def get_batch_end(self, batch_size, segment_length,
                      batch_arrays=None):
        """
        Return a batch from the stored data. This function returns only segments
        ending at the end of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_end, batch_size,
                               segment_length, batch_arrays)



def test_EMBatchProvider():
    n_inputs = 2
    n_outputs = 2
    n_pieces = 3
    min_piece_len = 5
    max_piece_len = 10

    # create some data
    piece_lens = np.random.randint(min_piece_len,
                                   max_piece_len + 1,
                                   n_pieces)
    # X = [np.random.random((n_instances, n_inputs))
    #      for n_instances in piece_lens]
    X = [np.column_stack((np.ones(n_instances) * i,
                          np.arange(n_instances))).astype(np.int)
         for i, n_instances in enumerate(piece_lens)]
    Y = [np.random.random((n_instances, n_outputs))
         for n_instances in piece_lens]
    rho_fun = lambda x, y: np.mean(x, -1) + np.mean(y, -1)
    bp = EMBatchProvider('float32', rho_fun)
    # store the data in the batch provider
    bp.store_data(X, Y)
    for x in X:
        print(x)

    bp.update_rho()

    batch_size = 10
    seq_len = 5
    # print('batch')
    # for p, s in bp._select_segments_valid(batch_size, seq_len):
    #     print(X[p][s-seq_len:s,:])
    # print('end batch')
    # if True:
    #     return
    # batch_size = 10
    # seq_len = 2

    # get data batches

    X, Y, R = bp.iter_pieces()

    for i, x in enumerate(R):
        print('sequence {}'.format(i))
        print(x)
        print('')

if __name__ == '__main__':
    n_inputs = 2
    n_outputs = 2
    n_pieces = 3
    min_piece_len = 5
    max_piece_len = 10

    # create some data
    piece_lens = np.random.randint(min_piece_len,
                                   max_piece_len + 1,
                                   n_pieces)
    # X = [np.random.random((n_instances, n_inputs))
    #      for n_instances in piece_lens]
    X = [np.column_stack((np.ones(n_instances) * i,
                          np.arange(n_instances))).astype(np.int)
         for i, n_instances in enumerate(piece_lens)]
    # Y = [np.random.random((n_instances, n_outputs))
    #      for n_instances in piece_lens]
    Y = X
    bp = RecurrentBatchProvider()
    bp.store_data(X, Y)

    bp_old = BatchProvider(np.float32)
    bp_old.store_data(X, Y)
