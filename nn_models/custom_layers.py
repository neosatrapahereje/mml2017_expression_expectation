# -*- coding: utf-8 -*-
"""
Custom Lasagne Layer classes
"""
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import Layer, MergeLayer, helper
from lasagne.random import get_rng
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# from mi_rnn import mi_layers
import mi_layers
import batch_normalization

# call multiplicative integration layers directly from nn_models
MILSTMLayer = mi_layers.MILSTMLayer
MIRecurrentLayer = mi_layers.MIRecurrentLayer
CustomMIRecurrentLayer = mi_layers.CustomMIRecurrentLayer
MIGRULayer = mi_layers.MIGRULayer
MIGate = mi_layers.MIGate
BatchNormLSTMLayer = batch_normalization.BatchNormLSTMLayer


class ConvolutionalDropoutLayer(Layer):
    """Convolutional Dropout layer
    """

    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(ConvolutionalDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            m = self._srng.binomial(size=(input_shape[2], input_shape[3]),
                                    p=retain_prob, dtype=theano.config.floatX)
            m = m.dimshuffle('x', 'x', 0, 1)

            return input * m


class OuttoHidRecurrentLayer(MergeLayer):
    """CustomRecurrentLayer with an Output to Hidden Connection
    """

    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 output_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=False,
                 mask_input=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        super(OuttoHidRecurrentLayer, self).__init__(incomings, **kwargs)

        self.output_to_hidden = output_to_hidden
        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that the input_to_hidden connection can appropriately handle
        # a first dimension of input_shape[0]*input_shape[1] when we will
        # precompute the input dot product
        if (self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                input_shape[0] is not None and
                input_shape[1] is not None and
                (input_to_hidden.output_shape[0] !=
                 input_shape[0] * input_shape[1])):
            raise ValueError(
                'When precompute_input == True, '
                'input_to_hidden.output_shape[0] must equal '
                'incoming.output_shape[0]*incoming.output_shape[1] '
                '(i.e. batch_size*sequence_length) or be None but '
                'input_to_hidden.output_shape[0] = {} and '
                'incoming.output_shape[0]*incoming.output_shape[1] = '
                '{}'.format(input_to_hidden.output_shape[0],
                            input_shape[0] * input_shape[1]))

        # Check that the first dimension of input_to_hidden and
        # hidden_to_hidden's outputs match when we won't precompute the input
        # dot product
        if (not self.precompute_input and
                input_to_hidden.output_shape[0] is not None and
                hidden_to_hidden.output_shape[0] is not None and
                (input_to_hidden.output_shape[0] !=
                 hidden_to_hidden.output_shape[0])):
            raise ValueError(
                'When precompute_input == False, '
                'input_to_hidden.output_shape[0] must equal '
                'hidden_to_hidden.output_shape[0] but '
                'input_to_hidden.output_shape[0] = {} and '
                'hidden_to_hidden.output_shape[0] = {}'.format(
                    input_to_hidden.output_shape[0],
                    hidden_to_hidden.output_shape[0]))

        # Check that input_to_hidden and hidden_to_hidden output shapes match,
        # but don't check a dimension if it's None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.output_shape[1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        # Check that input_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.input_shape[1:])):
            raise ValueError("The output shape for input_to_hidden "
                             "must be equal to the input shape of "
                             "hidden_to_hidden after the first dimension, but "
                             "input_to_hidden.output_shape={} and "
                             "hidden_to_hidden.input_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.input_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != len(hidden_to_hidden.output_shape):
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have the same shape as hidden_to_hidden.output_shape")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1,) + hidden_to_hidden.output_shape[1:],
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(OuttoHidRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        params += helper.get_all_params(self.output_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return ((input_shape[0], input_shape[1]) +
                self.hidden_to_hidden.output_shape[1:])

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len * num_batch,) + trailing_dims)
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        non_seqs += helper.get_all_params(self.output_to_hidden)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)

            # out_layers = helper.get_all_layers(self.output_to_hidden)
            # out_layers[1].incoming_layer = self.hidden_to_hidden
            hid_pre += helper.get_output(
                self.output_to_hidden, hid_previous, **kwargs)

            # If the dot product is precomputed then add it, otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(
                    self.input_to_hidden, input_n, **kwargs)

            # Clip gradients
            if self.grad_clipping:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)

            return self.nonlinearity(hid_pre)

        def step_masked(input_n, mask_n, hid_previous, *args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = step(input_n, hid_previous, *args)
            hid_out = hid * mask_n + hid_previous * (1 - mask_n)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        # When hid_init is provided as a TensorVariable, use it as-is
        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out


class CustomBiasLayer(Layer):
    """
    CustomBiasLayer(incoming, b=lasagne.init.Constant(0),
    shared_axes='auto', shape='auto', **kwargs)

    A layer that just adds a (trainable) bias term.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases and pass through its input
        unchanged. Otherwise, the bias shape must match the incoming shape,
        skipping those axes the biases are shared over (see the example below).
        See :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', int or tuple of int
        The axis or axes to share biases over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share biases over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.

    shape : 'auto', or tuple of int
        Shape of the output parameters. Workaround for RNNs with variable
        sequence length.

    Notes
    -----
    The bias parameter dimensionality is the input dimensionality minus the
    number of axes the biases are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:

    >>> layer = BiasLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.b.get_value().shape
    (30, 50)
    """
    def __init__(self, incoming, b=init.Constant(0), shared_axes='auto',
                 shape='auto', **kwargs):
        super(CustomBiasLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share biases over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        if b is None:
            self.b = None
        else:
            if shape == 'auto':
                # create bias parameter, ignoring all dimensions in shared_axes
                shape = [size for axis, size in enumerate(self.input_shape)
                         if axis not in self.shared_axes]
                if any(size is None for size in shape):
                    raise ValueError("if shape is automatically computed, the"
                                     "CustomBiasLayer needs specified input "
                                     "sizes for all axes that biases "
                                     "are not shared over.")
            self.b = self.add_param(b, shape, 'b', regularizable=False)

    def get_output_for(self, input, **kwargs):
        if self.b is not None:
            bias_axes = iter(range(self.b.ndim))
            pattern = ['x' if input_axis in self.shared_axes
                       else next(bias_axes)
                       for input_axis in range(input.ndim)]
            return input + self.b.dimshuffle(*pattern)
        else:
            return input

class ReverseLayer(MergeLayer):

    """
    The :class:`ReverseLayer` class performs inverse operations
    for a single layer of a neural network by applying the
    partial derivative of the layer to be inverted with
    respect to its input: transposed layer
    for a :class:`DenseLayer`, deconvolutional layer for
    :class:`Conv2DLayer`, :class:`Conv1DLayer`; or
    an unpooling layer for :class:`MaxPool2DLayer`.

    It is specially useful for building
    autoencoders with tied parameters.

    Note that if the layer to be inverted contains a nonlinearity
    and/or a bias, the :class:`InverseLayer` will include the derivative
    of that in its computation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    layer : a :class:`Layer` instance or a tuple
        The layer with respect to which the instance of the
        :class:`InverseLayer` is inverse to.

    Examples
    --------
    >>> import lasagne
    >>> from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer
    >>> from custom import ReverseLayer
    >>> l_in = InputLayer((100, 3, 28, 28))
    >>> l1 = Conv2DLayer(l_in, num_filters=16, filter_size=5)
    >>> l2 = DenseLayer(l1, num_units=20)
    >>> l_u2 = ReverseLayer(l2, l2)  # backprop through l2
    >>> l_u1 = ReverseLayer(l_u2, l1)  # backprop through l1
    """

    def __init__(self, incoming, layer, wrt=0, **kwargs):

        if hasattr(layer, 'input_layers'):
            input_layer = layer.input_layers[wrt]
        else:
            input_layer = layer.input_layer

        super(ReverseLayer, self).__init__(
            [incoming, layer, input_layer], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[2]

    def get_output_for(self, inputs, **kwargs):
        input, layer_out, layer_in = inputs
        return theano.grad(None, wrt=layer_in, known_grads={layer_out: input})


class PolynomialLayer(Layer):
    """
    PolynomialLayer
    """
    def __init__(self, incoming, p_order, W=init.GlorotUniform(),
                 num_leading_axes=1, **kwargs):

        super(PolynomialLayer, self).__init__(incoming, **kwargs)

        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "leaving no trailing axes for the dot product." %
                (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "requesting more trailing axes than there are input "
                "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                "A PolynomialLayer requires a fixed input shape (except for "
                "the leading axes). Got %r for num_leading_axes=%d." %
                (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.p_order = p_order
        self.W = self.add_param(W, (p_order + 1, num_inputs), name="W")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

    def get_output_for(self, input, **kwargs):

        input = T.concatenate([input ** i for i in range(self.p_order + 1)],
                              axis=1)

        activation = T.dot(input, self.W)

        return activation
