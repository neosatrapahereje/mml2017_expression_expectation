import lasagne
import theano.tensor as T

from lasagne import layers

# from nn_models.architectures import _build_mse_loss
from nn_models.mixture_density_networks import mixture_density_network
from nn_models import custom_layers as c_layers
from nn_models.architectures import ffnn

l1d_l2d = ffnn


def BatchNormRecurrentLayer(incoming, num_units, nonlinearity=None,
                            gradient_steps=-1, grad_clipping=0,
                            layer_type=layers.CustomRecurrentLayer,
                            name='', **kwargs):
    """
    Helper method to define a Vanilla Recurrent Layer with batch normalization
    """
    input_shape = incoming.output_shape
    # Define input to hidden connections
    in_to_hid_rf = layers.InputLayer((None, ) + input_shape[2:])
    in_to_hid_rf = layers.DenseLayer(
        in_to_hid_rf, num_units, b=None, nonlinearity=None,
        name='ith_{0}'.format(name))
    in_to_hid_rf_W = in_to_hid_rf.W

    # Use batch normalization in the input to hidden connections
    in_to_hid_rf = layers.BatchNormLayer(in_to_hid_rf,
                                         name='ith_bn_{0}'.format(name))

    # Define hidden to hidden connections
    hid_to_hid_rf = layers.InputLayer(
        (None, num_units))
    hid_to_hid_rf = layers.DenseLayer(
        hid_to_hid_rf, num_units, b=None, nonlinearity=None,
        name='hth_{0}'.format(name))

    l_r_f = layer_type(
        incoming, input_to_hidden=in_to_hid_rf,
        hidden_to_hidden=hid_to_hid_rf,
        gradient_steps=gradient_steps,
        grad_clipping=grad_clipping,
        nonlinearity=nonlinearity,
        name='l_r_{0}'.format(name),
        **kwargs)

    # Make layer parameters intuitively accessible
    l_r_f.W_in_to_hid = in_to_hid_rf_W
    l_r_f.W_hid_to_hid = hid_to_hid_rf.W
    l_r_f.beta = in_to_hid_rf.beta
    l_r_f.gamma = in_to_hid_rf.gamma
    l_r_f.mean = in_to_hid_rf.mean
    l_r_f.inv_std = in_to_hid_rf.inv_std
    l_r_f.hid_init = l_r_f.hid_init
    return l_r_f


def l1r_l2d(input_dim, output_dim, n_hidden,
            nonlinearity=lasagne.nonlinearities.tanh,
            layer_type=layers.CustomRecurrentLayer,
            learning_rate=1e-4,
            wl2=0.,
            wl1=0,
            r_reg_coeff=0.,
            grad_clipping=0.,
            bidirectional=False,
            skip_connection=False,
            **kwargs):

    # Specify the number of steps used before computing the gradient
    if 'gradient_steps' not in kwargs:
        gradient_steps = -1
    else:
        gradient_steps = kwargs.pop('gradient_steps')

    # Input Layer
    l_in = layers.InputLayer((None, None, input_dim))

    if skip_connection:
        # Input to output connection
        l_in_to_out = lasagne.layers.DenseLayer(
            lasagne.layers.ReshapeLayer(l_in, (-1, input_dim)),
            output_dim,
            nonlinearity=None,
            name='in_to_out')

    b_size, seqlen, _ = l_in.input_var.shape

    l_r_f = BatchNormRecurrentLayer(
        incoming=l_in,
        num_units=n_hidden,
        nonlinearity=nonlinearity,
        gradient_steps=gradient_steps,
        grad_clipping=grad_clipping,
        layer_type=layer_type,
        name='rf',
        **kwargs)

    if bidirectional:
        print 'Using bidirectional network'
        l_r_b = BatchNormRecurrentLayer(
            incoming=l_in,
            num_units=n_hidden,
            nonlinearity=nonlinearity,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping,
            layer_type=layer_type,
            name='rb',
            backwards=True,
            **kwargs)

    l_concat = l_r_f
    out_shape = n_hidden

    if bidirectional:
        print 'Concatenating Forward and Backward recurrent layers'
        l_concat = layers.ConcatLayer((l_concat, l_r_b), axis=-1)
        out_shape = out_shape + n_hidden
    l_re = layers.ReshapeLayer(l_concat, (-1, out_shape), name='reshape')

    l_d = layers.DenseLayer(l_re, output_dim,
                            nonlinearity=None,
                            name='dense')

    if skip_connection:
        # Combine input_to_output and hidden to output layers
        l_output = lasagne.layers.ElemwiseSumLayer([l_in_to_out, l_d])

    else:
        l_output = l_d

    if kwargs.get('only_return_final', False):
        out_shape = (b_size, 1, output_dim)
    else:
        out_shape = (b_size, seqlen, output_dim)

    l_out = layers.ReshapeLayer(l_output, out_shape)

    deterministic_out = layers.get_output(l_out, deterministic=True)
    deterministic_out.name = 'deterministic out'
    stochastic_out = layers.get_output(l_out, deterministic=False)
    stochastic_out.name = 'stochastic out'

    target = T.tensor3()
    target.name = 'target'

    params = layers.get_all_params(l_out, trainable=True)

    if wl2 > 0:
        print 'Using L2 norm regularization'
        weight_reg = wl2 * (T.mean(l_r_f.W_in_to_hid ** 2) +
                            T.mean(l_d.W ** 2))
        if skip_connection:
            weight_reg += wl2 * T.mean(l_in_to_out.W ** 2)
    else:
        weight_reg = 0.

    if wl1 > 0:
        print 'Using L1 norm regularization'
        weight_reg += wl1 * (T.mean(abs(l_r_f.W_in_to_hid)) +
                             T.mean(abs(l_d.W)))
        if skip_connection:
            weight_reg += wl1 * T.mean(abs(l_in_to_out.W))

    if r_reg_coeff > 0:
        print 'Using hid to hid eigenvalue regularization'
        weight_reg += r_reg_coeff * T.mean(
            (T.nlinalg.eigh(l_r_f.W_hid_to_hid)[0] - 1.) ** 2)

    stochastic_loss = (lasagne.objectives.squared_error(stochastic_out, target).mean()
                       + weight_reg)

    stochastic_loss.name = 'stochastic MSE (regularized)'

    deterministic_loss = lasagne.objectives.squared_error(
        deterministic_out, target).mean()
    deterministic_loss.name = 'MSE'

    updates = lasagne.updates.rmsprop(
        stochastic_loss, params, learning_rate=learning_rate)

    train_loss = [stochastic_loss]
    valid_loss = [deterministic_loss]

    return dict(l_in=l_in, l_out=l_out,
                train_loss=train_loss,
                valid_loss=valid_loss,
                target=target, updates=updates,
                predictions=deterministic_out,
                gradient_steps=gradient_steps,
                model_type='RNN')


def l1r_l2d_gmn(input_dim, output_dim, n_hidden,
                nonlinearity=lasagne.nonlinearities.tanh,
                layer_type=layers.CustomRecurrentLayer,
                eta_objective=None,
                mean_nonlinearity=None,
                n_components=2,
                learning_rate=1e-4,
                wl2=0.,
                wl1=0,
                r_reg_coeff=0.,
                grad_clipping=0.,
                loss_type='log_likelihood',
                gain=1.0,
                **kwargs):

    # Specify the number of steps used before computing the gradient
    if 'gradient_steps' not in kwargs:
        gradient_steps = -1
    else:
        gradient_steps = kwargs.pop('gradient_steps')

    # Input Layer
    l_in = layers.InputLayer((None, None, input_dim))
    b_size, seqlen, _ = l_in.input_var.shape

    # Define input to hidden connections
    in_to_hid_rf = layers.InputLayer((None, input_dim))
    in_to_hid_rf = layers.DenseLayer(
        in_to_hid_rf, n_hidden, b=None, nonlinearity=None,
        name='ith_rf')
    in_to_hid_rf_W = in_to_hid_rf.W
    # Use batch normalization in the input to hidden connections
    in_to_hid_rf = layers.BatchNormLayer(in_to_hid_rf,
                                         name='ith_bn_rf')

    # Define hidden to hidden connections
    hid_to_hid_rf = layers.InputLayer(
        (None, n_hidden))
    hid_to_hid_rf = layers.DenseLayer(
        hid_to_hid_rf, n_hidden, b=None, nonlinearity=None,
        name='hth_rf')

    l_r_f = layer_type(
        l_in, input_to_hidden=in_to_hid_rf,
        hidden_to_hidden=hid_to_hid_rf,
        gradient_steps=gradient_steps,
        grad_clipping=grad_clipping,
        name='l_r_f')

    l_nl = layers.NonlinearityLayer(
        l_r_f, nonlinearity=nonlinearity,
        name='nl')

    l_re = layers.ReshapeLayer(l_nl, (-1, n_hidden),
                               name='reshape_1')

    target = T.tensor3()
    # make Mixture density outputs
    out_dict = mixture_density_network(
        coding_layer=l_re,
        target=target,
        mean_nonlinearity=mean_nonlinearity,
        eta_objective=eta_objective,
        n_components=n_components,
        output_dim=output_dim,
        b_size=b_size,
        seqlen=seqlen,
        loss_type=loss_type,
        gain=gain,
        wl1=wl1,
        wl2=wl2,
        weight_reg_params=[in_to_hid_rf_W])
    out_dict['l_in'] = l_in
    out_dict['gradient_steps'] = gradient_steps

    if loss_type == 'log_likelihood':
        out_dict['model_type'] = 'MDRNN'
    elif loss_type == 'em':
        out_dict['model_type'] = 'MDRNN_EM'

    return out_dict


def l1lstm_l2d(input_dim, output_dim, n_hidden,
               nonlinearity=lasagne.nonlinearities.tanh,
               layer_type=layers.LSTMLayer,
               learning_rate=1e-4,
               wl2=0.,
               wl1=0,
               r_reg_coeff=0.,
               grad_clipping=0.,
               bidirectional=False,
               loss_type='MSE',
               skip_connection=False,
               **kwargs):

    # Specify the number of steps used before computing the gradient
    if 'gradient_steps' not in kwargs:
        gradient_steps = -1
    else:
        gradient_steps = kwargs.pop('gradient_steps')

    target = T.tensor3()
    target.name = 'target'

    # Input Layer
    l_in = layers.InputLayer((None, None, input_dim))
    input_layer = l_in
    if bidirectional:
        input_layer_b = l_in

    if skip_connection:
        # Input to output connection
        l_in_to_out = lasagne.layers.DenseLayer(
            lasagne.layers.ReshapeLayer(l_in, (-1, input_dim)),
            output_dim,
            nonlinearity=None,
            name='in_to_out')

    b_size, seqlen, _ = l_in.input_var.shape

    lstm_layers = (layers.LSTMLayer, c_layers.MILSTMLayer,
                   c_layers.BatchNormLSTMLayer)
    gru_layers = (layers.GRULayer, c_layers.MIGRULayer)
    if layer_type in lstm_layers:
        print 'Using {0}'.format(layer_type)
        name = 'lstm'
        l_r_f = layer_type(
            incoming=input_layer,
            num_units=n_hidden,
            nonlinearity=nonlinearity,
            gradient_steps=gradient_steps,
            name=name,
            **kwargs)
        if bidirectional:
            print 'Using bidirectional network'
            l_r_b = layer_type(
                incoming=input_layer_b,
                num_units=n_hidden,
                nonlinearity=nonlinearity,
                gradient_steps=gradient_steps,
                name=name + '_b',
                backwards=True,
                **kwargs)

    elif layer_type is layers.GRULayer:
        print 'Using {0}'.format(layer_type)
        name = 'gru'
        l_r_f = layer_type(
            incoming=input_layer,
            num_units=n_hidden,
            hidden_update=layers.Gate(nonlinearity=nonlinearity),
            gradient_steps=gradient_steps,
            name=name,
            **kwargs)

        if bidirectional:
            print 'Using bidirectional network'
            l_r_b = layer_type(
                incoming=input_layer_b,
                num_units=n_hidden,
                hidden_update=layers.Gate(nonlinearity=nonlinearity),
                gradient_steps=gradient_steps,
                name=name + '_b',
                backwards=True,
                **kwargs)

    elif layer_type is c_layers.MIGRULayer:
        print 'Using {0}'.format(layer_type)
        name = 'gru'
        l_r_f = layer_type(
            incoming=input_layer,
            num_units=n_hidden,
            hidden_update=c_layers.MIGate(nonlinearity=nonlinearity),
            gradient_steps=gradient_steps,
            name=name,
            **kwargs)
        if bidirectional:
            print 'Using bidirectional network'
            l_r_b = layer_type(
                incoming=input_layer_b,
                num_units=n_hidden,
                hidden_update=c_layers.MIGate(nonlinearity=nonlinearity),
                gradient_steps=gradient_steps,
                name=name + '_b',
                backwards=True,
                **kwargs)

    else:
        print 'Invalid layer_type {0}'.format(layer_type)

    l_concat = l_r_f
    out_shape = n_hidden

    if bidirectional:
        print 'Concatenating Forward and Backward recurrent layers'
        l_concat = layers.ConcatLayer((l_concat, l_r_b), axis=-1)
        out_shape = out_shape + n_hidden
    l_re = layers.ReshapeLayer(l_concat, (-1, out_shape), name='reshape')

    if loss_type == 'MSE':
        print 'Using MSE'
    l_d = layers.DenseLayer(l_re, output_dim,
                            nonlinearity=None,
                            name='dense')

    if skip_connection:
        # Combine input_to_output and hidden to output layers
        l_output = lasagne.layers.ElemwiseSumLayer([l_in_to_out, l_d])

    else:
        l_output = l_d

    if kwargs.get('only_return_final', False):
        out_shape = (b_size, 1, output_dim)
    else:
        out_shape = (b_size, seqlen, output_dim)

    l_out = layers.ReshapeLayer(l_output, out_shape)

    deterministic_out = layers.get_output(l_out, deterministic=True)
    deterministic_out.name = 'deterministic out'
    stochastic_out = layers.get_output(l_out)
    stochastic_out.name = 'stochastic out'

    params = layers.get_all_params(l_out, trainable=True)

    if layer_type in lstm_layers:
        # Get regularizable parameters of the LSTM
        reg_params_norm = [l_r_f.W_in_to_cell, l_r_f.W_in_to_forgetgate,
                           l_r_f.W_in_to_ingate, l_r_f.W_in_to_outgate]
        reg_params_rec = [l_r_f.W_hid_to_cell, l_r_f.W_hid_to_forgetgate,
                          l_r_f.W_hid_to_ingate, l_r_f.W_hid_to_outgate]
        if bidirectional:
            reg_params_norm += [l_r_b.W_in_to_cell, l_r_b.W_in_to_forgetgate,
                                l_r_b.W_in_to_ingate, l_r_b.W_in_to_outgate]
            reg_params_rec += [l_r_b.W_hid_to_cell, l_r_b.W_hid_to_forgetgate,
                               l_r_b.W_hid_to_ingate, l_r_b.W_hid_to_outgate]
    elif layer_type in gru_layers:
        # Get regularizable parameters of the GRU
        reg_params_norm = [l_r_f.W_in_to_updategate, l_r_f.W_in_to_resetgate,
                           l_r_f.W_in_to_hidden_update]
        reg_params_rec = [l_r_f.W_hid_to_updategate, l_r_f.W_hid_to_resetgate,
                          l_r_f.W_hid_to_hidden_update]

        if bidirectional:
            reg_params_norm += [
                l_r_b.W_in_to_updategate, l_r_b.W_in_to_resetgate,
                l_r_b.W_in_to_hidden_update]
            reg_params_rec += [
                l_r_b.W_hid_to_updategate, l_r_b.W_hid_to_resetgate,
                l_r_b.W_hid_to_hidden_update]

    if wl2 > 0:
        print 'Using L2 norm regularization'
        weight_reg = wl2 * (sum([T.mean(p ** 2) for p in reg_params_norm]) +
                            T.mean(l_d.W ** 2))

        if skip_connection:
            weight_reg += wl2 * T.mean(l_in_to_out.W ** 2)

    else:
        weight_reg = 0.

    if wl1 > 0:
        print 'Using L1 norm regularization'
        weight_reg += wl1 * (sum([T.mean(p ** 2) for p in reg_params_norm]) +
                             T.mean(l_d.W))
        if skip_connection:
            weight_reg += wl1 * T.mean(abs(l_in_to_out.W))

    if r_reg_coeff > 0:
        print 'Using hid to hid eigenvalue regularization'
        weight_reg += r_reg_coeff * sum(
            [T.mean((T.nlinalg.eigh(p)[0] - 1.) ** 2) for p in reg_params_rec])

    stochastic_loss = (lasagne.objectives.squared_error(stochastic_out, target).mean() +
                       weight_reg)

    stochastic_loss.name = 'stochastic MSE (regularized)'

    deterministic_loss = T.mean(
        lasagne.objectives.squared_error(deterministic_out, target))
    deterministic_loss.name = 'MSE'

    updates = lasagne.updates.rmsprop(
        stochastic_loss, params, learning_rate=learning_rate)

    train_loss = [stochastic_loss]
    valid_loss = [deterministic_loss]

    return dict(l_in=l_in, l_out=l_out,
                train_loss=train_loss,
                valid_loss=valid_loss,
                target=target, updates=updates,
                predictions=deterministic_out,
                gradient_steps=gradient_steps,
                model_type='RNN')
