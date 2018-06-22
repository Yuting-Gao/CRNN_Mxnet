# pylint:skip-file

import sys
#sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import resnet
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def resnet(data):
    conv1 = mx.symbol.Convolution(name='conv1',data=data, kernel=(3,3), num_filter=64,pad=(1,1))
    bn1 = mx.sym.BatchNorm(name='batchnorm1',data=conv1, fix_gamma=False)
    relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
    conv2 = mx.symbol.Convolution(name='conv2',data=relu1, kernel=(3,3), num_filter=64,pad=(1,1))
    shortcut1 = mx.sym.Convolution(data=data, num_filter=64, kernel=(1,1), no_bias=True, name='short_cut1')
    group1=conv2 + shortcut1                                     
    relu2 = mx.symbol.Activation(data=group1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv3 = mx.symbol.Convolution(name='conv3',data=pool1, kernel=(3,3), num_filter=128,pad=(1,1))
    bn2 = mx.sym.BatchNorm(name='batchnorm2',data=conv3, fix_gamma=False)
    relu3 = mx.symbol.Activation(data=bn2, act_type="relu")
    conv4 = mx.symbol.Convolution(name='conv4',data=relu3, kernel=(3,3), num_filter=128,pad=(1,1))
    shortcut2 = mx.sym.Convolution(data=pool1, num_filter=128, kernel=(1,1), no_bias=True, name='short_cut2')
    group2=conv4 + shortcut2      
    relu4 = mx.symbol.Activation(data=group2, act_type="relu")   
    pool2 = mx.symbol.Pooling(data=relu4, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv5 = mx.symbol.Convolution(name='conv5',data=pool2, kernel=(3,3), num_filter=256,pad=(1,1))
    bn3 = mx.sym.BatchNorm(data=conv5, fix_gamma=False,name='batchnorm3')
    relu5 = mx.symbol.Activation(data=bn3, act_type="relu")
    conv6 = mx.symbol.Convolution(name='conv6',data=relu5, kernel=(3,3), num_filter=256,pad=(1,1))
    shortcut3 = mx.sym.Convolution(data=pool2, num_filter=256, kernel=(1,1), no_bias=True, name='short_cut3')
    group3=conv6 + shortcut3     
    relu6 = mx.symbol.Activation(data=group3, act_type="relu")

    pool3 = mx.symbol.Pooling(data=relu6, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv7 = mx.symbol.Convolution(name='conv7',data=pool3, kernel=(1,1), num_filter=512,pad=(0,0))
      
    return conv7

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)

def multilayer_bi_lstm_unroll_new(seq_len,
                num_hidden, num_label,batch_size,num_lstm_layer,dropout=0):
    shape = {"data" : (32, 3, 32, 32)}
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    from importlib import import_module
    resnet = import_module('resnet')

    conv=resnet.get_symbol(2, 34, '3,32,'+str(seq_len*8))
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len,axis=3, squeeze_axis=1)
    forward_param_cells = []
    backward_param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("f_l%d_i2h_weight" % i),
                                             i2h_bias=mx.sym.Variable("f_l%d_i2h_bias" % i),
                                             h2h_weight=mx.sym.Variable("f_l%d_h2h_weight" % i),
                                             h2h_bias=mx.sym.Variable("f_l%d_h2h_bias" % i)))
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("b_l%d_i2h_weight" % i),
                                             i2h_bias=mx.sym.Variable("b_l%d_i2h_bias" % i),
                                             h2h_weight=mx.sym.Variable("b_l%d_h2h_weight" % i),
                                             h2h_bias=mx.sym.Variable("b_l%d_h2h_bias" % i)))

    last_states.append(LSTMState(c=mx.sym.Variable("f_l%d_init_c" % 0),h=mx.sym.Variable("f_l%d_init_h" % 0)))
    last_states.append(LSTMState(c=mx.sym.Variable("b_l%d_init_c" % 0),h=mx.sym.Variable("b_l%d_init_h" % 0)))
    last_states.append(LSTMState(c=mx.sym.Variable("f_l%d_init_c" % 1),h=mx.sym.Variable("f_l%d_init_h" % 1)))
    last_states.append(LSTMState(c=mx.sym.Variable("b_l%d_init_c" % 1),h=mx.sym.Variable("b_l%d_init_h" % 1)))
    
    assert(len(last_states) == num_lstm_layer * 2)
    hidden_all = []
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param_cells[0],
                          seqidx=seqidx, layeridx=0)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)
    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden =mx.sym.Flatten(data=column_features[k])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param_cells[0],
                          seqidx=k, layeridx=0)
        hidden = next_state.h
        last_states[1] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    forward_hidden = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=hidden_all[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[2],
                          param=forward_param_cells[1],
                          seqidx=seqidx, layeridx=1)
        hidden = next_state.h
        last_states[2] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        forward_hidden.append(hidden)
    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden =mx.sym.Flatten(data=hidden_all[k])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[3],
                          param=backward_param_cells[1],
                          seqidx=k, layeridx=1)
        hidden = next_state.h
        last_states[3] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1',data=hidden_concat, num_hidden=37) 
    softmax_pred = pred 
    pred = mx.sym.Reshape(data=pred, shape=(-4,seq_len,-1, 0))
    loss = mx.contrib.sym.ctc_loss(data=pred, label=label)
    ctc_loss = mx.sym.MakeLoss(loss)
    
    softmax_class = mx.symbol.SoftmaxActivation(data=softmax_pred)
    softmax_loss = mx.sym.BlockGrad(softmax_class)
    
    sm = mx.sym.Group([softmax_loss,ctc_loss])
    return sm
