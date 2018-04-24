# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
import numpy as np
import mxnet as mx
print (mx.__version__)
from text_lstm import multilayer_bi_lstm_unroll_new
from io import BytesIO
import argparse
import cv2, random
from text_bucketing_iter import TextIter


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Where you store the dataset')
    parser.add_argument('--save_prefix', type=str, help='Model prefix you want to save')
    return parser.parse_args()

def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret

BATCH_SIZE = 28

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    pred = pred
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(len(pred)/BATCH_SIZE):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':
    args = init_args()
    num_hidden = 256
    num_lstm_layer = 2
    num_epoch = 200
    num_label = 25
    contexts = [mx.gpu(0)]
    def sym_gen(seq_len):
        return multilayer_bi_lstm_unroll_new(seq_len,num_hidden=num_hidden,num_label = num_label,batch_size=BATCH_SIZE,num_lstm_layer=2,dropout=0.),('data','f_l0_init_c','f_l0_init_h','f_l1_init_c','f_l1_init_h','b_l0_init_c','b_l1_init_c','b_l0_init_h','b_l1_init_h',), ('label',)
        
    f_init_c = [('f_l%d_init_c'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    f_init_h = [('f_l%d_init_h'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    b_init_c = [('b_l%d_init_c'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    b_init_h = [('b_l%d_init_h'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = f_init_c + f_init_h + b_init_c + b_init_h

    path='train.lst'
    path_test='valid.lst'
    data_root=args.data_root
    test_root=args.data_root
    
    buckets=[4*i for i in range(1,num_label+1) ]
    data_train=TextIter(path,data_root, BATCH_SIZE, init_states,num_label,buckets=buckets)
    data_val=TextIter(path_test, test_root, BATCH_SIZE,init_states,num_label,buckets=buckets)
    
    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = 100,
        context             = contexts)    
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print ('begin fit')

    prefix='model/' + args.save_prefix
    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.np(Accuracy),
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate': 0.01,
                                'momentum': 0.9,
                                'wd': 0 },
        kvstore             = 'device',
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           =100,
        epoch_end_callback =mx.callback.do_checkpoint(prefix),
        batch_end_callback  = mx.callback.Speedometer(BATCH_SIZE, 20))