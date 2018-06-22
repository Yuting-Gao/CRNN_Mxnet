# -*- coding: utf-8 -*-
import sys, random
import numpy as np
import mxnet as mx
from text_lstm import multilayer_bi_lstm_unroll_new
from io import BytesIO
import glob
import cPickle
import json
import cv2
import numpy as np
from PIL import Image
import shutil
import argparse
from time import time
reload(sys)
sys.setdefaultencoding('utf8')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, help='The image root you saved')
    parser.add_argument('--image_name', type=str, help='The image name')
    parser.add_argument('--model', type=str, help='Model prefix')
    parser.add_argument('--epoch', type=int, help='Model epoch')
    return parser.parse_args()

def ctc_label(p):
    ret = []
    p1 = [0] + list(p)
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def get_data_batch(data_root,path):
    data=[]
    base_hight=32
    max_ratio=25
    img=cv2.imread(data_root+'/'+path)
    shape=img.shape
    hight=shape[0]
    width=shape[1]
    ratio=(1.0*width/hight)
    if ratio>max_ratio:
        ratio=max_ratio
    if ratio<1:
        ratio=1
    img=cv2.resize(img,(int(32*ratio),32))
    hight=32
    width=int(32*ratio)
    assert hight==base_hight
    img=np.transpose(img,(2,0,1))
    if width % hight !=0:
        padding_ratio=(min(int(ratio+1),max_ratio))
        new_img=np.zeros((3,base_hight,base_hight*padding_ratio))
        for i in range(3):
            padding_value = int(np.mean(img[i][:][-1]))
            z=np.ones((base_hight,base_hight*padding_ratio-width))*padding_value
            new_img[i]=np.hstack((img[i],z))
        data.append(new_img)
    else:
        data.append(img)

    return np.array(data)

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


def sym_gen(seq_len):
    return multilayer_bi_lstm_unroll_new(seq_len,num_hidden=num_hidden,num_label = num_label,batch_size=BATCH_SIZE,num_lstm_layer=2,dropout=0.),('data','f_l0_init_c','f_l0_init_h','f_l1_init_c','f_l1_init_h','b_l0_init_c','b_l1_init_c','b_l0_init_h','b_l1_init_h',), ('label',)
     
if __name__ == '__main__':
    args = init_args()
    
    BATCH_SIZE=1
    num_hidden=256
    num_lstm_layer=2
    num_label=25
    contexts=[mx.gpu(0)]
    prefix='model/' + args.model
    n_epoch_load=args.epoch

    f_init_c = [('f_l%d_init_c'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    f_init_h = [('f_l%d_init_h'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    b_init_c = [('b_l%d_init_c'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    b_init_h = [('b_l%d_init_h'% l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = f_init_c + f_init_h + b_init_c + b_init_h
	
    init_state_names = [x[0] for x in init_states]
    init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
    provide_data = [('data', (BATCH_SIZE, 3, 32, 800))] + init_states
    provide_label = [('label', (BATCH_SIZE, num_label))]	
    model = mx.mod.BucketingModule(sym_gen=sym_gen,default_bucket_key=100,context=contexts)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, n_epoch_load)
    model.bind(data_shapes=provide_data,label_shapes=provide_label, for_training=False)
    model.init_params(arg_params=arg_params, aux_params=aux_params)

    data_root = args.image_root
    image = args.image_name
    data = get_data_batch(data_root, image)
    
    hight = data.shape[2]
    width = data.shape[3]
    bucket_key = width / hight * 4
    
    label = np.array([[0] * 10])
    data_all = [mx.nd.array(data)] + init_state_arrays
    label_all = [mx.nd.array(label)]
    data_names = ['data'] + init_state_names
    label_names = ['label']
    data_batch = SimpleBatch(data_names, data_all, label_names, label_all, bucket_key)
    model.forward(data_batch, is_train=False)
    preds = model.get_outputs()
    pred_label = preds[0].asnumpy().argmax(axis=1)
    p = ctc_label(pred_label)
    idx2c = json.load(open('idx2char.json', 'r'))
    idx2c = json.JSONDecoder().decode(idx2c)
    ret = ''
    for i in p:
	    ret += idx2c[str(i)].encode('utf-8')
    print (ret)
