#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import random
import json

flabels=open('gt.txt','r')
fl=flabels.readlines()
ll=len(fl)

data_list={}
max_length=25

for i in range(ll):
    content=fl[i].strip()
    name=content.split(' ')[0]
    value=content.split(' ')[1]
    if len(value)>max_length:
        print content
        continue
    data_list[name]=value
    
keys=data_list.keys()
random.shuffle(keys)
keys_length=len(keys)
split=int(round(0.9*keys_length))

keys_train=keys[0:split]
keys_valid=keys[split:]
list_train=open('train.lst','w')
list_valid=open('valid.lst','w')

char2idx=json.load(open('char2idx.json','r'))
char2idx=json.JSONDecoder().decode(char2idx)
idx2char=json.load(open('idx2char.json','r'))
idx2char=json.JSONDecoder().decode(idx2char)

counter=0
for key in keys_train:
    value=data_list[key]
    value=value.decode('utf-8')
    label=''
    for v in value:
        char_id=char2idx[v]
        label+=str(char_id)+'\t'
    if len(value) < max_length:
        for i in range(max_length-len(value)):
            label+='0'+'\t'
    line=str(counter)+'\t'+label+key
    list_train.write(line)
    list_train.write('\n')
    counter+=1

counter=0
for key in keys_valid:
    value=data_list[key]
    value=value.decode('utf-8')
    label=''
    for v in value:
        char_id=char2idx[v]
        label+=str(char_id)+'\t'
    if len(value) < max_length:
        for i in range(max_length-len(value)):
            label+='0'+'\t'
    line=str(counter)+'\t'+label+key
    list_valid.write(line)
    list_valid.write('\n')
    counter+=1

list_train.close()
list_valid.close()