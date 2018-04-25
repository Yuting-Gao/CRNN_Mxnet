# CRNN_MXnet
This repo contains MXnet verison of Convolutional Recurrent Neural Network(CRNN) for scene text recognition.

CRNN is proposed in "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"

Paper link: http://arxiv.org/abs/1507.05717  Original torch version: https://github.com/bgshih/crnn

The architecture is this repo is: ResNet-34, 2-layer BLSTM and CTC. In order to reduce LSTM cost and tackle variable text length, Bucketing is applied.

## Mxnet Version
Use the Mxnet CTC loss layer in contrib package, CTCLoss was added in MXnet 0.10.0. 

So the MXnet Version must >= 0.10.0. I use 0.10.1 and it works.

## Training

1.put your own dataset in a folder, e.g.icdar2013

2.put text label into a txt file, eg.icdar2013_gt.txt, the content should as follows, image name and label should be separated by blank.
```python
word_1.png proper
word_2.png food
word_3.png pronto
```
3.use make_train_list.py to generate train.lst and valid.lst 
```python
python make_train_list.py
```

4.training from scratch
```python
mkdir model
python text_deep_ocr_bucketing.py --data_root icdar2013 --save_prefix ic13
```
Your can also resume training procedure using
```python
python text_deep_ocr_bucketing_resume.py --data_root icdar2013 --save_prefix ic13_resume --load_prefix ic13 --epoch 1
```

## Test
I upload one model in model folder that is pretrained on a subset of Synth90k and finetuned on icdar2013, you can test image on it.

I test it on icdar2013 test word dataset and the accuracy is around 83%.
```python
python predict.py --image_root . --image_name test.png --model ic13 --epoch 1
```
I will upload a more robust model soon and you can also resume training on this model.