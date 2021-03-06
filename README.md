# face-recognition-supervised-by-center-loss
Deep Face Recognition ; pytorch ;  center_loss ; triplet_loss;

[中文](https://blog.csdn.net/weixin_40087578/article/details/88560676)
## Introduction
In this repository,we provide code to train deep face neural network using pytorch. the network was training supervised by center loss. we also provide
triplet loss Method to train the network,but my experients indicate the result is not good comparing  using center loss. so I
recommend you use center loss.

the architecture I used is standard vgg16. you can replace it with other architecture like resnet50,101...  these will have higher accurancy. As for why I use vgg16. maybe.....too young too sample....  if you use resnet50, the accurancy could improve two or three points I guess.

the training data I used is vggface2 which have 8631 identities and 3 million pictures. MS1M have larger pictures. Using MS1M the accurancy would have a higher accurancy comparing vggface2,maybe one point or two.

the test data I used is LFW. 

### My Result:   
vgg16+vggface2:   test accurancy:  0.967     auc:  0.99

![roc and auc](https://github.com/llllllllllllllily/face-recognition-supervised-by-center-loss/blob/master/resource/test_roc.png)

(the code will drawing roc curve in checkpoints when test)

### Environment：
python3, pytorch  

the code can be run under both GPU and CPU

## Data prepare
All face images should be aligned by MTCNN and cropped to 112x112

if you don't alighed,the accurancy would be very low

LFW download path(BaiduNetdisk):

https://pan.baidu.com/s/1eilswvW-qXy3XsHoTeHRzg 

passward：iyd7

the LFW data above is  alighed. vggface2 is too large.You can get it in  https://github.com/deepinsight/insightface. this repo not only provide alighed datasets like vggface2 and mslm, but also alighed code in src/align. You can use the code in src/align to align your own datasets.

you must put the data in the directory "./datasets". And the format is as follows:
```
--datasets
   
   --vggface2
   
        --people name/id
        
         --111.jpg
```        
that is say the images must be grouped by id or name.

## Train
### Using center loss (recommand)
paper: A Discriminative Feature Learning Approach for Deep Face Recognition

1.modify config file -------> training_certer.json

I write all the config parameters in this json file. the explain is as follows:

| parameter  | default value | explain |
| -----------| --------------| --------|
|  dataroot  | "datasets/vggface2"| traing data dir|  
|  lfw_dir  | "datasets/lfw_alighed"| test data dir|
|  lfw_pairs_path  | "datasets/lfw_pairs.txt"| triplet pairs of LFW|
|  log_dir  | "./checkpoints" | is loding pretrained checkpoint|
|  resume  | false | is loding pretrained checkpoint|
|  start_epoch  | 0 | start epoch index|
|  epochs  | 50 | epoch num|
|  no_cuda  | false| is not using gpu.false is use gpu.true is not use gpu |
|  gpu_id  | "0,1"| gpu index|
|  num_workers  | 10| threads num of loding data|
|  seed  | 0| random seed|
|  val_interval  | 200| every $(val_interval) batchs test on test dataset|
|  log_interval  | 10| every $(log_interval) batchs print training message contain loss |
|  save_interval  | 500| every $(save_interval) batchs save checkpoint|
|  batch_size  | 256| traing batch_size|
|  test_batch_size  |128| test batch_size|
|  optimizer  | "sgd"| optimizer in sgd/adam/adagrad|
|  lr  |0.1| learning rate|
|  center_loss_weight  | 0.5| center_loss weight|
|  alpha  |0.5| center_loss learning rate|
|  beta1  | 0.9| adam param1|
|  lr_decay  |1e-4|  adam param2|
|  wd  |0.0|  adam param3|

2. run

```python train_center.py```

### Using triplet loss
paper：FaceNet: A Unified Embedding for Face Recognition and Clustering

1.modify config file -------> training_triplet.json

there are almost the same with training_certer.json, but two more parameters:

| parameter  | default value | explain |
| -----------| --------------| --------|
|  n_triplets  |1000000|  triplet pairs num for training |
|  margin  |0.5|margin in paper FaceNet| 

2.run

```python train_triplet.py```

you can check training process using tensorboard. the specific way is writen in "checkpoints/readme.txt"

## pretrained checkpoints
There is a checkpoint I trained using vgg16. You can use it as pretrained model.

download path(BaiduNetdisk): https://pan.baidu.com/s/1IPdmFy0bFfPt1xqV8S7eDg 

passward：89sf 

## Contact
Email:  18811423186@163.com
vx:     18811423186

## please give me a star. Thank you very much.


