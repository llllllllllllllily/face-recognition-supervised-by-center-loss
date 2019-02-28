训练数据集： vggface2
验证数据集： lfw_aligned
都已对齐 大小112*112
损失函数：center_loss  （推荐）
                triplet_loss




训练过程

center_loss ：
论文：A Discriminative Feature Learning Approach for Deep Face Recognition
修改配置文件  training_center.json  

############################## training_center.json ####################################################
   配置项                            默认值                                                             说明

  "dataroot":                      "/data/tee/face/vggface_train/imgByClass",       训练数据集
  "lfw_dir":                         "/data/tee/face/LFW/lfw_aligned",                     验证数据集
  "lfw_pairs_path":              "/data/tee/face/LFW/lfw_pairs.txt",                    triplet对信息
  "log_dir":                         "./checkpoints",                                                 存储及加载checkpoint的地址
  "resume":                        true,                                                                   是否加载checkpoint finetune
  "start_epoch":                  0,                                                                       开始epoch
  "epochs":                        50,                                                                      epoch大小
  "no_cuda":                      false,                                                                   是否使用gpu 使用false,不使用 true
  "gpu_id":                         "0,1",                                                                   gpu id
  "num_workers":               40,                                                                      加载数据集的线程数目
  "seed":                             0,                                                                       随机种子
  "val_interval":                   200,                                                                    间隔多少batch_size 进行验证
  "log_interval":                  10,                                                                      间隔多少batch_size 打印训练信息（loss等） 

  "embedding_size":          512,                                                                     feature 大小（不要改动）
  "batch_size":                   256,                                                                     训练batch_size
  "test_batch_size":            128,                                                                     验证batch_size
 
 "optimizer":                   "sgd",                                                                     优化器类型  sgd /adam/adagrad 任一
  "lr":                                0.001,                                                                    学习率
  "center_loss_weight":     0.005,                                                                    center_loss 占比  total_loss=center_loss_weight * center_loss+cross_entropy_loss   推荐区间0.001-0.01

  "alpha":                         0.5,                                                                        center  学习率
  "beta1":                         0.9,                                                                        adam 参数1
  "lr_decay":                     1e-4,                                                                      adam 参数2
  "wd":                             0.0,                                                                        adam 参数3

############################training_center.json###################################################

如果要用已有模型训练 请将resume=true 同时将checkpoint 放入log_dir中 ； checkpoint 命名规则：模型名_1.pth  _ 后面一定要接数字，代码会选择所有checkpoint中数字最大的加载
开始训练
运行 python  train_center.py





triplet_loss：
论文：FaceNet: A Unified Embedding for Face Recognition and Clustering
修改配置文件  training_triplet.json 

#####################################  training_triplet.json #############################################
#ͬ同上 多出来两个
  "n_triplets":5000000,                                       triplets对数
  "margin":0.5                                                   正负例距离                                           
####################################################################################################

如果要用已有模型训练 请将resume=true 同时将checkpoint 放入log_dir中 ； checkpoint 命名规则：模型名_1.pth  _ 后面一定要接数字，代码会选择所有checkpoint中数字最大的加载
开始训练
运行 python train_triplet.py



tensorboard 查看训练经过

进入 ./checkpoints
输入命令 tensorboard --logdir="./run-optim_sgd-lr0.01-wd0.0-embeddings512-center0.005-vggface"
####"run-optim_sgd-lr0.01-wd0.0-embeddings512-center0.005-vggface" 会根据配置文件随时生成 训哪个用哪个
本地连接服务器端口(默认6006) 查看tensorboard  进入cmd 输入命令
ssh -L 6006:127.0.0.1:6006 intern@192.168.1.10 (这里intern修改为自己的用户名)
打开浏览器输入网址：127.0.0.1:6006
