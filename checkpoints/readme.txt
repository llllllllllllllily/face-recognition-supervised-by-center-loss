put checkpoints there
naming rule：like checkpoint_Acc0.92.pth,must have 'Acc' and number(like 0.8,0.9...). the format must be '.pth'
when there are not only one checkpoint. the code will choice which have the max number behind 'Acc'

the directory which name like 'run-optim_sgd-lr0.1-wd0.0-embeddings512-center0.005-vggface'  is the logdir of tensorboard
you can check the training process using this command on linux server:
$ tensorboard --logdir="checkpoints/run-optim_sgd-lr0.1-wd0.0-embeddings512-center0.005-vggface"
the default port is 6006
then you must use your computer ssh the linux server use : ssh -L 6006:127.0.0.1:6006 root@192.168.1.10
(replace username and IP to your server's)
Finally,you can check the training process using chrome  url: 127.0.0.1:6006
