#coding=utf-8
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm
from model import CenterLossModel,loadCheckpoint
from eval_metrics import evaluate
from logger import Logger
from LFWDataset import LFWDataset
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test,get_time,AverageMeter,plot_roc
import json
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")

def main():
    test_display_triplet_distance= True
    print('Number of Classes:{}'.format(len(train_dir.classes)))

    checkpoint = None
    if args.resume:
        checkpoint=loadCheckpoint(args);

    model = CenterLossModel(embedding_size=args.embedding_size,num_classes=args.num_classes,checkpoint=checkpoint)

    if args.cuda:
        model = model.cuda();
        print("now gpus are:" + str(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print("using cpu");
    if args.cuda and len(device_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = create_optimizer(model, args.lr) #choise fron adam adagrad sgd

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(model, optimizer, epoch)
        #if test_display_triplet_distance:
          #display_triplet_distance_test(model,test_loader,args.log_dir+"/test_{}".format(epoch))



def train(model, optimizer, epoch):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    top1 = AverageMeter()
    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data,label= data.cuda(),label.cuda();
        data_v = Variable(data)
        target_var = Variable(label)
        # we put loss in forward propagation to avoid gpus'loading are not balancing
        prec,loss= model(data_v,target_var,args)
        #the result is list because we use multi-gpu
        prec=prec.mean();
        loss = loss.mean();
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        #adjust_learning_rate(optimizer)
        logger.log_value('total_loss', loss.item()).step()
        assert  prec<=100;
        top1.update(prec)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t '
                'Train Prec@1: {:.2f}% (mean: {:.2f}%)'.format(
                    epoch, batch_idx * len(data_v), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),float(top1.val), float(top1.avg)))

        if batch_idx % args.val_interval == 0:  # we validate on LFW dataset every val_interval
            accuracy=validate(model,epoch);
            model.train()

        if batch_idx % args.save_interval == 0 and batch_idx!=0:  # we save every save_interval
            if args.cuda and len(device_ids)>1:
                torch.save({'epoch': epoch + 1,
                            'state_dict': model.module.state_dict(),
                            'centers': model.module.centers},
                           '{}/checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir, get_time(), epoch, accuracy))
            else:
                torch.save({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'centers': model.centers},
                           '{}/checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir, get_time(), epoch,accuracy))

            print("=> saving model:checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth".format(get_time(),epoch,accuracy))

    logger.log_value('Train Prec@1 ',float(top1.avg))
    # do checkpointing
    if args.cuda and len(device_ids)>1:
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'centers': model.module.centers},
                   '{}/checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir, get_time(), epoch, accuracy))
    else:
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'centers': model.centers},
                   '{}/checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir, get_time(), epoch,
                                                                      accuracy))
    print("=> saving model:checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth".format(get_time(), epoch, accuracy))


def validate(model,epoch):
    model.eval()
    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))

    for batch_idx, (data_a, data_p, label) in pbar:  #label这里是 0 1
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), Variable(data_p, volatile=True), Variable(label)

        out_a = model(data_a,None,None)
        out_p = model(data_p,None,None)

        #one batch dists
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
            
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    
    tpr, fpr, accuracy, best_threshold = evaluate(distances,labels)
    print('\n\33[91mTest set: Accuracy: {:.8f} best_threshold: {:.2f}\33[0m'.format(np.mean(accuracy),best_threshold))
    logger.log_value('Test Accuracy', np.mean(accuracy))
    plot_roc(fpr,tpr,args.log_dir,figure_name="roc_test_epoch_{}.png".format(epoch))
    return np.mean(accuracy);




def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
    args = parser.parse_args()
    #read args from json file
    jsonPath='training_center.json'
    if os.path.isfile(jsonPath):
        with open(jsonPath,'r') as trainParams:
            params=json.load(trainParams)
            if 'dataroot' in params:
                args.dataroot =params['dataroot']
            if 'lfw_dir' in params:
                args.lfw_dir =params['lfw_dir']
            if 'lfw_pairs_path' in params:
                args.lfw_pairs_path =params['lfw_pairs_path']
            if 'log_dir' in params:
                args.log_dir =params['log_dir']
            if 'resume' in params:
                args.resume =params['resume']
            if 'start_epoch' in params:
                args.start_epoch =params['start_epoch']
            if 'epochs' in params:
                args.epochs =params['epochs']
            if 'center_loss_weight' in params:
                args.center_loss_weight =params['center_loss_weight']
            if 'alpha' in params:
                args.alpha =params['alpha']
            if 'embedding_size' in params:
                args.embedding_size =params['embedding_size']
            if 'batch_size' in params:
                args.batch_size =params['batch_size']
            if 'test_batch_size' in params:
                args.test_batch_size =params['test_batch_size']
            if 'lr' in params:
                args.lr =params['lr']
            if 'beta1' in params:
                args.beta1 =params['beta1']
            if 'lr_decay' in params:
                args.lr_decay =params['lr_decay']
            if 'wd' in params:
                args.wd =params['wd']
            if 'optimizer' in params:
                args.optimizer =params['optimizer']
            if 'no_cuda' in params:
                args.no_cuda =params['no_cuda']
            if 'gpu_id' in params:
                args.gpu_id =params['gpu_id']
            if 'seed' in params:
                args.seed =params['seed']
            if 'log_interval' in params:
                args.log_interval =params['log_interval']
            if 'num_workers' in params:
                args.num_workers =params['num_workers']
            if 'val_interval' in params:
                args.val_interval=params['val_interval']
            if 'save_interval' in params:
                args.save_interval = params['save_interval']

    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # must assign os.environ['CUDA_VISIBLE_DEVICES'] before,  or torch.cuda.is_available() is False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device_ids = range(torch.cuda.device_count());

    np.random.seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.cuda:
        cudnn.benchmark = True

    LOG_DIR = args.log_dir + '/run-optim_{}-lr{}-wd{}-center{}-vggface'.format(args.optimizer, args.lr, args.wd,args.center_loss_weight)
    # create logger
    logger = Logger(LOG_DIR)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': args.num_workers}
    l2_dist = PairwiseDistance(2)

    transform = transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                                   std = [ 0.5, 0.5, 0.5 ])
                         ])

    train_dir = ImageFolder(args.dataroot,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dir,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                         transform=transform), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.num_classes = len(train_dir.classes)

    main()
