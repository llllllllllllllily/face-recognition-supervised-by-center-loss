#coding=utf-8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import json
from torch.autograd import Variable,Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import TripletLossModel,loadCheckpoint
from eval_metrics import evaluate
from logger import Logger
from TripletFaceDataset import TripletFaceDataset
from LFWDataset import LFWDataset
from PIL import Image
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test,get_time,plot_roc
import collections
import warnings
warnings.filterwarnings("ignore")



class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class TripletMarginLoss(nn.Module):
    """Triplet loss function.
       inherit from Module,it can autograd
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

def main():
    test_display_triplet_distance = True
    print('Number of Classes:{}'.format(len(train_dir.classes)))

    checkpoint=None;
    if args.resume:
        if os.path.isdir(args.log_dir):
            checkpoint = loadCheckpoint(args);
        else:
            print('=> no found dir {}'.format(args.log_dir))

    model = TripletLossModel(embedding_size=args.embedding_size,num_classes=len(train_dir.classes),checkpoint=checkpoint);
    device_ids = range(torch.cuda.device_count());

    if args.cuda:
        model.cuda()
        print("now gpus are:" + str(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print("using cpu")
    if args.cuda and len(device_ids)>1:
        model=nn.DataParallel(model,device_ids=device_ids)

    optimizer = create_optimizer(model, args.lr)

    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)

        #if test_display_triplet_distance:
            #display_triplet_distance(model,train_loader,LOG_DIR+"/train_{}".format(epoch))
            #display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(epoch))


def train(train_loader, model, optimizer, epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    labels, distances = [], []

    for batch_idx, (data_a, data_p, data_n,label_p,label_n) in pbar:
        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()

        # compute output
        #triplet_loss,distsAN,distsAP,len_hard_triplets0= model(data_a,data_p,data_n,label_p,label_n,args)
        out_a, out_p, out_n= model(data_a), model(data_p), model(data_n)
        #because the special loss function,we can't put loss in forward propagation

        # Choose the hard negatives
        d_p = l2_dist.forward(out_a, out_p)
        d_n = l2_dist.forward(out_a, out_n)
        all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue

        out_selected_a = out_a[hard_triplets]
        out_selected_p = out_p[hard_triplets]
        out_selected_n = out_n[hard_triplets]

        # we only use triplet loss,not combine with softmax there
        #selected_data_a = Variable(torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).cuda())
        #selected_data_p = Variable(torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).cuda())
        #selected_data_n = Variable(torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).cuda())

        #selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
        #selected_label_n= torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
        triplet_loss = TripletMarginLoss(args.margin).forward(out_selected_a, out_selected_p, out_selected_n)

        #cls_a = model.forward_classifier(selected_data_a)
        #cls_p = model.forward_classifier(selected_data_p)
        #cls_n = model.forward_classifier(selected_data_n)

        #criterion = nn.CrossEntropyLoss()
        #predicted_labels = torch.cat([cls_a,cls_p,cls_n])
        #true_labels = torch.cat([Variable(selected_label_p.cuda()),Variable(selected_label_p.cuda()),Variable(selected_label_n.cuda())])

        #cross_entropy_loss = criterion(predicted_labels.cuda(),true_labels.cuda())

        #loss = cross_entropy_loss + triplet_loss

        # compute gradient and update weights
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        #adjust_learning_rate(optimizer)

        logger.log_value('triplet_loss', triplet_loss.item()).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    triplet_loss.item(),len(hard_triplets[0])))

        dists = l2_dist.forward(out_selected_a,out_selected_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))

        dists = l2_dist.forward(out_selected_a,out_selected_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

        if batch_idx % args.val_interval == 0:  #每val_interval 个batch 一验证
            testaccuracy=validate(model,epoch);
            model.train()
        if batch_idx % args.save_interval == 0: #and batch_idx!=0:  # 每val_interval 个batch 一验证
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                       '{}/triplet_loss_checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir, get_time(), epoch, testaccuracy))
            print('=>saving model:triplet_loss_checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(get_time(), epoch, testaccuracy))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)

    print('\n\33[91mTrain set: Accuracy: {:.8f}\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Train Accuracy', np.mean(accuracy))

    plot_roc(fpr,tpr,figure_name="roc_train_epoch_{}.png".format(epoch))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/triplet_loss_checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(args.log_dir,get_time(), epoch,testaccuracy))
    print('=>saving model:triplet_loss_checkpoint_{}_epoch{}_lfwAcc{:.4f}.pth'.format(get_time(), epoch, testaccuracy))


def validate(model, epoch):
    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)
        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, best_threshold= evaluate(distances,labels)
    print('\n\33[91mTest set: Accuracy: {:.8f} best_threshold:{:.3f}\33[0m'.format(np.mean(accuracy),best_threshold))
    logger.log_value('Test Accuracy', np.mean(accuracy))

    plot_roc(fpr,tpr,args.log_dir,figure_name="roc_test_epoch_{}.png".format(epoch))
    return accuracy.mean();



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
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
    args = parser.parse_args()
    jsonPath = 'training_triplet.json'
    if os.path.isfile(jsonPath):
        with open(jsonPath, 'r') as trainParams:
            # print(trainParams)
            params = json.load(trainParams)
            if 'dataroot' in params:
                args.dataroot = params['dataroot']
            if 'lfw_dir' in params:
                args.lfw_dir = params['lfw_dir']
            if 'lfw_pairs_path' in params:
                args.lfw_pairs_path = params['lfw_pairs_path']
            if 'log_dir' in params:
                args.log_dir = params['log_dir']
            if 'resume' in params:
                args.resume = params['resume']
            if 'start_epoch' in params:
                args.start_epoch = params['start_epoch']
            if 'epochs' in params:
                args.epochs = params['epochs']
            if 'center_loss_weight' in params:
                args.center_loss_weight = params['center_loss_weight']
            if 'alpha' in params:
                args.alpha = params['alpha']
            if 'embedding_size' in params:
                args.embedding_size = params['embedding_size']
            if 'batch_size' in params:
                args.batch_size = params['batch_size']
            if 'test_batch_size' in params:
                args.test_batch_size = params['test_batch_size']
            if 'lr' in params:
                args.lr = params['lr']
            if 'beta1' in params:
                args.beta1 = params['beta1']
            if 'lr_decay' in params:
                args.lr_decay = params['lr_decay']
            if 'wd' in params:
                args.wd = params['wd']
            if 'optimizer' in params:
                args.optimizer = params['optimizer']
            if 'no_cuda' in params:
                args.no_cuda = params['no_cuda']
            if 'gpu_id' in params:
                args.gpu_id = params['gpu_id']
            if 'seed' in params:
                args.seed = params['seed']
            if 'log_interval' in params:
                args.log_interval = params['log_interval']
            if 'num_workers' in params:
                args.num_workers = params['num_workers']
            if 'val_interval' in params:
                args.val_interval = params['val_interval']
            if 'save_interval' in params:
                args.save_interval = params['save_interval']
            if 'n_triplets' in params:
                args.n_triplets = params['n_triplets']
            if 'margin' in params:
                args.margin = params['margin']

    # order to prevent any memory allocation on unused GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.cuda:
        cudnn.benchmark = True

    LOG_DIR = args.log_dir + '/run-optim_{}-lr{}-wd{}-embeddings{}-triplet-vggface'.format(args.optimizer, args.lr,
                                                                                            args.wd,
                                                                                            args.embedding_size)
    # create logger
    logger = Logger(LOG_DIR)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': args.num_workers}
    l2_dist = PairwiseDistance(2)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    #train_dir we need to make n_triplets triplets
    train_dir = TripletFaceDataset(dir=args.dataroot, n_triplets=args.n_triplets, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dir,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=args.lfw_dir, pairs_path=args.lfw_pairs_path,
                   transform=transform),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.num_classes = len(train_dir.classes)
    args.val_interval = len(train_loader) / args.batch_size * args.val_interval
    args.log_interval = len(train_loader) / args.batch_size * args.log_interval
    args.save_interval = len(train_loader) / args.batch_size * args.save_interval

    main()