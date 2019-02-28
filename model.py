#coding=utf-8
import torch
import torch.nn as nn
from torchvision.models import vgg16,resnet18
import os
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter

class TripletLossModel(nn.Module):
    def __init__(self,embedding_size,num_classes,checkpoint):
        super(TripletLossModel, self).__init__()
        self.featuresvgg16 = vgg16(pretrained=False).features
        self.featuresLayer = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embedding_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, num_classes),
        )
        self.embedding_size = embedding_size

        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            param = param.data
                        own_state[name].copy_(param)


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward_GetFeature(self, x):
        x = self.featuresvgg16(x)
        x = x.view(x.size(0), -1)
        x = self.featuresLayer(x) #512
        x=self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        return x*alpha

    def forward(self,data):
        # the loss is special so couldn't put loss in there.
        return self.forward_GetFeature(data);


    def forward_classifier(self, x):
        features= self.forward_GetFeature(x)
        res = self.model.classifier(features)
        return res






##======================================================================================================================






class CenterLossModel(nn.Module):
    def __init__(self,embedding_size,num_classes,checkpoint):
        super(CenterLossModel, self).__init__()
        self.featuresvgg16 = vgg16(pretrained=False).features
        self.featuresLayer=nn.Sequential(
                nn.Linear(512*3*3, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, embedding_size),
        )
        self.classifier=nn.Sequential( 
                nn.Linear(embedding_size, num_classes),
        )
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        self.num_classes = num_classes

        if checkpoint is not None:
            # Check if there are the same number of classes
            if list(checkpoint['state_dict'].values())[-1].size(0) == num_classes:
                self.load_state_dict(checkpoint['state_dict'])
                assert checkpoint['centers'].shape[0]==num_classes;
                assert checkpoint['centers'].shape[1] == embedding_size;
                self.centers = checkpoint['centers']
            else:
                own_state = self.state_dict()
                for name, param in checkpoint['state_dict'].items():
                    if "classifier" not in name:
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)

    def get_center_loss(self,features,target,args):
        batch_size = target.size(0)
        features_dim = features.size(1)
        target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
        if args.cuda:
            self.centers=self.centers.cuda()
        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0,target_expand)
        criterion = nn.MSELoss()
        center_loss = criterion(features,  centers_batch)

        # next is update center with manual operation .it will be much easier if you put it in optimizer.the code like this:
        '''
        optimizer = optim.SGD([
                                {'params': model.parameters(),'lr':args.lr},
                                {'params': model.centers ,'lr':args.alpha}   # different learning rate
                              ],  momentum = conf.momentum)
        '''
        #numpy's computation must on cpu . if we can replace it by torch .the speed can improve
        diff = centers_batch - features
        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)  # 防止除数为0 加一个很小的数
        #∆c_j =(sum_i=1^m δ(yi = j)(c_j − x_i)) / (1 + sum_i=1^m δ(yi = j))
        diff_cpu = args.alpha * diff_cpu
        if args.cuda:
            diff_cpu.cuda()
        assert self.centers.shape[0] == args.num_classes;
        assert self.centers.shape[1] == args.embedding_size;
        for i in range(batch_size):
            #Update the parameters c_j for each j by c^(t+1)_j = c^t_j − α · ∆c^t_j
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
    
    def forward_GetFeature(self, x):
        x = self.featuresvgg16(x)
        x = x.view(x.size(0), -1)
        x = self.featuresLayer(x)
        x=self.l2_norm(x)
        return x

    '''
    put loss in forward function because if we put it outsise, computate cost function will occupy lots of memory of main GPU which result in out of memory
    however,the other gpus only occupy little memory. so we transfer this task(loss) from main GPU to each gpu, and we only need return loss result to main GPU
    this is a strategy of load balancing
    '''
    def forward(self,x,target_var,args):
        feature512=self.forward_GetFeature(x)
        if target_var is None:
            return feature512;
        classifyResult = self.classifier(feature512)
        center_loss = self.get_center_loss(feature512, target_var,args)
        criterion = nn.CrossEntropyLoss()
        cross_entropy_loss = criterion(classifyResult, target_var)
        #CrossEntropyLoss  contains softmax
        loss = args.center_loss_weight * center_loss + cross_entropy_loss
        prec = accuracy(classifyResult.data, target_var, topk=(1,))
        return prec[0],loss


def loadCheckpoint(args):
    '''
    the strategy to load chechpoint is choose the max index of all checkpoints in lod_dir
    '''
    checkpoint=None;
    checkpointList = os.listdir(args.log_dir)
    if (len(checkpointList) > 0):
        newest = "";
        numMax = 0;
        for fileName in checkpointList:
            if os.path.isdir(os.path.join(args.log_dir, fileName)):
                continue
            houzhui=fileName.split('pth');
            if  len(houzhui)==2:
                num = int(fileName.split('Acc')[-1].split(".")[0])
                if (num >= numMax):
                    newest = fileName;
                    numMax = num;
        checkpointPath = os.path.join(args.log_dir, newest)
        if not os.path.isdir(checkpointPath):
            print('=> loading checkpoint {}'.format(checkpointPath))
            checkpoint = torch.load(checkpointPath)
            args.start_epoch = checkpoint['epoch']
        else:
            print('=> no checkpoint found at {}'.format(args.log_dir))
    else:
        print('=> no checkpoint found at {}'.format(args.log_dir))
    return checkpoint;

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _,pred = output.topk(k=1,dim=1,largest=True, sorted=True)  #
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred.long()))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

