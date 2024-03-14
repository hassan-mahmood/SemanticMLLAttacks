import argparse
from engine import *
from models import *
import sys
import os
import sys
from torch import optim
sys.path.append(os.getcwd())
from voc import *
from torch.optim.lr_scheduler import MultiStepLR
#sys.path.append(os.getcwd())
#from Datasets.Dataset import ImageDataset


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    # train_dataset = voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    # val_dataset = voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    train_dataset = voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    val_dataset = voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    # classnames_path='/mnt/raptor/hassan/data/KG_data/voc/node_names'
    # tree_path='tree_path=/mnt/raptor/hassan/data/KG_data/voc/tree'
    # train_images_folders=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/JPEGImages/','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/']

    # train_labels_files=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/train_labels.h5','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/train_labels.h5']

    # val_labels_files=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/val_labels.h5','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/val_labels.h5']
    # dataset_name='voc'
    # meanstdfile='/mnt/raptor/hassan/data/KG_data/voc/meanstd'

    #train_dataset=ImageDataset(224,classnames_path,tree_path,meanstdfile,train_labels_files,train_images_folders,dataset_name,'train')
    #val_dataset=ImageDataset(224,classnames_path,tree_path,meanstdfile,val_labels_files,train_images_folders,dataset_name,'val')

    num_classes = 35

    # load model
    #model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')

    # define loss function (criterion)
    #criterion = nn.MultiLabelSoftMarginLoss()
    criterion=nn.BCEWithLogitsLoss()
    #criterion=nn.BCELoss()
    # define optimizer
    #optimizer= optim.Adam(model.parameters(), lr=args.lr)#,weight_decay=1e-4)
    optimizer= optim.Adam(model.get_config_optim(args.lr,args.lrp), lr=args.lr)#,weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #scheduler = MultiStepLR(optimizer, milestones=[50,100,200,300,500,700], gamma=0.1)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    #state['save_model_path'] = 'checkpoint/voc2007/'
    #state['save_model_path'] = '/mnt/raptor/hassan/weights/voc/sep/voc_gcn_1/'
    state['save_model_path'] = '/mnt/raptor/hassan/weights/voc/sep/voc_gcn_1/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)#,scheduler)



if __name__ == '__main__':
    main_voc2007()
