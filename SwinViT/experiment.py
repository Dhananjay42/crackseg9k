import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import BCELoss
from torchvision.transforms.transforms import CenterCrop
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_crack
from config import get_config
from dataloader import *
from utils import DiceLoss


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='crack', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=448, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = get_config(args)
    
    dataset_name = args.dataset

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    
    args.num_classes = 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)
    db_train = CrackSegDataset(partition = "train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[448,448])]))
    
    trainloader = DataLoader(db_train, batch_size= 4, shuffle=True, num_workers=8, pin_memory=True)
    
    # for sample in trainloader:
               
    #     img , mask = sample["img"].cuda(),sample["mask"].cuda()
    #     out = net(img)
    #     dice_loss = DiceLoss(1)
    #     CE_loss = BCELoss()
    #     print(dice_loss(out,mask))
    #     print(CE_loss(torch.nn.Sigmoid()(out[:,0]),mask.float()))
       
        
    #     break
   
   


    trainer = {'crack' : trainer_crack}
    trainer[dataset_name](args, net, args.output_dir)
