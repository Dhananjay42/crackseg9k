from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from torchvision import transforms
from dataloaders import custom_transforms as tr
import cv2
import numpy as np
import os
from utils.metrics import Evaluator

model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn= None,
                        freeze_bn=False)

model = torch.nn.DataParallel(model, device_ids=[0])
patch_replication_callback(model)
model = model.cuda()

checkpoint = torch.load('/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/run/crack/deeplab-resnet/final/checkpoint.pth.tar')
model.module.load_state_dict(checkpoint['state_dict'])

model.eval()
print('Model loaded')
composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

evaluator = Evaluator(2)
print('Evaluator Loaded')


image_folder_path = '/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/crack/JPEGImages/'
label_folder_path = '/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/crack/SegmentationClass/'

file = open('Webbed.txt', 'r')

for line in file:        
    img = Image.open(image_folder_path + line[:-1]).convert('RGB')
    img_label = Image.open(label_folder_path + line[:-1])
    

    sample = {'image': img, 'label': img_label}
    sample = composed_transforms(sample)
    img = sample['image']
    target = sample['label']
    img = img.cuda()
    # img = torch.unsqueeze(img, 0)
    img = img.repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(img)
    
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().numpy()
                
    evaluator.add_batch(target, pred[0])
        
mIoU = evaluator.Mean_Intersection_over_Union()
F1 = evaluator.F1_Score()

print('miou: ', mIoU)
print('F1: ', F1)
