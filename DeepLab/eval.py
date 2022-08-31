from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from torchvision import transforms
from dataloaders import custom_transforms_feat as tr
import cv2
import numpy as np

model = DeepLab(num_classes=2,
                        backbone='xception_feat',
                        output_stride=16,
                        sync_bn= None,
                        freeze_bn=False)

model = torch.nn.DataParallel(model, device_ids=[0])
patch_replication_callback(model)
model = model.cuda()

checkpoint = torch.load('/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/run/crack/deeplab-xception_feat/experiment_9/checkpoint.pth.tar')
model.module.load_state_dict(checkpoint['state_dict'])

model.eval()
print('Model loaded')
composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

test_images = ['CFD_008', 'CFD_010', 'CFD_012', 'CFD_016', 'CFD_024', 'CFD_031', 'CFD_035', 'CFD_036', 'CFD_050', 'CFD_072']
for rgb_img_path in test_images:
    
    img = Image.open('/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/crack/JPEGImages/' + rgb_img_path + '.jpg').convert('RGB')
    feat = Image.open('/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/crack/atten_3/' + rgb_img_path + '.jpg')

    sample = {'image': img, 'label': img, 'feature': feat}
    sample = composed_transforms(sample)
    img, feat = sample['image'], sample['feature']
    img, feat = img.cuda(), feat.cuda()
    
    # img = torch.unsqueeze(img, 0)
    img = img.repeat(2, 1, 1, 1)
    feat = torch.unsqueeze(feat, 0)
    feat = feat.repeat(2, 1, 1, 1)
    # print('image shape: ', img.shape)
    # print('feature shape: ', feat.shape)
    
    final_input = torch.cat((img, feat), 1)
    with torch.no_grad():
        output = model(final_input)

    pred = torch.max(output[:3], 1)[1].detach().cpu().numpy()
    # print('shape of output: ', pred[0].shape)


    # print('pred[0] max value is: ', np.max(pred[0]))
    cv2.imwrite('paper_images/' + rgb_img_path + '.png', pred[0] * 255)
    # im.save("paper_images/CFD_001.png")

    print('Image saved ', rgb_img_path)