from torchsummary import summary
from modeling.deeplab import *

model = DeepLab(num_classes=2,
                        backbone='xception_feat',
                        output_stride=16,
                        sync_bn= None,
                        freeze_bn=False)

model = torch.nn.DataParallel(model, device_ids=[0])
summary(model, (4, 440, 440))