import tensorflow as tf
import os
import time
import numpy as np
import random
import sys, getopt
import cv2
import datetime
import glob
from sklearn.metrics import f1_score
from segmentation_models import FPN

def normalize(image):
    image = (image / 127.5) - 1
    input_image = tf.clip_by_value(image, -1, 1)
    return image

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    last = FPN('resnet50',  classes=1, activation='sigmoid', input_shape=(None, None, 3), encoder_weights = None)
    x = inputs
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  
    down1 = downsample(64, 4, False)(x)  
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2) 
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) 
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def pix2pix(model, image):
  inp_image = image/127.5 - 1
  inp_image = np.reshape(inp_image, (1,256,256,3)).copy()
  pred = model(inp_image, training = True)
  pred = np.reshape(pred, (256,256)).copy()
  pred = 0.5*pred + 0.5
  return pred

def stitched_pix2pix_1(model, image):
  #use on square image between size 256 and 512
  dim = np.shape(image);
  k = 256

  l = dim[0];
  ind = l - k;

  pred = np.zeros([l,l])

  topleft = pix2pix(model, image[0:k,0:k])
  topright = pix2pix(model, image[0:k,ind:l])
  bottomleft = pix2pix(model, image[ind:l,0:k])
  bottomright = pix2pix(model, image[ind:l,ind:l])
  
  pred[0:ind,0:ind] = topleft[0:ind,0:ind]
  pred[0:ind,k:l] = topright[0:ind, k-ind:l-ind]
  pred[k:l,0:ind] = bottomleft[k-ind:l-ind,0:ind]
  pred[k:l,k:l] = bottomright[k-ind:l-ind,k-ind:l-ind]

  pred[0:ind,ind:k] = 0.5*(topleft[0:ind,ind:k] + topright[0:ind,0:k-ind])
  pred[ind:k,0:ind] = 0.5*(topleft[ind:k,0:ind] + bottomleft[0:k-ind,0:ind])
  pred[ind:k,k:l] = 0.5*(topright[ind:k,k-ind:l-ind] + bottomright[0:k-ind,k-ind:l-ind])
  pred[k:l,ind:k] = 0.5*(bottomleft[k-ind:l-ind,ind:k] + bottomright[k-ind:l-ind,0:k-ind])

  pred[ind:k, ind:k] = 0.25*(topleft[ind:k,ind:k] + topright[ind:k,0:k-ind] + bottomleft[0:k-ind,ind:k] + bottomright[0:k-ind,0:k-ind])

  return pred

def stitched_pix2pix_2(model, image):
    w = np.shape(image)
    x_step = w[0]//256
    y_step = w[1]//256
    pred = np.zeros((w[0],w[1]))

    for i in range(0,x_step):
        for j in range(0,y_step):
            pred[i*256:(i+1)*256,j*256:(j+1)*256] = pix2pix(model, image[i*256:(i+1)*256,j*256:(j+1)*256,:])
    
    for i in range(0,x_step):
        pred[i*256:(i+1)*256,w[1]-256:] = pix2pix(model, image[i*256:(i+1)*256,w[1]-256:])
    
    for j in range(0,y_step):
        pred[w[0]-256:w[0], j*256:(j+1)*256] = pix2pix(model, image[w[0]-256:w[0], j*256:(j+1)*256])
    
    return pred

def get_inference(model, image):
  size = np.shape(image)
  if size[0]<256 or size[1]<256:
    print('Error. Image Dimensions are too small. Ensure minimum 256 - length and width.')
    sys.exit()
  elif size[0]==256 and size[1]==256:
    pred = pix2pix(model, image)
  elif size[0]==size[1] and size[0]<=512:
    pred = stitched_pix2pix_1(model, image)
  else:
    pred = stitched_pix2pix_2(model, image)
  
  pred = 255*(np.round(pred))
  return pred

def miou_f1(mask,pred):
  mask2 = mask.flatten()
  pred2 = pred.flatten()
  
  m = tf.keras.metrics.MeanIoU(num_classes=2)
  m.update_state(mask2, pred2)
  val = m.result().numpy()
  m.reset_state()
  f1 = f1_score(mask2,pred2)
  return val,f1

def main(argv):

  global OUTPUT_CHANNELS

  OUTPUT_CHANNELS = 1
  IMG_WIDTH = 256
  IMG_HEIGHT = 256

  input_directory = ''
  checkpoint_directory = ''

  try:
    opts, args = getopt.getopt(argv,"hi:c:f:",["input_dir=","checkpoint_dir=","options="])
  except getopt.GetoptError:
    print('Format should be train.py -i <input_dir> -c <checkpoint_dir> -f <options>. The input and checkpoint directory paths are necessary.')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('Format should be train.py -i <input_dir> -c <checkpoint_dir> -f <options>. The input and checkpoint directory paths are necessary.')
      sys.exit()
    elif opt in ("-i", "--input_dir"):
      input_directory = arg
    elif opt in ("-c", "--checkpoint_dir"):
      checkpoint_directory = arg
    elif opt in ("-f", "--options"):
        flag = arg
  
  if flag == '1':
    print('Inference from .txt file.')
  elif flag == '2':
    print('Inference on all images in a folder.')
  else:
    print('Please provide the correct option - either 1 or 2 based on input format.')
  
  assert input_directory != ''
  assert checkpoint_directory!= ''
  assert flag == '1' or flag == '2'

  print('Input Directory is ', input_directory)
  print('Checkpoint Directory is ', checkpoint_directory)

  generator = Generator()
  discriminator = Discriminator()
  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
  
  latest = tf.train.latest_checkpoint(checkpoint_directory)
  print('The latest checkpoint is:', latest)

  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
  print('Model Restored. ')

  if flag == '1':
    test_list_dir = r'{}'.format(input_directory + 'test.txt')
    with open(test_list_dir) as f:
        test_list = f.readlines()
        for i in range(0,len(test_list)):
            test_list[i] = test_list[i].strip()
            test_list[i] = r'{}'.format(input_directory+'Images/'+test_list[i])
  else:
    test_list = glob.glob(input_directory+'Images/*.jpg')

  miou_scores = []
  f1_scores = []

  for image in test_list:
    im = cv2.imread(image)
    mask_name = image
    mask_name = mask_name.replace("Images","Masks")
    mask = cv2.imread(mask_name, 0)
    mask = np.round(mask/255)

    pred = get_inference(generator, im)
    pred = np.round(pred/255) 
    miou, f1 = miou_f1(mask, pred)
    miou_scores.append(miou)
    if f1!=0:
        f1_scores.append(f1)

  print('Evaluation Completed.')
  print(f'The MIOU score is {np.mean(miou_scores)}.')
  if len(f1_scores)!=0:
    print(f'The F1 score is {np.mean(f1_scores)}.')
  else:
    print('The F1 score is N/A.')

if __name__ == "__main__":
   main(sys.argv[1:])