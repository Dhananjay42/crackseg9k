import tensorflow as tf
import tensorflow_addons as tfa
import os
import time
import numpy as np
import random
import sys, getopt
import cv2
import datetime
from segmentation_models import FPN

def crop(input_image, real_image):
    cropped_input = tf.image.central_crop(input_image, central_fraction = 256/448 )
    cropped_real = tf.image.central_crop(real_image, central_fraction = 256/448 )

    cropped_input = tf.image.resize(cropped_input, [256, 256],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped_real = tf.image.resize(cropped_real, [256, 256],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return cropped_input, cropped_real

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    input_image = tf.clip_by_value(input_image, -1, 1)
    real_image = tf.clip_by_value(real_image, -1, 1)

    return input_image, real_image

def flipleftright(input_image,real_image):
  if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image

def flipupdown(input_image,real_image):
  if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_up_down(input_image)
      real_image = tf.image.flip_up_down(real_image)
  return input_image, real_image

def randomrotate(input_image,real_image):
  p = tf.random.uniform(())
  if p <= 0.25:
    input_image = tfa.image.rotate(input_image, np.pi/2)
    real_image = tfa.image.rotate(real_image, np.pi/2)
  elif p<=0.5:
    input_image = tfa.image.rotate(input_image, np.pi)
    real_image = tfa.image.rotate(real_image, np.pi)
  elif p<=0.75:
    input_image = tfa.image.rotate(input_image, 1.5*np.pi)
    real_image = tfa.image.rotate(real_image, 1.5*np.pi)
  
  return input_image, real_image

def multiply(input_image,real_image):
  alpha = random.uniform(0.8, 1.5)
  input_image = alpha*input_image
  
  return input_image, real_image

def gaussianblur(input_image,real_image):
  sig = random.uniform(0, 5)
  input_image = tfa.image.gaussian_filter2d(input_image, filter_shape =  (3, 3), sigma = sig)

  return input_image, real_image

def roundoff(input_image, real_image):
  
  real_image = tf.round(real_image)

  return input_image, real_image

@tf.function()
def preprocessing(input_image, real_image):
    input_image, real_image = crop(input_image, real_image)
    funcs = [flipleftright,flipupdown,randomrotate,multiply,gaussianblur]
    random.shuffle(funcs)

    for function in funcs:
      input_image, real_image = function(input_image, real_image)

    return input_image, real_image

def load_image_train(input_image, real_image):
    input_image, real_image = preprocessing(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    input_image, real_image = roundoff(input_image, real_image)
    
    return input_image, real_image

def load_image_test(input_image, real_image):
    input_image, real_image = crop(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    input_image, real_image = roundoff(input_image, real_image)

    return input_image, real_image

def parse_image(img_path: str):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)

    mask_path = tf.strings.regex_replace(img_path, "Images", "Masks")
    print('mask path is', mask_path)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=3)

    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask = tf.image.rgb_to_grayscale(mask)

    return image, mask

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

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    last = FPN('resnet50',  classes=1, activation='sigmoid', input_shape=(None, None, 3), encoder_weights = None)
    x = inputs
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

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

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

@tf.function
def train_step(input_image, target, epoch):
    global summary_writer
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def generate_images(model, test_input, tar):
  global trained_epoch, input_directory
  prediction = model(test_input, training=True)

  test_input = test_input[0].numpy()
  prediction = prediction.numpy()
  tar = tar.numpy()

  plot_tar = np.reshape(tar[0],(256,256))
  plot_pred = np.reshape(prediction[0],(256,256)).copy()

  tar_3d = np.stack((plot_tar, plot_tar, plot_tar), axis = 2)
  pred_3d = np.stack((plot_pred, plot_pred, plot_pred), axis = 2)
  
  out = np.hstack((test_input, tar_3d, pred_3d))
  cv2.imwrite(input_directory+'progress/out_epoch_'+str(trained_epoch)+'.jpg', 127.5*(out+1))

def fit(train_ds, epochs, test_ds):
  global trained_epoch, train_len, BATCH_SIZE

  if train_len%BATCH_SIZE == 0:
    iter = train_len/BATCH_SIZE
  else:
    iter = int(train_len/BATCH_SIZE) + 1

  for epoch in range(epochs):
    start = time.time()

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)

    print("Training commenced for epoch: ", epoch+1)

    curr = 0

    for n, (input_image, target) in train_ds.enumerate():
      if n>=curr*iter/100 and n<(curr+1)*iter/100:
        continue
      else:
        print('.', end='')
      train_step(input_image, target, epoch)

    checkpoint.save(file_prefix=checkpoint_prefix)
    trained_epoch = trained_epoch + 1

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

def main(argv):

  global OUTPUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT, BUFFER_SIZE, BATCH_SIZE, LAMBDA, loss_object, trained_epoch, input_directory, train_len
  OUTPUT_CHANNELS = 1
  IMG_WIDTH = 256
  IMG_HEIGHT = 256

  input_directory = ''
  checkpoint_directory = ''
  BUFFER_SIZE = 400
  BATCH_SIZE = 8
  LAMBDA = 100
  EPOCHS = 0

  trained_epoch = 0

  try:
    opts, args = getopt.getopt(argv,"hi:c:b:l:e:x:",["input_dir=","checkpoint_dir=","batch_size=","lambda=","epochs=","buffer_size="])
  except getopt.GetoptError:
    print('Format should be train.py -i <input_dir> -c <checkpoint_dir> -b <batch_size> -l <lambda> -e <epochs> -x <buffer_size>. The input and checkpoint directory paths are necessary.')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('Format should be train.py -i <input_dir> -c <checkpoint_dir> -b <batch_size> -l <lambda> -e <epochs> -x <buffer_size>. The input and checkpoint directory paths are necessary.')
      sys.exit()
    elif opt in ("-i", "--input_dir"):
      input_directory = arg
    elif opt in ("-c", "--checkpoint_dir"):
      checkpoint_directory = arg
    elif opt in ("-b", "--batch_size"):
      BATCH_SIZE = arg
    elif opt in ("-l", "--lambda"):
      LAMBDA = arg
    elif opt in ("-e", "--epochs"):
      EPOCHS = arg
    elif opt in ("-x", "--buffer_size"):
      BUFFER_SIZE = arg
  
  assert input_directory != ''
  assert checkpoint_directory!= ''

  EPOCHS = np.int64(EPOCHS)
  BUFFER_SIZE = np.int64(BUFFER_SIZE)
  BATCH_SIZE = np.int64(BATCH_SIZE)

  print('Input Directory is ', input_directory)
  print('Checkpoint Directory is ', checkpoint_directory)
  print('Batch Size is ', BATCH_SIZE)
  print('Lambda is ', LAMBDA)
  print('Epochs is ', EPOCHS)
  print('Buffer Size is ', BUFFER_SIZE)

  train_list_dir = r'{}'.format(input_directory + 'train.txt')
  test_list_dir = r'{}'.format(input_directory + 'test.txt')

  with open(train_list_dir) as f:
    train_list = f.readlines()
  for i in range(0,len(train_list)):
    train_list[i] = train_list[i].strip()
    train_list[i] =  r'{}'.format(input_directory+'Images/'+train_list[i])

  with open(test_list_dir) as f:
    test_list = f.readlines()
  for i in range(0,len(test_list)):
    test_list[i] = test_list[i].strip()
    test_list[i] = r'{}'.format(input_directory+'Images/'+test_list[i])

  train_len = len(train_list[i])
    
  train_dataset =  tf.data.Dataset.from_tensor_slices(train_list)
  train_dataset = train_dataset.map(parse_image)
  train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
  #train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.batch(BATCH_SIZE)

  test_dataset = tf.data.Dataset.from_tensor_slices(test_list)
  test_dataset = test_dataset.map(parse_image)
  test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(BATCH_SIZE)
  

  # input_directory = r'{}'.format(input_directory)
  # output_directory = r'{}'.format(output_directory)
  # checkpoint_directory = r'{}'.format(checkpoint_directory)

  global generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_prefix, checkpoint, summary_writer
  
  log_dir= input_directory + "logs/"
  summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  generator = Generator()
  discriminator = Discriminator()
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
  
  latest = tf.train.latest_checkpoint(checkpoint_directory)
  print('The latest checkpoint is:', latest)

  if latest is None:
    trained_epoch = 0
  else:
    trained_epoch = int(latest[-1])
    i = 2
    while latest[-i].isnumeric():
      trained_epoch = int(latest[-i:])
      i = i + 1

  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
  print('Checkpoint Restored. ')

  fit(train_dataset, EPOCHS, test_dataset)
  
if __name__ == "__main__":
   main(sys.argv[1:])



