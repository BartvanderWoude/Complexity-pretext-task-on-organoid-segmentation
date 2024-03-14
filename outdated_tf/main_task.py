import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

from keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

import random

random.seed(777)

BATCH_SIZE = 16
SEED = 26
EPOCHS = 30

TRAIN_ENCODER = False

SHAPE = (320, 320, 3)
MODEL_WEIGHTS = None


mask_path = [
        "5ZBQ5VZO6IA5VHDX3X0U/Control_1_0H_-_For_ASMAA-Stitching-02/Masks_crops_complete_window100_w636_h636",
        "L9YXESHQ6UDYDYTG3UP2/Control_1_24H_For_ASMAA_stitched/Masks_crops_complete_window100_w636_h636",
        "A5VHDX3X0UWN2VMLYD0Y/Control_1_48H_For_Asmaa_Stitched/Masks_crops_complete_window100_w636_h636",
        "6QL9YXESHQ6UDYDYTG3U/Control_1_72H_For_ASMAA_stitched/Masks_crops_complete_window100_w636_h636",
        "P2R2WP36RVDLHB15I48C/Control_1_96H_-_For_ASMAA_stitched/Masks_crops_complete_window100_w636_h636",
        "48JUXOC36SAUYOEQSK0W/Starvation_1_0H_-_For_ASMAA_Stitched/Masks_crops_complete_window100_w636_h636",
        "1CA5KMCR7K7MT53OS7FG/Starvation_1_24H_For_ASMAA_Stitched/Masks_crops_complete_window100_w636_h636",
        "289RM7EZ02HD117WJ5IM/Starvation_1_48H_-_For_ASMAA-Stitched/Masks_crops_complete_window100_w636_h636",
        "NPCPNEZBA0U9BCP1SLTH/Starvation_1_72H_-_For_ASMAA_stitched/Masks_crops_complete_window100_w636_h636",
        "SUN6BXJ3O0O61Z9MYE5C/Starvation_1_96H_-_For_ASMAA_stitched/Masks_crops_complete_window100_w636_h636",
    ]

############# FUNCTIONS #######################################################
def create_image_list(image_path):
    images = []


    # Iterate through image paths and convert image to tensor
    for path in image_path:
        image_list = sorted(glob.glob(path + '/*.png'))

        images.extend(image_list)

    return images


def decode_seg_mask(images):
    image_string = tf.io.read_file(images[0])
    image_decoded = tf.io.decode_png(image_string, channels=1)
    image_decoded = tf.image.grayscale_to_rgb(image_decoded)
    image_decoded = tf.image.resize(image_decoded, (SHAPE[0], SHAPE[1]))

    mask_string = tf.io.read_file(images[1])
    mask_decoded = tf.io.decode_png(mask_string, channels=1)
    mask_decoded = tf.image.resize(mask_decoded, (SHAPE[0], SHAPE[1]))

    # Normalize
    image_decoded = tf.cast(image_decoded, tf.float32) / 255.0
    mask_decoded = tf.cast(mask_decoded, tf.float32) / 255.0

    if tf.random.uniform(()) > 0.5:
        image_decoded = tf.image.flip_left_right(image_decoded)
        mask_decoded = tf.image.flip_left_right(mask_decoded)

    return image_decoded, mask_decoded


def make_pairs(x, y):
    pairs = []
    
    for idx in range(len(x)):
        j, k = x[idx], y[idx]
        
        pairs.append([j, k])
        
    return pairs

def data_gen(datapoint):
    datapoint = tf.constant(datapoint)
    dataset = (tf.data.Dataset.from_tensor_slices(datapoint)
              .map(decode_seg_mask, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(1024, seed=SEED)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE)
              .repeat(EPOCHS))
                          
    return dataset
               
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/resnet50_unet.py

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape, name="input")

 
    """ ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    resnet50.trainable = False
      

    """ Encoder """
    s1 = resnet50.get_layer("input").output          
    s2 = resnet50.get_layer("conv1_relu").output       
    s3 = resnet50.get_layer("conv2_block3_out").output  
    s4 = resnet50.get_layer("conv3_block4_out").output  

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## Bottleneck

    """ Decoder """
    d1 = decoder_block(b1, s4, 320)
    d2 = decoder_block(d1, s3, 160)                     
    d3 = decoder_block(d2, s2, 80)                     
    d4 = decoder_block(d3, s1, 40)                      

    """ Output """
    outputs = Conv2D(3, 3, padding="same", activation="sigmoid")(d4)

    u_net = Model(inputs, outputs, name="ResNet50_U-Net")
    return u_net
               
def dice_score(y_true, y_pred, smooth=0.5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

    
def dice_loss(y_true, y_pred):
        
    return 1 - dice_score(y_true, y_pred)
               
def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)

############# SCRIPT #######################################################
main_percentage = sys.argv[3]
pretext_percentage = sys.argv[4]

images = list(np.genfromtxt('simple_complex_pretext_learning/main_task.csv', delimiter=',', dtype="|U"))
masks = create_image_list(mask_path)

setA = set([x[-80:] for x in images])

mskres = [x for x in masks if x[-80:] in setA]

images = sorted(images)
masks = sorted(mskres)

print(len(images), len(masks))

image_count = int(len(images) * main_percentage)

img_div, msk_div = images[:image_count], masks[:image_count]

aug_type = sys.argv[2]
pretext_percentage_text = "p_" + str(pretext_percentage) + "_"
WEIGHT = pretext_percentage_text + aug_type + pretext_percentage_text + "fold_0.h5"
# if aug_type == '0.25':
#     WEIGHT = "pre_train_unetpd_0.25_fold_0.h5"
# elif aug_type == '0.5':
#     WEIGHT = "pre_train_unetpd_0.5_fold_0.h5"  
# elif aug_type == '0.75':
#     WEIGHT = "pre_train_unetpd_0.75_fold_0.h5"
# elif aug_type == 'blur':
#     WEIGHT = "pre_train_unet_blur_fold_0.h5"
# elif aug_type == 'ssim_blur':
#     WEIGHT = "pre_train_ssim_unet_blur_fold_0.h5"
    
img_div = np.array(img_div)
msk_div = np.array(msk_div)

    
for kfold, (train, val) in enumerate(KFold(n_splits=5).split(img_div,msk_div)):
    print("----------------------------------------------------------------------------")
    x_train, x_val = img_div[train], img_div[val]
    y_train, y_val = msk_div[train], msk_div[val]
               
    print(len(x_train))
               
    train_step = len(x_train) // BATCH_SIZE
    val_step = len(x_val) // BATCH_SIZE
    
    pairs_train = make_pairs(x_train, y_train)
    pairs_val = make_pairs(x_val, y_val)
    
    train_ds = data_gen(pairs_train)
    val_ds = data_gen(pairs_val)
    
    loss_logger = tf.keras.callbacks.CSVLogger("simple_complex_pretext_learning/main_loss_logs/mainPerc_%s_pretextPerc_%s_main_task_mae_%s_%s_fold_%s.csv" 
                            % (str(main_percentage*100), str(pretext_percentage*100), sys.argv[1], sys.argv[2], kfold))
               
    model = build_resnet50_unet(SHAPE)
    model.load_weights("simple_complex_pretext_learning/pretext_model_output/%s" % (WEIGHT))
    new_output = Conv2D(1, 1, padding="same", activation="sigmoid")(model.layers[-2].output)
    model = Model(model.input, new_output)
                        
                       
            
    if sys.argv[1] == 'dice_loss':
        model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(), metrics=dice_score)
    elif sys.argv[1] == 'bce_loss':
        model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(), metrics='accuracy')  
    elif sys.argv[1] == 'iou_loss':
        model.compile(loss=jaccard_distance, optimizer=tf.keras.optimizers.Adam(), metrics=tf.keras.metrics.MeanIoU(2)) 
                       
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        steps_per_epoch=train_step,
                        validation_steps=val_step,
                        validation_data=val_ds,
                        callbacks=[loss_logger])
                       
    
    model.save_weights("simple_complex_pretext_learning/main_model_output/mainPerc_%s_pretextPerc_%s_main_task_mae_%s_%s_fold_%s.h5" 
                            % (str(main_percentage*100), str(pretext_percentage*100), sys.argv[1], sys.argv[2], kfold))
