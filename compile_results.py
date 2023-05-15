import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import sys

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from PIL import Image

from tqdm import tqdm


SHAPE = (320, 320, 3)
BATCH_SIZE = 16
SEED = 777
EPOCHS = 10



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


def create_image_list(image_path):
    images = []


    # Iterate through image paths and convert image to tensor
    for path in image_path:
        image_list = sorted(glob.glob(path + '/*.png'))

        images.extend(image_list)

    return images


images = list(np.genfromtxt('main_test.csv', delimiter=',', dtype="|U"))
masks = create_image_list(mask_path)

setA = set([x[-80:] for x in images])

mskres = [x for x in masks if x[-80:] in setA]


images = sorted(images)
masks = sorted(mskres)

print(len(images), len(masks))



def normalize(y_true, y_pred):
    #middle_value = 0.5
    
    middle_value = int((np.max(y_pred) - np.min(y_pred)) / 2)
    y_pred[y_pred > middle_value] = 255
    y_pred[y_pred <= middle_value] = 0

    middle_value = int((np.max(y_true) - np.min(y_true)) / 2)
    y_true[y_true > middle_value] = 255
    y_true[y_true <= middle_value] = 0


    return y_true, y_pred


def conf_matrix(y_true, y_pred):
    #print(y_true)

    assert y_true.shape == y_pred.shape
    # if not a 1d array -> flatten the array
    if len(y_true.shape) != 1:
        y_true_flatten = y_true.flatten()
    if len(y_pred.shape) != 1:
        y_pred_flatten = y_pred.flatten()
        

    cm = confusion_matrix(y_true_flatten, y_pred_flatten).ravel()#, labels=[255,0])
    
    tn, fp, fn, tp = cm.reshape(-1)
    
    return np.array([tn, fp, fn, tp])

def gen_metrics(cm):
    tn, fp, fn, tp = cm
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    precision = (tp) / (tp + fp) if tp + fp != 0 else 0
 
        
    if tp+fn != 0:
        recall = (tp) / (tp + fn)
    else:
        recall = 0
    
    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    dice = 2 * tp / ((tp + fp) + (tp + fn))
    jaccard_index = tp / (tp + fn + fp)
    
    return np.array([accuracy, precision, recall, f1_score, dice, jaccard_index])


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

def build_resnet50_unet(input_shape, model=None):
    """ Input """
    inputs = Input(input_shape, name="input")

    if model is None:
        """ Pre-trained ResNet50 Model """
        resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    else:
        resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)
        resnet50.load_weights(model)
        
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
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    u_net = Model(inputs, outputs, name="ResNet50_U-Net")
    return u_net

# https://github.com/zhixuhao/unet/blob/master/model.py

def build_simple_unet():
    inputs = Input(SHAPE)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)


    up6 = Conv2D(512, 2, activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal', name="up6")(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu',
                   padding = 'same',
                   kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)

    #model.compile(optimizer = Adam(), loss=loss, metrics=[metric])

    return model


images_test, masks_test = images[:2000], masks[:2000]
# images_test, masks_test = images, masks

def compile_run(this_model):
    for idx, paths in tqdm(enumerate(zip(images_test, masks_test)), total=len(images_test)):
        image_path, mask_path = paths
        
        image = load_img(image_path, target_size=(320, 320))
        image = img_to_array(image) / 255.

        mask = Image.open(mask_path)
        mask = mask.resize((320,320))
        mask = np.array(mask).astype('float32')/255.
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))



        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

        pred_mask = this_model.predict(image)
        pred_mask *= 255.0
        

        y_true, y_pred = normalize(mask, pred_mask[0])
        

        if idx == 0:
            cm = conf_matrix(y_true, y_pred)
            metrics = gen_metrics(cm)
        else:
            cm = np.vstack((cm, conf_matrix(y_true, y_pred)))
            metrics = np.vstack((metrics, gen_metrics(conf_matrix(y_true, y_pred))))
            
    return cm, metrics


# ver = sys.argv[1]
# encoder = sys.argv[2]
# #loss_list = ["bce", "dice", "iou"]
# loss = sys.argv[3]

# if sys.argv[4] == "0":
#     pic_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# else:
#     pic_list = [800, 900, 1000]

ver = sys.argv[1]
encoder = sys.argv[2]
loss = sys.argv[3]
pic_list = [100]
pretextPerc = sys.argv[4]
pretextTask = sys.argv[5]
pretextFold = sys.argv[6]

# ver = "resnet50"
# encoder = "frozenEncoder"
# loss = "iou"
# pic_list = [10]
# pretextPerc = "p_0.1"
# pretextTask = "d_4"
# pretextFold = "fold_0"

for picno in pic_list:
    print("############ " + str(picno) + "################")
    for idx in range(3):
        
        if ver == "resnet50":
            model = build_resnet50_unet(SHAPE)
        else:
            model = build_simple_unet()
            
        model.load_weights("Models/%s_%s_%s_%s_%s_%s_%s_%s.h5" 
                           % (picno, ver, encoder, loss, idx, pretextPerc, pretextTask, pretextFold))
 

        cm, metrics = compile_run(model)


        tempcm = cm.sum(axis=0) / len(images_test)
        tempmetrics = metrics.sum(axis=0) / len(images_test)

        print("------------------------------------------------------")
        print(ver+" "+ encoder+" "+loss +" fold: " +str(idx+1))
        print(tempcm)
        print(tempmetrics)
        print("------------------------------------------------------")
    
        np.savetxt(("Metrics/cm_%s_%s_%s_%s_%s_%s_%s_%s.csv" % (picno, ver, encoder, loss, idx, pretextPerc, pretextTask, pretextFold)), cm, delimiter=",")
        np.savetxt(("Metrics/metrics_%s_%s_%s_%s_%s_%s_%s_%s.csv" % (picno, ver, encoder, loss, idx, pretextPerc, pretextTask, pretextFold)), metrics, delimiter=",")
