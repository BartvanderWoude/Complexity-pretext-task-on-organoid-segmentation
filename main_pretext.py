#  python ImageTransformation/main_pretext.py -p 0.1 -w -s 8 -n Name
#
from Aux.waveFilter import *
from Aux.shuffle import *
from Aux.rotate import *
from Aux.rotateShuffle import *
from Aux.overlap import *
from Aux.drop import *
from Aux.blur import *
from Aux.boxBlur import *
from Aux.boxDrop import *
from Aux.boxRotate import *
from Aux.dfBoxBlur import *
from Aux.dfBoxDrop import *
from Aux.dfBoxRotate import *
from Aux.dfDrop import *
from Aux.dfRotate import *

import getopt, sys

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input

from sklearn.model_selection import KFold


#image = Image.open(r"organoids01.png")
#over = Image.open(r"organoids02.png")
boxDimensions = 50
numberOfBoxes = 4

outputFolder = "Output/"
output = "output"


BATCH_SIZE = 16
SEED = 26
EPOCHS = 50
    
SHAPE = (320, 320, 3)


PERC = 0.2



print ("* " * 50)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print ("* " * 50)


####################
# Helper Functions #
####################

def make_pairs(x):
    pairs = []
    
    for idx in range(len(x)):
        j = x[idx]
        
        pairs.append([j, j])
        
    return pairs


def decode_image(image):
    image_string = tf.io.read_file(image)
    image = tf.io.decode_png(image_string, channels=1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.
    
    #print ("decode_image :: --- tf.shape(image) === ", tf.shape(image))
    #print ("decode_image :: --- type(image) ======= ", type(image))
    #print (image)
    
    image = tf.image.resize(image, (SHAPE[0], SHAPE[1]))
    
    #print(image.dtype)
    #print(image.shape)
    print("GT SHAPE: ")
    print(image.shape)
    print("################3")

    return image

def preprocess_pairs(images):

    tensorObject = False
    output = None
    seg = None
    tasks = 1
    task = ""
    area = (0,0,SHAPE[0],SHAPE[1])

    if args.Quarter:
        quarterTasks = str(args.Quarter)
        tasks = 4
        quarterAreaDictionary = {
            0: (0,          0,          SHAPE[0]//2, SHAPE[1]//2),
            1: (SHAPE[0]//2, 0,          SHAPE[0],   SHAPE[1]//2),
            2: (0,          SHAPE[1]//2, SHAPE[0]//2, SHAPE[1]),
            3: (SHAPE[0]//2, SHAPE[1]//2, SHAPE[0],   SHAPE[1])
        }
    
    for taskIdx in range(0, tasks):
        if (args.Quarter):
            print("len qt: ", len(quarterTasks))
            x = random.randint(0, len(quarterTasks) - 1)
            task = quarterTasks[x]
            print("task: ", task)
            temp = ""
            for i in range(len(quarterTasks)):
                if i != x:
                    temp = temp + quarterTasks[i]
            quarterTasks = temp
            print("qt: ", quarterTasks)
            area = quarterAreaDictionary[taskIdx]


        if args.Wave or task == 'w':
            #print ("-- Wave:", args.Wave)
            output = tf_wave(images[0], area=area)
            tensorObject = True

        if args.Shuffle or task == 's':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.Shuffle)
            elif args.Quarter:
                numB = 2

            #print ("-- Shuffle:", args.Shuffle)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_shuffleXBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_shuffleXBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_shuffleBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_shuffleBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True

        if args.Rotate or task == 'r':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.Rotate)
            elif args.Quarter:
                numB = 2

            #print ("-- Rotate:", args.Rotate)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_rotateBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_rotateBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_rotateBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_rotateBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True

        if args.ShuffleRotate or task == 'S':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.ShuffleRotate)
            elif args.Quarter:
                numB = 2

            #print ("-- ShuffleRotate:", args.ShuffleRotate)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_shuffleRotateBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_shuffleRotateBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_shuffleRotateBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_shuffleRotateBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True

        if args.Drop or task == 'd':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.Drop)
            elif args.Quarter:
                numB = 2

            #print ("-- Drop:", args.Drop)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dropBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dropBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dropBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dropBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True

        if args.Blur or task == 'b':
            #print ("-- Blur:", args.Blur)
            if tensorObject and output is not None:
                output = tf_blur(output, area=area, tensorObject=True)
            else:
                output = tf_blur(images[0], area=area)
            tensorObject = True
        
        if args.BoxBlur:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.BoxBlur)
            elif args.Quarter:
                numB = 2

            #print ("-- BoxBlur:", args.BoxBlur)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_boxBlur(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_boxBlur(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_boxBlur_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_boxBlur_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.BoxDrop or task == 'D':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.BoxDrop)
            elif args.Quarter:
                numB = 2

            #print ("-- BoxDrop:", args.BoxDrop)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_boxDrop(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_boxDrop(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_boxDrop_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_boxDrop_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.BoxRotate or task == 'R':
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.BoxRotate)
            elif args.Quarter:
                numB = 2

            #print ("-- BoxRotate:", args.BoxRotate)
            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_boxRotate(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_boxRotate(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_boxRotate_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_boxRotate_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.DiffBoxBlur:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.DiffBoxBlur)
            elif args.Quarter:
                numB = 2

            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dfBoxBlur(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dfBoxBlur(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dfBoxBlur_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dfBoxBlur_seg(images[0], bDim, numB, area=area)
                tensorObject = True

        if args.DiffBoxDrop:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.DiffBoxDrop)
            elif args.Quarter:
                numB = 2

            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dfBoxDrop(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dfBoxDrop(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dfBoxDrop_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dfBoxDrop_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.DiffBoxRotate:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.DiffBoxRotate)
            elif args.Quarter:
                numB = 2

            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dfBoxRotate(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dfBoxRotate(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dfBoxRotate_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dfBoxRotate_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.DiffDrop:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.DiffDrop)
            elif args.Quarter:
                numB = 2

            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dfDropBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dfDropBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dfDropBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dfDropBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        if args.DiffRotate:
            bDim = int(args.BoxDims)
            if args.Shuffle:
                numB = int(args.DiffRotate)
            elif args.Quarter:
                numB = 2

            if not args.Seg:
                if tensorObject and output is not None:
                    output = tf_dfRotateBoxes(output, bDim, numB, area=area, tensorObject=True)
                else:
                    output = tf_dfRotateBoxes(images[0], bDim, numB, area=area)
                tensorObject = True
            else:
                if tensorObject and output is not None:
                    output, seg = tf_dfRotateBoxes_seg(output, bDim, numB, area=area, tensorObject=True, seg=seg)
                else:
                    output, seg = tf_dfRotateBoxes_seg(images[0], bDim, numB, area=area)
                tensorObject = True
        
        taskIdx = taskIdx + 1

    if output is None:
        raise Exception("No Pretext Task was chosen")

    if not args.Seg:
        return (output, decode_image(images[1]))
    else:
        return (output, seg)


def data_gen(datapoint):
    datapoint = tf.constant(datapoint)
    dataset = tf.data.Dataset.from_tensor_slices(datapoint)
    
    #dataset = dataset.map(preprocess_pairs, num_parallel_calls=tf.data.AUTOTUNE)

    # replace
    #spectrogram_time_step_ds = spectrogram_ds.map(get_time_step_spectrogram_and_label_id) 
    #by 
    #spectrogram_time_step_ds = tf.py_function(func=get_time_step_spectrogram_and_label_id, inp=[spectrogram_ds], Tout=[tf.int64, np.ndarray]) 
    
    #dataset = tf.py_function(func=preprocess_pairs, inp=[dataset], Tout=[tf.float64, tf.float64])
    #dataset = dataset.map(lambda x: tf.py_function(preprocess_pairs, [x], [tf.uint8, tf.float32]))
    dataset = dataset.map(lambda x: tf.py_function(preprocess_pairs, [x], [tf.float32, tf.float32]))

                
    dataset = (dataset.batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    
    
    '''
    i = 0

    for data in dataset: 
        d1, d2 = data
        print ("d1, d2 ============ ", d1.shape, d2.shape)
        print(data[0].shape, data[1].shape, type(data))
        i = i + 1

        for _compress_image, _decode_image in zip(d1, d2):


            print ("_compress_image.shape ============== ", _compress_image.shape)
            print ("_decode_image.shape ================ ", _decode_image.shape)
            # _compress_image, _decode_image

            idx = 0
            print(data[0][idx].shape, data[1][idx].shape, type(data))
            #"""
            plt.figure(figsize=(6,10)) 
            
            plt.subplot(3, 2, 1)
            plt.imshow(_compress_image[:,:,0].numpy().astype("float32"), "gray")
            plt.subplot(3, 2, 2)
            plt.imshow(_decode_image[:,:,0].numpy().astype("float32"), "gray")

            plt.subplot(3, 2, 3)
            plt.imshow(_compress_image[:,:,1].numpy().astype("float32"), "gray")
            plt.subplot(3, 2, 4)
            plt.imshow(_decode_image[:,:,1].numpy().astype("float32"), "gray")

            plt.subplot(3, 2, 5)
            plt.imshow(_compress_image[:,:,2].numpy().astype("float32"), "gray")
            plt.subplot(3, 2, 6)
            plt.imshow(_decode_image[:,:,2].numpy().astype("float32"), "gray")
            plt.show()
            #"""

        #break

    print ("i ==================================== ", i)
    for idx in range(3):
        display(dataset, idx)
    '''

    return dataset


##################
# ResNet50 Model #
##################

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

def ssim_mae_loss(y_pred, y_true):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = ssim_loss(y_pred, y_true)

    return (mae / 2) + (ssim / 2)


def ssim_score(y_pred, y_true):
    return tf.reduce_mean(tf.image.ssim(y_pred, y_true, max_val=1.0))

def ssim_loss(y_pred, y_true):
    return 1 - ssim_score(y_pred, y_true)





#################
# Main Function #
#################

parser = argparse.ArgumentParser(
                    prog = 'PreTextTask',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

#parser.add_argument('filename')           # positional argument
parser.add_argument('-p', '--Percentage', required=True)      # option that takes a value
parser.add_argument('-seg', '--Seg', default=False, action="store_true",
                    help="The pretext task type is segmentation")
parser.add_argument('-nb', '--NumBoxes')      # option that takes a value
parser.add_argument('-bd', '--BoxDims')      # option that takes a value
parser.add_argument('-w', '--Wave', default=False, action="store_true",
                    help="Wave is used")
parser.add_argument('-s', '--Shuffle')
parser.add_argument('-r', '--Rotate')
parser.add_argument('-S', '--ShuffleRotate')
#parser.add_argument('-o', '--Overlap')
parser.add_argument('-d', '--Drop')
parser.add_argument('-b', '--Blur', default=False, action="store_true",
                    help="Blur is used")
parser.add_argument('-R', '--BoxRotate')
parser.add_argument('-D', '--BoxDrop')
parser.add_argument('-B', '--BoxBlur')
## Different sized boxes: df ##
parser.add_argument('-dfB', '--DiffBoxBlur')
parser.add_argument('-dfD', '--DiffBoxDrop')
parser.add_argument('-dfR', '--DiffBoxRotate')
parser.add_argument('-dfd', '--DiffDrop')
parser.add_argument('-dfr', '--DiffRotate')
## Random task per quarter image ##
parser.add_argument('-q', '--Quarter')
######
parser.add_argument('-n', '--Name', required=True)      # option that takes a value
parser.add_argument('-psm', '--PeregrineSaveModels', 
    type=int, choices=range(0,5),
    metavar="[0-4]", 
    help='number of the model')      # option that takes a value

args = parser.parse_args()
print(args.Percentage, args.NumBoxes, args.BoxDims, args.Wave )


########################
# Read Images to train #
########################

#image_files = np.genfromtxt('self_supervised_network/triplet_/main_pretrain2.csv', delimiter=',', dtype="|U")
#image_files = np.genfromtxt('/home/amath/Desktop/self_supervised_pretext/main_pretrain3.csv', delimiter=',', dtype="|U")
#image_files = np.genfromtxt('/data/pg-organoid_data/simple_complex_pretext_learning/main_pretrain.csv', delimiter=',', dtype="|U")
#image_files = np.genfromtxt('/tmp/data/simple_complex_pretext_learning/main_pretrain.csv', delimiter=',', dtype="|U")
image_files = np.genfromtxt('main_pretrain_test.csv', delimiter=',', dtype="|U")


random.shuffle(image_files)

print("Original len(image_files) === ", len(image_files))


print ("-- Percentage:", args.Percentage)
#image_count = int(len(image_files))
image_count = int(float(args.Percentage) * len(image_files))

print("Original len(image_files) === ", len(image_files))
sub_files = image_files[:image_count]
print("len(sub_files) === ", len(sub_files))



print ("-- sm:", args.PeregrineSaveModels, type(args.PeregrineSaveModels))


if args.NumBoxes:
    print ("-- NumBoxes:", args.NumBoxes)



if args.BoxDims:
    print ("-- BoxDims:", args.BoxDims)
else: 
    args.BoxDims = 50
    print ("-- NEW BoxDims:", args.BoxDims)
    


#################################################
# Train the model with 5-folds Cross Validation #
#################################################

def train_model(train, val, kfold):
    x_train, x_val = sub_files[train], sub_files[val]
        
    pairs_train = make_pairs(x_train)
    pairs_val = make_pairs(x_val)

    print ("pairs_train === ", len(pairs_train))
    print ("pairs_val === ", len(pairs_val))

    train_ds = data_gen(pairs_train)
    val_ds = data_gen(pairs_val)

    print ("train_ds === ", len(train_ds))
    print ("val_ds === ", len(val_ds))


    #loss_logger = tf.keras.callbacks.CSVLogger("self_supervised_network/triplet_/pretext_loss_logs/try_PERC_%s_fold_%s.csv" 
    loss_logger = tf.keras.callbacks.CSVLogger("simple_complex_pretext_learning/pretext_loss_logs/%s_p_%s_fold_%s.csv" 
                                               % (str(args.Name), str(args.Percentage), kfold))

    u_net = build_resnet50_unet(SHAPE)


    u_net.compile(loss=ssim_loss, optimizer=tf.keras.optimizers.Adam(), metrics=ssim_score)
    
    history = u_net.fit(train_ds, 
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[loss_logger])


    #u_net.save_weights("self_supervised_network/triplet_/pretext_model_output/try_PERC_%s_fold_%s.h5"
    u_net.save_weights("simple_complex_pretext_learning/pretext_model_output/%s_p_%s_fold_%s.h5"
                      % (str(args.Name), str(args.Percentage), kfold))




# Running each model separately on peregrine
if args.PeregrineSaveModels or args.PeregrineSaveModels == 0:

    print ("PeregrineSaveModels:", int(args.PeregrineSaveModels))

    # create and save the validation models for each fold
    if args.PeregrineSaveModels == 0:
        print ("model 0")
        for kfold, (train, val) in enumerate(KFold(n_splits=5).split(sub_files)):
            print (kfold)
            print (val)

            # save the validation models in pretext_val_models
            # write a list without using a loop
            with open('simple_complex_pretext_learning/pretext_PeregrineSaveModels/%s_val_%s.txt' % (str(args.Name), str(kfold)), 'w') as fp:
                fp.write('\n'.join(str(v) for v in val))
            with open('simple_complex_pretext_learning/pretext_PeregrineSaveModels/%s_train_%s.txt' % (str(args.Name), str(kfold)), 'w') as fp:
                fp.write('\n'.join(str(v) for v in train))


    # for all models [0, 1, 2, 3, 4]
    # read the validation data + create the train data + run the models

    file = open('simple_complex_pretext_learning/pretext_PeregrineSaveModels/%s_val_%s.txt' % (str(args.Name), str(args.PeregrineSaveModels)), "r")
    val = file.read()
    file.close()
    file = open('simple_complex_pretext_learning/pretext_PeregrineSaveModels/%s_train_%s.txt' % (str(args.Name), str(args.PeregrineSaveModels)), "r")
    train = file.read()
    file.close()

    val = val.split('\n')
    val = [eval(v) for v in val]

    train = train.split('\n')
    train = [eval(v) for v in train]

    train_model(train, val, str(args.PeregrineSaveModels))




else: 

    for kfold, (train, val) in enumerate(KFold(n_splits=5).split(sub_files)):
        """
        x_train, x_val = sub_files[train], sub_files[val]
        
        pairs_train = make_pairs(x_train)
        pairs_val = make_pairs(x_val)

        print ("pairs_train === ", len(pairs_train))
        print ("pairs_val === ", len(pairs_val))

        train_ds = data_gen(pairs_train)
        val_ds = data_gen(pairs_val)

        print ("train_ds === ", len(train_ds))
        print ("val_ds === ", len(val_ds))


        #loss_logger = tf.keras.callbacks.CSVLogger("self_supervised_network/triplet_/pretext_loss_logs/try_PERC_%s_fold_%s.csv" 
        loss_logger = tf.keras.callbacks.CSVLogger("simple_complex_pretext_learning/pretext_loss_logs/%s_p_%s_fold_%s.csv" 
                                                   % (str(args.Name), str(args.Percentage), kfold))

        u_net = build_resnet50_unet(SHAPE)


        u_net.compile(loss=ssim_loss, optimizer=tf.keras.optimizers.Adam(), metrics=ssim_score)
        
        history = u_net.fit(train_ds, 
                        epochs=EPOCHS,
                        validation_data=val_ds,
                        callbacks=[loss_logger])


        #u_net.save_weights("self_supervised_network/triplet_/pretext_model_output/try_PERC_%s_fold_%s.h5"
        u_net.save_weights("simple_complex_pretext_learning/pretext_model_output/%s_p_%s_fold_%s.h5"
                          % (str(args.Name), str(args.Percentage), kfold))
        """
        train_model(train, val, kfold)


print ("------------------------- DONE -------------------------")
print ("------------------------- DONE -------------------------")
print ("------------------------- DONE -------------------------")

    


exit()
# checking each argument
for currentArgument, currentValue in arguments:
    print ("currentArgument === ", currentArgument)
    print ("currentValue ====== ", currentValue)

    # percentage of the data
    if currentArgument in ("-p", "--Percentage"):

        #image_count = int(len(image_files))
        image_count = int(currentValue) * len(image_files)

        print("Original len(image_files) === ", len(image_files))
        sub_files = image_files[:image_count]
        print("len(sub_files) === ", len(sub_files))

    # number of boxes
    if currentArgument in ("-nb", "--NumBoxes"):
        numberOfBoxes = int(currentValue)

    # Box dimensions
    if currentArgument in ("-bd", "--BoxDims"):
        boxDimensions = int(currentValue)

    if currentArgument in ("-w", "--Wave"):
        image = wave(image)

    elif currentArgument in ("-s", "--Shuffle"):
        image = shuffleXBoxes(image, boxDimensions, int(currentValue))

    elif currentArgument in ("-r", "--Rotate"):
        image = rotateBoxes(image, boxDimensions, int(currentValue))
    
    elif currentArgument in ("-S", "--ShuffleRotate"):
        image = shuffleRotateBoxes(image, boxDimensions, int(currentValue))

    elif currentArgument in ("-o", "--Overlap"):
        image = overlap(image, over)

    elif currentArgument in ("-d", "--Drop"):
        image = dropBoxes(image, boxDimensions, int(currentValue))

    elif currentArgument in ("-b", "--Blur"):
        image = blur(image)
    """
    
    elif currentArgument in ("-n", "--Name"):
        output = currentValue
    """





#image.save(outputFolder + output + ".png")
