#%% Imports
import glob
import random
import cv2
import h5py
import numpy as np
from keras.applications.inception_v3 import InceptionV3
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add,Reshape
from keras.models import Model
from keras import regularizers
from keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras import applications
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_normal
from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model
from keras.optimizers import *
from PIL import Image
import tensorflow as tf
from scipy.stats import mode
from keras.activations import relu
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Input,concatenate

from keras.models import Model
from keras.optimizers import Adam
from keras import applications

from evaluation_tests import *
from data_visualisation import *


#%% Definition of the Encoding Model
def triplet_loss(y_true, y_pred):
    alpha = 0.5
    anchor, positive, negative = y_pred[: , 0, :], y_pred[:, 1, :], y_pred[:, 2 , :]

    positive_distance = K.mean(K.square(anchor - positive), axis=-1)
    negative_distance = K.mean(K.square(anchor - negative), axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))

Encoding_model = applications.vgg19.VGG19(include_top=False,weights='imagenet', pooling='max')

for layer in Encoding_model.layers[:15]:
    layer.trainable = False
Encoding_model.summary()

from keras.layers import  Input,concatenate , Reshape

channels=3
img_width=224
img_height=224

anchor_in = Input(shape=(img_width, img_height,channels),name="anchor")
pos_in    = Input(shape=(img_width, img_height,channels),name="positive")
neg_in    = Input(shape=(img_width, img_height,channels),name="negative")

anchor_out = Reshape((1,512))(Encoding_model(anchor_in))
pos_out = Reshape((1,512)) (Encoding_model(pos_in))
neg_out = Reshape((1,512))(Encoding_model(neg_in))
merged_vector = concatenate([anchor_out, pos_out, neg_out],axis=1)

merged_vector.shape


#%% Defininig the siamese Model

model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
model.compile(optimizer=Adam(lr=0.000005),loss=triplet_loss)
model.load_weights('model_rgb_224_weights_55k_alllll.h5')
model.summary()




#%% Test Predection With Preprocessing

path_image1 = "C:/Users/Asus/Desktop/Signature_OCR_Wevioo/Tunisian_signatures/2_01.PNG"
path_image2 = "C:/Users/Asus/Desktop/Signature_OCR_Wevioo/Tunisian_signatures/3_03.PNG"

image_size=224
image1=Image.open(path_image1)
image2=Image.open(path_image2)


image1=process_one_image(image1,padding=True)
image1_rgb=torgb(image1)
print(image1_rgb.shape)
display_one(image1_rgb.reshape(224,224,3)/255,'image1_rgb')


image2=process_one_image(image2,padding=True)
image2_rgb=torgb(image2)
display_one(image2_rgb.reshape(224,224,3)/255,'image2_rgb')


x = model.predict([image1_rgb, image2_rgb, image1_rgb])
a1, p1, useless=x[0,0,:], x[0,1,:], x[0,2,:]
distance = np.linalg.norm(a1 - p1)
cos_similarity=round(1-spatial.distance.cosine(a1,p1),2)

max_accept=200
min_not_accept=350
decision=''
if (distance <= max_accept):
    if(cos_similarity>=0.78):
        decision = 'Accepted'
    else:
        decision = 'Can t judge'
else:
    if (distance >= min_not_accept):
        decision = 'Not Accepted'
    else:
        if (max_accept < distance < min_not_accept):
            if (cos_similarity >= 0.78):
                decision = 'Accepted'
            else:
                if (cos_similarity < 0.75):
                    decision = 'Not Accepted'
                else:
                    decision = 'Can t judge'
        else:
            decision = 'Not Accepted'

print('Decision : ', decision , ' with distance :',distance, ' and percentage :', cos_similarity*100)


display_two(image1_rgb.reshape(224,224,3)/255,image2_rgb.reshape(224,224,3)/255)


#%% Data Augmentation
# from data_augmentation_data import *
#
# generated=one_image_augmentation(path_image2,50,224)
#
# new_images=[]
# new_images.append(image2_rgb)
# for im in generated:
#     im=torgb(im)
#     print(im.shape)
#     new_images.append(im)
#     #display_one(im.reshape(224,224,3))
#
# results=[]
# print(image1_rgb.shape)
#
# for img in new_images :
#     print('-----------',img.shape,'----------',image1_rgb.shape)
#     x = model.predict([image1_rgb, img, image1_rgb])
#     a1, p1, useless=x[0,0,:], x[0,1,:], x[0,2,:]
#     print("Cos Percentage :", (1 - spatial.distance.cosine(a1, p1))*100)
#     results.append(np.linalg.norm(a1 - p1))
#
#
# print(results)
#
# #for im in new_images:
#     #display_one(im.reshape(224,224,3))
#
# mean_distance=np.mean(results)
# similarity_score=''
#
# if mean_distance==0:
#     similarity_score = 'Accepted : Similarity = 100%'
# else:
#     if mean_distance<= 310:
#         similarity_score='Accepted '+str(int(100 - np.min(results)))
#     if 310< mean_distance <= 320:
#         similarity_score='Can t jusdge '+str(int(100 - np.min(results)))
#     if mean_distance>320:
#         similarity_score='Not Accepted '+str(int(100 - np.max(results)))
#
#
# print('Final Result',similarity_score,'%')
#




#%% ------------ Testing With Saif Trainning Set ---------------------------------

X1,X2,label=get_test_data_saif_training_set()

print(len(X1), ' ', X1[1].shape)
print(len(X2))
print(len(label))

not_accepted_distances, accepted_distances,distances=evaluate(model,X1,X2,label,440,200,350,'saif_eval_[200,350]')

#%%
j=70
display_two(X1[j].reshape(224,224,3)/255,X2[j].reshape(224,224,3)/255)
print(label[j])

print(distances[j])



#%% -------------------------------------Testing with the actual data set----------------------------------------
X1,X2,label=get_actual_training_set()

print(len(X1), ' ', X1[1].shape)
print(len(X2))
print(len(label))

not_accepted_distances, accepted_distances,distances,cos_accepted,cos_not_accepted=evaluate(model,X1,X2,label,468,200,350,'actual_data_testing_[200,350].txt')


#%%--------------------------------------Testing with Dutch---------------------------------------------
X1,X2,label=get_dutch_set()

print(len(X1), ' ', X1[1].shape)
print(len(X2))
print(len(label))

not_accepted_distances, accepted_distances,distances=evaluate(model,X1,X2,label,80,200,350,'Dutch_evaluation_80_[200,350]_cos_sim.txt')
plot_distances(accepted_distances,cos_accepted,not_accepted_distances,cos_not_accepted)
#%%-----------------------------------Testing Tunisian set------------------------------------------------

X1,X2,label=get_tunisian_set()
print(len(X1))
print(len(X2))
print(len(label))

not_accepted_distances, accepted_distances,distances,cos_accepted,cos_not_accepted=evaluate(model,X1,X2,label,len(X1),200,350,'Tun_evaluation_80_[200,350]_cos_sim.txt')

plot_distances(accepted_distances,cos_accepted,not_accepted_distances,cos_not_accepted)


#%% VISUALIZE ROC Curve
from visual_evaluation import *
acc,recall,F1,thresholds_keras,fpr_keras,tpr_keras,auc_keras=get_output(model)
plot_AUC(fpr_keras,tpr_keras,auc_keras)

plot_distances(accepted_distances,cos_accepted,not_accepted_distances,cos_not_accepted)

#%%
