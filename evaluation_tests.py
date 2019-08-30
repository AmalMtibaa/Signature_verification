import glob
import random
import cv2
import h5py
import numpy as np
from data_visualisation import display_one,display_two
from scipy import spatial
from pre_processing import *

def load_saif_training_set():
    path_image = "DataSets/train set/*.png"
    images_paths = glob.glob(path_image)

    dict_gen = {}
    for i in images_paths:
        x = i.split('\\')[-1].split('_')[1]
        if x in dict_gen:
            dict_gen[x].append(i)
        else:
            dict_gen[x] = [i]
    return  dict_gen

def load_actual_training_set():
    path_image = "DataSets/actual_train_set/genuines/*.png"
    images_paths = glob.glob(path_image)

    dict_gen = {}
    for i in images_paths:
        x = i.split('\\')[-1].split('.')[0]
        if x[0:7] in dict_gen:
            dict_gen[x[0:7]].append(i)
        else:
            dict_gen[x[0:7]] = [i]

    return  dict_gen

def load_dutch_set():
    path_image = "C:/Users/Asus/Desktop/Stage/Dutch/TrainingSet/Offline Genuine/*.PNG"
    images_paths = glob.glob(path_image)

    dict_gen = {}
    for i in images_paths:
        x = i.split('\\')[-1].split('_')[0]
        if x in dict_gen:
            dict_gen[x].append(i)
        else:
            dict_gen[x] = [i]

    return dict_gen

def load_tunisian_set():
    path_image = "C:/Users/Asus/Desktop/Tunisian_signatures/*.PNG" # couldn't share this database for confidential resons
    images_paths = glob.glob(path_image)

    dict_gen = {}
    for i in images_paths:
        x = i.split('\\')[-1].split('_')[0]
        if x in dict_gen:
            dict_gen[x].append(i)
        else:
            dict_gen[x] = [i]

    return dict_gen

def get_test_data_saif_training_set():
    dict_gen=load_saif_training_set()
    X1 = []
    X2 = []
    label = []

    for i in range(1, 56):
        for j in range(4):
            X1.append(cv2.imread(dict_gen[str(i)][j]))
            X2.append(cv2.imread(dict_gen[str(i)][j + 5]))
            label.append(1)

        for j in range(4):
            X1.append(cv2.imread(dict_gen[str(i)][j]))
            X2.append(cv2.imread(dict_gen[str((i + j) % 55 + 1)][0]))
            label.append(0)

    return X1,X2,label

def get_actual_training_set():
    dict_gen=load_actual_training_set()
    X1 = []
    X2 = []
    label = []

    list_keys = list(dict_gen.keys())
    list_keys.remove('NFI-018')

    for i in list_keys:
        for j in range(3):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            X2.append(process_image_rgb(Image.open(dict_gen[i][j + 4])))
            label.append(1)
            #display_two(process_image_rgb(Image.open(dict_gen[i][j])).reshape(224,224,3)/255,process_image_rgb(Image.open(dict_gen[i][j + 4])).reshape(224,224,3)/255,str(i),str(j))

        for j in range(3):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            ind = i
            while ind == i:
                ind = random.choice(list_keys)
            X2.append(process_image_rgb(Image.open(dict_gen[ind][random.randint(1, 4)])))
            label.append(0)
    return X1,X2,label

def get_dutch_set():
    dict_gen=load_dutch_set()
    X1 = []
    X2 = []
    label = []

    list_keys = list(dict_gen.keys())

    for i in list_keys:
        for j in range(4):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            X2.append(process_image_rgb(Image.open(dict_gen[i][j + 5])))
            label.append(1)

        for j in range(4):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            ind = i
            while ind == i:
                ind = random.choice(list_keys)
            X2.append(process_image_rgb(Image.open(dict_gen[ind][random.randint(1, 4)])))
            label.append(0)
    return X1,X2,label

def get_tunisian_set():
    dict_gen = load_tunisian_set()
    X1 = []
    X2 = []
    label = []

    list_keys = list(dict_gen.keys())
    print(list_keys)
    for i in list_keys:
        for j in range(2):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            X2.append(process_image_rgb(Image.open(dict_gen[i][j + 1])))
            label.append(1)

        for j in range(2):
            X1.append(process_image_rgb(Image.open(dict_gen[i][j])))
            ind = i
            while ind == i:
                ind = random.choice(list_keys)
            X2.append(process_image_rgb(Image.open(dict_gen[ind][random.randint(0, 2)])))
            label.append(0)
    return X1, X2, label

def evaluate(model,X1,X2,label,test_number,max_accept,min_not_accept,history_filename,data_reshape=(1,224,224,3)):
    success_rate = 0
    fail_rate = 0
    cant_judge = 0
    distances = []

    not_accepted_distances=[]
    accepted_distances=[]
    cos_similarity_accepted=[]
    cos_similarity_not_accepted=[]

    file = open(history_filename, 'w')
    for i in range(test_number):
        x = model.predict([X1[i].reshape(data_reshape), X2[i].reshape(data_reshape), X1[i].reshape(data_reshape)])
        a1, p1, useless = x[0, 0, :], x[0, 1, :], x[0, 2, :]
        distance = np.linalg.norm(a1 - p1)
        cos_similarity=round(1-spatial.distance.cosine(a1,p1),2)
        if label[i]==1:
            accepted_distances.append(distance)
            cos_similarity_accepted.append(cos_similarity)
        if label[i]==0:
            not_accepted_distances.append(distance)
            cos_similarity_not_accepted.append(cos_similarity)

        distances.append(distance)
        decision=''
        if (distance <= max_accept and label[i] == 1):
            if(cos_similarity>=0.78):
                success_rate += 1
                decision='Accepted'
            else:
                cant_judge +=1
                decision='Can t judge'
        else:
            if (distance >= min_not_accept and label[i] == 0):
                success_rate += 1
                decision='Not Accepted'
            else:
                if (max_accept < distance < min_not_accept):
                    if(cos_similarity>=0.78 and label[i]==1):
                        success_rate += 1
                        decision = 'Accepted'
                    else:
                        if (cos_similarity< 0.75 and label[i] == 0):
                            success_rate += 1
                            decision = 'Not Accepted'
                        else:
                            cant_judge +=1
                            decision='Can t judge'
                else:
                    fail_rate += 1
                    print('We got Fail with distance =', distance ,'should be: ',label[i])
                    file.write('We got Fail with distance ='+ str(distance) +'should be: '+str(label[i])+'\n')


        file.write('Distance :' +str(distance)+' '+
              'Cos Similarity :'+str(1-spatial.distance.cosine(a1,p1))+' '
              'decision :'+decision+' '+
              'Actual label'+str(label[i])+' '
              'i: '+str(i)+'\n')

        print('Distance :' ,distance,' ',
              'Cos Similarity :',1-spatial.distance.cosine(a1,p1),' ',
              'decision :',decision,' ',
              'Actual label',label[i],
              'i: ',i)

    file.write('\n \n')
    print('success_rate = ', success_rate, 'success_range', success_rate/test_number)
    print('fail_rate = ', fail_rate, 'Fail_range', fail_rate/test_number)
    print('can t judge rate', cant_judge, 'Can t judge range', cant_judge/test_number)
    print('accuracy : ', (success_rate + cant_judge) / test_number)

    file.write('success_rate = '+ str(success_rate)+ 'success_range'+ str(success_rate/test_number)+'\n')
    file.write('fail_rate ='+ str(fail_rate)+ 'Fail_range'+ str(fail_rate/test_number)+'\n')
    file.write('can t judge rate = '+ str(fail_rate)+ 'Can t judge range'+ str(cant_judge/test_number)+'\n')
    file.write('accuracy : '+ str((success_rate + cant_judge) / test_number)+'\n')
    file.close()
    return not_accepted_distances,accepted_distances,distances,cos_similarity_accepted,cos_similarity_not_accepted

def evaluate_cos(model,X1,X2,label,test_number,max_accept,min_not_accept,history_filename,data_reshape=(1,224,224,3)):
    success_rate = 0
    fail_rate = 0
    cant_judge = 0
    distances = []

    not_accepted_distances=[]
    accepted_distances=[]
    cos_similarity_accepted=[]
    cos_similarity_not_accepted=[]

    file = open(history_filename, 'w')
    for i in range(test_number):
        x = model.predict([X1[i].reshape(data_reshape), X2[i].reshape(data_reshape), X1[i].reshape(data_reshape)])
        a1, p1, useless = x[0, 0, :], x[0, 1, :], x[0, 2, :]
        distance = np.linalg.norm(a1 - p1)
        cos_similarity=round(1-spatial.distance.cosine(a1,p1),2)
        if label[i]==1:
            accepted_distances.append(distance)
            cos_similarity_accepted.append(cos_similarity)
        if label[i]==0:
            not_accepted_distances.append(distance)
            cos_similarity_not_accepted.append(cos_similarity)

        distances.append(distance)
        decision=''
        if(cos_similarity>=0.77 and label[i]==1):
            success_rate+=1
            decision='Accepted'
        else:
            if(cos_similarity<0.75 and label[i]==0):
                decision='Not Accepted'
                success_rate+=1
            else:
                if(0.75<=cos_similarity<0.77):
                    cant_judge+=1
                    decision='Can t judge'
                else:
                    fail_rate+=1
                    print('We got Fail with distance =', distance, 'should be: ', label[i])
                    file.write('We got Fail with distance = ' + str(distance) + 'should be: ' + str(label[i]) + '\n')


        file.write('Distance :' +str(distance)+' '+
              'Cos Similarity :'+str(1-spatial.distance.cosine(a1,p1))+' '
              'decision :'+decision+' '+
              'Actual label'+str(label[i])+' '
              'i: '+str(i)+'\n')

        print('Distance :' ,distance,' ',
              'Cos Similarity :',1-spatial.distance.cosine(a1,p1),' ',
              'decision :',decision,' ',
              'Actual label',label[i],
              'i: ',i)

    file.write('\n \n')
    print('success_rate = ', success_rate, 'success_range', success_rate/test_number)
    print('fail_rate = ', fail_rate, 'Fail_range', fail_rate/test_number)
    print('can t judge rate', cant_judge, 'Can t judge range', cant_judge/test_number)
    print('accuracy : ', (success_rate + cant_judge) / test_number)

    file.write('success_rate = '+ str(success_rate)+ 'success_range'+ str(success_rate/test_number)+'\n')
    file.write('fail_rate ='+ str(fail_rate)+ 'Fail_range'+ str(fail_rate/test_number)+'\n')
    file.write('can t judge rate = '+ str(fail_rate)+ 'Can t judge range'+ str(cant_judge/test_number)+'\n')
    file.write('accuracy : '+ str((success_rate + cant_judge) / test_number)+'\n')
    file.close()
    return not_accepted_distances,accepted_distances,distances,cos_similarity_accepted,cos_similarity_not_accepted


