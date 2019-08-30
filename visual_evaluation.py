from evaluation_tests import *
from pre_processing import *
from sklearn import metrics
import matplotlib.pyplot as plt

def predict_tunisian_set(model):
    X1,X2,label=get_actual_training_set()
    prediction=[]
    data_reshape=(1,224,224,3)
    prob_prediction=[]
    number_of_tests=len(X1)

    print(len(X1))
    print(len(X2))
    print(len(label))

    for i in range(len(X1)):
        x = model.predict([X1[i].reshape(data_reshape), X2[i].reshape(data_reshape), X1[i].reshape(data_reshape)])
        a1, p1, useless = x[0, 0, :], x[0, 1, :], x[0, 2, :]
        distance = np.linalg.norm(a1 - p1)
        cos_similarity=round(1-spatial.distance.cosine(a1,p1),2)

        print(distance+ " "+i)
        if (distance <= 200):
            if(cos_similarity>=0.78):
                prediction.append(1)
                prob_prediction.append(cos_similarity)
            else:
                #number_of_tests-=1 # can t judge won't be evaluate it
                del label[i]
        else:
            if (distance >= 350 ):
                prediction.append(0)
                prob_prediction.append(cos_similarity)
            else:
                    if(cos_similarity>=0.78):
                        prediction.append(1)
                        prob_prediction.append(cos_similarity)
                    else:
                        if (cos_similarity< 0.75):
                            prediction.append(1)
                            prob_prediction.append(cos_similarity)
                        else:
                            #number_of_tests=-1
                            del label[i]

    return prediction,prob_prediction,label



def get_output(model):
    '''
    Returns experiment output as a dict.
    Args:
        print_output: if True, prints results to stdout
    '''

    ## Test the classifier, save outputs
    y_pred, y_preba,label = predict_tunisian_set(model)
    fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(label, y_preba)

    # fpr = False Positive Rate
    # tpr = True Positive Rate
    # Thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.

    auc_keras = metrics.auc(fpr_keras, tpr_keras)  # Area under the Curve
    acc = metrics.accuracy_score(label, y_pred)  # accuracy of classification
    recall = metrics.recall_score(label, y_pred)  # recall = tpr / (tpr + fpr)
    F1 = metrics.f1_score(label, y_pred)  #


    print('Accuracy:' , acc)
    print('Recall: ', recall)
    print('F1 Score:', F1)
    print('-----------------------------------------------------')
    print('ROC Curve / AUC as follows:')
    print('Thresholds: ', thresholds_keras)
    print('False positive rates for each possible threshold:', fpr_keras)
    print('True positive rates for each possible threshold:', tpr_keras)
    print('AUC:', auc_keras)

    return acc,recall,F1,thresholds_keras,fpr_keras,tpr_keras,auc_keras

def plot_AUC( fpr_keras, tpr_keras, auc_keras):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=('CNN Keras Model', '(area = {:.3f})'.format(auc_keras)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # if save_img:
    #     if not os.path.exists(Experiment.IMG_DIR):
    #         os.makedirs(Experiment.IMG_DIR)
    #     figname = 'sig_id_{}.png'.format(self.sig_id)
    #     plt.savefig(Experiment.IMG_DIR + figname)
    # else:
    plt.show()

