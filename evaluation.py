import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import csv
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss

import itertools
from sklearn.metrics import confusion_matrix

currentPath = os.getcwd()
print(currentPath)

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
print(gpus)

trainDB = currentPath + os.sep + "dataset" + os.sep + "train20"
valDB = currentPath + os.sep + "dataset"+ os.sep +"val20"
batch_size = 10

model_name = "DenseNet161Model"
classList = os.listdir(trainDB)

evaluationTrainDBFile = model_name + "traindb.csv"
evaluationValDBFile = model_name + "valdb.csv"

evaluationAccTrainDBFile = model_name + "Acctraindb.csv"
evaluationAccValDBFile = model_name + "Accvaldb.csv"

img = mpimg.imread(trainDB + os.sep + "stadium" + os.sep + "stadium_449_5_msrgb.jpg")
imgplot = plt.imshow(img)

img_rows, img_cols, img_channel = 224, 224, 3
num_categories = len(classList)

train_data_gen = ImageDataGenerator()
    
train_generator = train_data_gen.flow_from_directory(
    directory=trainDB,
    class_mode='categorical'
)

val_data_gen = ImageDataGenerator()
    
val_generator = val_data_gen.flow_from_directory(
    directory=valDB,
    class_mode='categorical'
)

print(train_generator.class_indices)
imgs, labels = next(train_generator)

# Models
model = currentPath + os.sep + "models_weights" + os.sep + model_name + ".h5"
loadedModel = load_model(model)

print(classList)

classTypeIndex = 0
classCnt = 0
classCorrectCnt = 0
predictionTrainList = list()
with open(evaluationTrainDBFile, mode='w') as db_file:
    with open(evaluationAccTrainDBFile, mode='w') as accdb_file: 
        db_writer = csv.writer(db_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        db_writer.writerow(['File', "InputClass", 'Prediction', 'PredictionStatus',"Percentage"]) 
        db_writeracc = csv.writer(accdb_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        db_writeracc.writerow([ "InputClass", "ClassAccuracy"])     
        accuracy = tensorflow.keras.metrics.CategoricalAccuracy()
        
        for classtype in classList:
            path = trainDB + os.sep + classtype + os.sep
            correctPredictionVector = np.zeros((1,len(classList)))
            correctPredictionVector[0][classTypeIndex] = 1.
            classTypeIndex = classTypeIndex + 1
            
            classCnt = 0
            classCorrectCnt = 0
            imagesPredict = [f for f in listdir(path) if isfile(join(path,f))]
            for file in imagesPredict:
                classCnt = classCnt + 1
                img = load_img(path + file, target_size=(img_rows, img_cols))
                tensorImage = img_to_array(img) /255.
                tensorImage = np.expand_dims(tensorImage, axis=0)
                prediction = loadedModel.predict(tensorImage, batch_size = 1)
                predictionTrainList.append(prediction[0].tolist())
                index = np.where(prediction[0] == max(prediction[0]))[0]
                if (0 != index[0].size):
                    print("file "+ file + " is: " + classList[int(index[0])])
                    accuracy.reset_states()
                    _ = accuracy.update_state(correctPredictionVector, prediction)
                    auxStringCorrectPrediction = "Incorrect"
                    if(1. == accuracy.result().numpy()):
                        classCorrectCnt = classCorrectCnt + 1
                        auxStringCorrectPrediction = "Correct"
                    db_writer.writerow([file, classtype , classList[int(index[0])], auxStringCorrectPrediction,max(prediction[0])])

                else:
                    db_writer.writerow([file,classtype, "Unknown", "Unknown",0])
            accuracyPercentage = classCorrectCnt / classCnt
            db_writeracc.writerow([classtype , accuracyPercentage])
            
classTypeIndex = 0
classCnt = 0
classCorrectCnt = 0
predictionValList = list()
with open(evaluationValDBFile, mode='w') as db_file:
    with open(evaluationAccValDBFile, mode='w') as accdb_file: 
        db_writer = csv.writer(db_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        db_writer.writerow(['File', "InputClass", 'Prediction', 'PredictionStatus',"Percentage"]) 
        db_writeracc = csv.writer(accdb_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        db_writeracc.writerow([ "InputClass", "ClassAccuracy"])     
        accuracy = tensorflow.keras.metrics.CategoricalAccuracy()
    
        for classtype in classList:
            path = valDB + os.sep + classtype + os.sep
            correctPredictionVector = np.zeros((1,len(classList)))
            correctPredictionVector[0][classTypeIndex] = 1.
            classTypeIndex = classTypeIndex + 1

            classCnt = 0
            classCorrectCnt = 0
            imagesPredict = [f for f in listdir(path) if isfile(join(path,f))]
            for file in imagesPredict:
                classCnt = classCnt + 1
                img = load_img(path + file, target_size=(img_rows, img_cols))
                tensorImage = img_to_array(img) /255.
                tensorImage = np.expand_dims(tensorImage, axis=0)
                prediction = loadedModel.predict(tensorImage, batch_size = 1)
                predictionValList.append(prediction[0].tolist())
                index = np.where(prediction[0] == max(prediction[0]))[0]
                if (0 != index[0].size):
                    print("file "+ file + " is: " + classList[int(index[0])])
                    accuracy.reset_states()
                    _ = accuracy.update_state(correctPredictionVector, prediction)
                    auxStringCorrectPrediction = "Incorrect"
                    if(1. == accuracy.result().numpy()):
                        classCorrectCnt = classCorrectCnt + 1
                        auxStringCorrectPrediction = "Correct"
                    db_writer.writerow([file, classtype , classList[int(index[0])], auxStringCorrectPrediction,max(prediction[0])])

                else:
                    db_writer.writerow([file,classtype, "Unknown", "Unknown",0])
            accuracyPercentage = classCorrectCnt / classCnt
            db_writeracc.writerow([classtype , accuracyPercentage])
            
modelTrainResultsDF = pd.read_csv(evaluationTrainDBFile,dtype={"InputClass": str, "Prediction": str, "PredictionStatus": str},)
print(modelTrainResultsDF.head(5))
print("F1 Score: ",f1_score(modelTrainResultsDF['InputClass'], modelTrainResultsDF['Prediction'], average='macro'))
print("Hamming Loss: ",hamming_loss(modelTrainResultsDF['InputClass'], modelTrainResultsDF['Prediction']))
print("Jaccard Score: ",jaccard_score(modelTrainResultsDF['InputClass'], modelTrainResultsDF['Prediction'], average='macro'))
print("log loss: ",log_loss(modelTrainResultsDF['InputClass'],  predictionTrainList))

modelValResultsDF = pd.read_csv(evaluationValDBFile,dtype={"InputClass": str, "Prediction": str, "PredictionStatus": str},)
print(modelValResultsDF.head(5))
print("F1 Score: ",f1_score(modelValResultsDF['InputClass'], modelValResultsDF['Prediction'], average='macro'))
print("Hamming Loss: ",hamming_loss(modelValResultsDF['InputClass'], modelValResultsDF['Prediction']))
print("Jaccard Score: ",jaccard_score(modelValResultsDF['InputClass'], modelValResultsDF['Prediction'], average='macro'))
print("log loss: ",log_loss(modelValResultsDF['InputClass'],  predictionValList))

y_true = modelTrainResultsDF['InputClass']
y_pred = modelTrainResultsDF['Prediction']
class_names=classList

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if(0.0 != cm[i, j]):
            plt.text(j, i, float("{:.3f}".format(cm[i, j])),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()