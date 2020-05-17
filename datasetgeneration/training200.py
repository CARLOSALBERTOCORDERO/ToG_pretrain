import os
import random as rnd
import math
import shutil

## Declare variables
dirpath = os.getcwd()
copyPath = dirpath + os.sep + "train200"
dirpathPH = dirpath.split(os.sep)
trainingFolder = dirpathPH.pop(0)
foldersUp = 1
pathLength = len(dirpathPH)
cnt = 0
imagePath = ""
categories = list()
categoriesNum = 0
categoryPath = ""
categoryList = list()
categoryImages = 0
randNum = 0
imageSampleNum = 15
copyPathAux = ""
originalFile = ""
copyFile = ""

## Get the Path of the images
try:
    shutil.rmtree(copyPath)
except:
    print('No previews categories')
os.mkdir(copyPath)
for folder in dirpathPH:
    if(cnt < (pathLength - foldersUp)):
        trainingFolder = trainingFolder + os.sep + folder
        cnt = cnt + 1

imagePath = trainingFolder + os.sep + "tesis" + os.sep + "fMoW" + os.sep + "training"
categories = os.listdir(imagePath)
categoriesNum = len(categories)
print ("categories : {}".format(categoriesNum) )
for category in categories:
    categoryPath = imagePath + os.sep + category
    categoryList = os.listdir(categoryPath)
    categoryImages = len(categoryList)
    print(category + ": {}".format(categoryImages))
    copyPathAux = copyPath + os.sep + category
    os.mkdir(copyPathAux)
    for imageTaken in range(0,imageSampleNum):
        randNum = math.floor(rnd.random() * categoryImages)
        originPath = categoryList[randNum]
        originalFile = categoryPath + os.sep + originPath
        copyFile = copyPath + os.sep + category + os.sep + originPath
        shutil.copyfile(originalFile, copyFile)
