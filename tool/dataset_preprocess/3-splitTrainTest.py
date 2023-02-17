import os
import shutil
from tqdm import tqdm

rateOfTrainTest = 0.5

def processData(datalist,targetDir):
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    for dir in tqdm(datalist):
        name = dir.split("/")[-1]
        if os.path.exists(os.path.join(targetDir,name)):
            continue
        print(dir)
        shutil.copytree(dir,os.path.join(targetDir,name))



def main():
    rootDir = "/media/dmmm/CE31-3598/DataSets/DenseCV_Data/improvedOriData"
    trainDir = os.path.join(rootDir, "train")
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    classListDrone = os.listdir(os.path.join(rootDir, "newdata/drone"))
    for i in range(len(classListDrone)):
        classListDrone[i] = os.path.join(os.path.join(rootDir, "newdata/drone"),classListDrone[i])
    classListSatellite = os.listdir(os.path.join(rootDir, "newdata/satellite"))
    for i in range(len(classListDrone)):
        classListSatellite[i] = os.path.join(os.path.join(rootDir, "newdata/satellite"), classListSatellite[i])
    numOfClass = len(classListDrone)
    numOfClassForTrain = int(numOfClass * rateOfTrainTest)
    numOfClassForTest = numOfClass - numOfClassForTrain
    sorted(classListDrone)
    sorted(classListSatellite)
    ClassForTrainDrone = classListDrone[:numOfClassForTrain]
    ClassForTrainSatellite = classListSatellite[:numOfClassForTrain]
    ClassForTestDrone = classListDrone[numOfClassForTest:]
    ClassForTestSatellite = classListSatellite[numOfClassForTest:]

    # process train data
    processData(ClassForTrainDrone,os.path.join(trainDir,"drone"))
    processData(ClassForTrainSatellite,os.path.join(trainDir,"satellite"))
    testDir = os.path.join(rootDir, "test")
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    query_drone = os.path.join(testDir,"query_drone")
    gallery_drone = os.path.join(testDir,"gallery_drone")
    query_satellite = os.path.join(testDir,"query_satellite")
    gallery_satellite = os.path.join(testDir,"gallery_satellite")

    if not os.path.exists(query_drone):
        os.mkdir(query_drone)
        os.mkdir(gallery_drone)
        os.mkdir(query_satellite)
        os.mkdir(gallery_satellite)

    # process test data
    processData(ClassForTestDrone,query_drone)
    processData(ClassForTestSatellite,query_satellite)
    processData(classListDrone,gallery_drone)
    processData(classListSatellite,gallery_satellite)



if __name__ == '__main__':
    main()