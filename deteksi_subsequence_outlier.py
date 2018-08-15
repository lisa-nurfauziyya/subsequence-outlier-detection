import pandas as pd
import numpy as np
import glob 
import time
from array_split import array_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from numpy.linalg import pinv

#Load Dataset
def loadDataset(filename):
    data = (pd.read_csv(filename)).value
    return data

#Calculate Euclidean Distance
def EuclideanDist(x, y):
    return np.sqrt(np.sum((x-y)**2))

#Calculate Mahalanobis Distance
def MahalanobisDist(x, y):
    covariance_xy = np.cov(np.vstack((x,y)).T)
    inv_covariance_xy = pinv(covariance_xy)
    return np.sqrt(np.dot(np.dot((np.array(x)-np.array(y)).T,inv_covariance_xy), (np.array(x)-np.array(y))))

#Split value
def splitValue(data, subset):
    global myarray
    global subsets
    subsets = subset
    myarray = np.array(data, dtype = 'f') 
    if len(myarray)%subset != 0:
        myarray = np.pad(myarray, (0, subset - myarray.size%subset), mode = 'mean').reshape(-1, subset)
    else : 
        datas = []
        while len(data) > subset:
            pice = data[:subset]
            datas.append(pice)
            data = data[subset:]
        datas.append(data)
        myarray = np.array(datas)
    return myarray

#Clustering
def Clustering(arrays, k, iter):    
    cluster = np.array(array_split(arrays, k))

    global len_data1
    len_data1 = float(len(arrays))

    global centroids
    centroids = np.empty((k, subsets))
    for ind_i, i in enumerate(cluster):
        centroids[ind_i] = (np.sum(i, axis = 0)/len(i))

    global counter
    counter = 0
    for n in range(iter):        
        distances = []
        global labels
        labels = []
        global clusters
        counter += 1
        for i in range(len(arrays)):
            distance1 = []
            for j in range(len(centroids)):
                distance = MahalanobisDist(arrays[i], centroids[j])
                distance1.append(distance)
            distances.append(distance1)
        label = []
        for i in range(len(distances)):
            label = distances[i].index(min(distances[i]))
            labels.append(label)
        clusters = []
        for i in range(k):
            cluster = []
            clusters.append(cluster)
        for i in range(len(labels)):
            j = labels[i]
            clusters[j].append(arrays[i])

        centroids1 = np.empty((k, subsets))
        for i in range(len(clusters)):
            np.nan_to_num(clusters[i])
            centroids1[i] = (np.mean(clusters[i], axis = 0))
            centroids1[i] = np.nan_to_num(centroids1[i])
        centroids1 = np.asarray(centroids1)
        if np.array_equal(centroids, centroids1):
            break
        centroids = centroids1

    labels = np.array(labels)
    return labels

#Evaluasi Cluster Silhouette
def Silhouette(arrays):
    silhouette  = silhouette_score(arrays, labels)
    return silhouette

#Scoring
def Scoring(x):
    Scoring1 = []    
    Scoring2 = []
    label = []
    for ind_s, i in enumerate(clusters):
        len_assignments = len(clusters[ind_s])
        Scoring = ((len_assignments/len_data1)*(counter**2))
        Scoring1.append(Scoring)
    for i in range(len(labels)):
        for j in range(len(Scoring1)):
            if (labels[i] == j):
                labels[i] = Scoring1[j]
        Scoring2.append(labels[i])
    outlier = min(Scoring2)

    for x in Scoring2:
        if ( x != outlier):
            label.append(0)
        else:
            label.append(1)
    return label

#Confusion matriks
def con_matrix(x, y):
    expected = x
    predicted = y
    results = confusion_matrix(expected, predicted)
    print results
    TP = np.float(results[0][0])
    FP = np.float(results[0][1])
    FN = np.float(results[1][0])
    TN = np.float(results[1][1])
    accuracy = (TP + TN)/(TP + FP + FN + TN)*100
    sensitivity = TP / (TP + FN) * 100
    return 'akurasi : ', accuracy, 'sensitivitas : ', sensitivity

def main():
    a1 = []
    datas1 = []
    for filename in glob.glob('Dataset/A1Benchmark/real_00.csv'):
        start = time.time()
        data = pd.read_csv(filename)
        tanya1 = raw_input("Masukkan ukuran subsequence : ")
        n = int(tanya1)
        data = splitValue(data.value, n)
        tanya2 = raw_input("Masukkan jumlah cluster : ")
        o = int(tanya2)
        tanya3 = raw_input("Masukkan maksimal iterasi : ")
        p = int(tanya3)
        cluster = Clustering(data, o, p)
        a = Scoring(cluster)
        a1.append(a)

    for filenames in glob.glob('label eval/real10/real_00.csv'):
        data1 = pd.read_csv(filenames)
        datas = pd.Series(data1.is_anomaly, dtype = 'int32')
        datas = datas.tolist()
        datas1.append(datas)

    for i in range(len(a1)):
        con_matrix1 = con_matrix(a1[i], datas1[i])
        print con_matrix1
        end = time.time()
        waktu = end - start
        print waktu
main()