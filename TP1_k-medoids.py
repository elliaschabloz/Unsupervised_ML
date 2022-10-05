import numpy as np
import matplotlib.pyplot as plt
import time


from sklearn import metrics
import kmedoids
from sklearn import cluster
from scipy.io import arff

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances


def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data


# A f f i c h a g e en 2D
# E x t r a i r e chaque v a l e u r de f e a t u r e s pour en f a i r e une l i s t e
# Ex pour f 0 = [ − 0 . 4 9 9 2 6 1 , −1.51369 , −1.60321 , . . . ]
# Ex pour f 1 = [ − 0 . 0 6 1 2 3 5 6 , 0 . 2 6 5 4 4 6 , 0 . 3 6 2 0 3 9 , . . . ]
data=create_data("rings")
f0=[f[0] for f in data]
f1=[f[1] for f in data]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

def k_medoids(data):
    for k in range(2,10):
        print("k=",k)
        tps1 = time.time()
        distmatrix=euclidean_distances(data)
        fp=kmedoids.fasterpam(distmatrix,k)
        tps2=time.time()
        iter_kmed = fp.n_iter
        labels_kmed = fp.labels
        print("Loss with FasterPAM : " , fp.loss)
        plt.scatter(f0 , f1 , c=labels_kmed , s =8)
        plt.title(" Donnees apres clustering KMedoids " )
        plt.show()
        print(" nb clusters=" ,k , " , nb iter=" , iter_kmed , " runtime = " , round ((tps2 - tps1) * 1000,2), " ms")
    

k_medoids(data)





