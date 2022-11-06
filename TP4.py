import hdbscan
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy.io import arff


def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data


def find_min_cluster_size(min_size, max_size, datanp, f0, f1):
    max_score = -1
    best_size = 0
    best_labels=[]
    for size in range(min_size, max_size):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        cluster_labels = clusterer.fit_predict(datanp)
        silhouette_score= metrics.silhouette_score(datanp, cluster_labels, metric='euclidean')
        
        if(silhouette_score>max_score):
            max_score=silhouette_score
            best_size=size
            best_labels=cluster_labels
    print("La meilleure taille de cluster min est :", best_size, "avec un score de", max_score)
    plt.scatter(f0, f1, c = best_labels, s=8)
    title="Resultat du clustering"
    plt.title(title)
    plt.show()



examples = ["donut1", "xclara", "smile1", "diamond9", "shapes", "banana", "cassini"]    
for e in examples:
    datanp=create_data(e)
    f0=[f[0] for f in datanp]
    f1=[f[1] for f in datanp]
    find_min_cluster_size(3, 10, datanp, f0, f1)