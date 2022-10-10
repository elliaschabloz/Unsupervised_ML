import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import cluster
from sklearn import metrics
from scipy.io import arff


# Donnees dans datanp


def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data

datanp=create_data("banana")
f0=[f[0] for f in datanp]
f1=[f[1] for f in datanp]
xmin=np.min(f0)
xmax=np.max(f0)
print(xmin)
xrange = np.max(f0) - np.min(f0)

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

"""
print ( " Dendrogramme 'single' donnees initiales " )
linked_mat = shc.linkage ( datanp , 'single')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat ,
orientation = 'top',
distance_sort = 'descending',
show_leaf_counts = False )
plt.show ()"""

def hierachical(mode):
# mode 0 is distance threshold mode 1 is number of cluster
    itera = 0
    if mode==int(0) :
        max_score = -1
        best_dist = 0
        best_k = 0
        
        for dist in np.arange(xmin, xmax, xmax/10.0):
            itera = itera + 1
        #for dist in np.arange(0.0, 1.0, 0.1):
            
            # set di stance_threshold ( 0 ensures we compute the full tree )
       
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(distance_threshold = dist, linkage ='single', n_clusters = None)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            print(np.max(labels)+1)
            k = model.n_clusters_
            
            silhouette_score= metrics.silhouette_score(datanp, labels, metric='euclidean')
            #silhouette_score_man = metrics.silhouette_score(datanp, labels, metric='manhattan')
            if(silhouette_score>max_score):
                max_score=silhouette_score
                best_dist=dist
                best_k=k
                
            leaves = model.n_leaves_
            # Affichage clustering
            plt.scatter(f0, f1, c = labels, s=8)
            plt.title(" Resultat du clustering ")
            plt.show()
            print("iter=",itera,"dist=",dist ," nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = ", round(( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        
        print("La meilleure distance est :", best_dist, "avec un score de", max_score, "et un k associé à", best_k)
        
    if mode==int(1):  
        max_score = -1
        best_k = 0
        for k in range(2, 10):
        # set thringse number of clusters
            print(k)
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(linkage = 'single', n_clusters = k)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            kres = model.n_clusters_
            
            silhouette_score= metrics.silhouette_score(datanp, labels, metric='euclidean')
            #silhouette_score_man = metrics.silhouette_score(datanp, labels, metric='manhattan')
            if(silhouette_score>max_score):
                max_score=silhouette_score
                best_k=k                
            
            leaves = model.n_leaves_
            # Affichage clustering
            plt.scatter(f0, f1, c = labels, s=8)
            plt.title(" Resultat du clustering ")
            plt.show()
            print("iter=",itera," nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = ", round(( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        
        print("Le meilleur k est :", best_k, "avec un score de", max_score)

hierachical(int(0))

"""utiliser le drendrogramme pour déterminer la distance min à utiliser"""