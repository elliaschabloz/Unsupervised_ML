import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.neighbors import NearestNeighbors
from sklearn import cluster
from sklearn import metrics
from scipy.io import arff



# Donnees dans datanp


def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data

datanp=create_data("cluto-t4-8k")
f0=[f[0] for f in datanp]
f1=[f[1] for f in datanp]


plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()


#Distances k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors ( n_neighbors = k )
neigh.fit(datanp)
distances,indices = neigh.kneighbors(datanp)
# retirer le point " origine "
newDistances = np.asarray( [np.average(distances[i][1:]) for i in range (0,distances.shape[0])])
trie = np.sort(newDistances)
plt.title("Plus proches voisins(5)")
plt.plot(trie);
plt.show()

print(np.mean(trie))

# max_score = -1
# best_eps = 0
# for e in range(2, 10):          

#     tps1 = time.time()
#     model = cluster.DBSCAN(eps=e,  min_samples=14, metric='euclidean')
#     model = model.fit(datanp)
#     tps2 = time.time()
#     labels = model.labels_
    
#     silhouette_score= metrics.silhouette_score(datanp, labels, metric='euclidean')
#     if(silhouette_score>max_score):
#         max_score=silhouette_score
#         best_eps=e 

#     # Affichage clustering
#     plt.scatter(f0, f1, c = labels, s=8)
#     plt.title(" Resultat du clustering ")
#     plt.show()

#print("Le meilleur k est :", best_eps, "avec un score de", max_score)