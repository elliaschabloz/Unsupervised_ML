import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn import cluster
from sklearn import metrics
from scipy.io import arff


def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data

def plot_k_nearest_neighbors(k):
    #Distances k plus proches voisins
    # Donnees dans X
    neigh = NearestNeighbors ( n_neighbors = k )
    neigh.fit(datanp)
    distances,indices = neigh.kneighbors(datanp)
    # retirer le point " origine "
    newDistances = np.asarray( [np.average(distances[i][1:]) for i in range (0,distances.shape[0])])
    trie = np.sort(newDistances)
    title=("Plus proches voisins(", k, ")")
    plt.title(title)
    plt.plot(trie);
    plt.show()


def find_eps(min_samples, e_min, e_step, e_nb): 
    max_score = -1
    best_eps = 0
    best_labels=[]
    
    for e in range(0, e_nb):
        eps=e*e_step+e_min
        model = cluster.DBSCAN(eps=eps,  min_samples=min_samples, metric='euclidean')
        model = model.fit(datanp)
        labels = model.labels_        
        silhouette_score= metrics.silhouette_score(datanp, labels, metric='euclidean')
        if(silhouette_score>max_score):
            max_score=silhouette_score
            best_eps=eps
            best_labels=labels
    nb_clusters = len(set(best_labels)) - (1 if -1 in labels else 0)
    print("Il y a", nb_clusters, "clusters avec le meilleur epsilon (", best_eps, ") qui a un score de", max_score)
    plt.scatter(f0, f1, c = best_labels, s=8)
    title="Resultat du clustering", best_eps
    plt.title(title)
    plt.show()


datanp=create_data("donut1")
f0=[f[0] for f in datanp]
f1=[f[1] for f in datanp]


plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

plot_k_nearest_neighbors(5)

find_eps(5, 0.002, 0.001, 20)
