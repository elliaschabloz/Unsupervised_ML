import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import metrics
from sklearn import cluster
from scipy.io import arff


import kmedoids

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

###################################
############ TP1.1 ################
###################################


# P a r s e r un f i c h i e r de donnees au format a r f f
# data e s t un t a b l e a u d ’ exemples avec pour chacun
# l a l i s t e d e s v a l e u r s d e s f e a t u r e s
#
# Dans l e s j e u x de donnees c o n s i d e r e s :
# i l y a 2 f e a t u r e s ( d i m e n s i o n 2 )
# Ex : [ [ − 0 . 4 9 9 2 6 1 , − 0 . 0 6 1 2 3 5 6 ] ,
# [ − 1 . 5 1 3 6 9 , 0 . 2 6 5 4 4 6 ] ,
# [ − 1 . 6 0 3 2 1 , 0 . 3 6 2 0 3 9 ] , . . . . .
# ]
#
# Note : chaque exemple du j e u de donnees c o n t i e n t a u s s i un
# numero de c l u s t e r . On r e t i r e c e t t e i n f o r m a t i o n

def create_data(name):
    path='./dataset_p2/'
    databrut=(open(path+name+".txt", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data


# A f f i c h a g e en 2D
# E x t r a i r e chaque v a l e u r de f e a t u r e s pour en f a i r e une l i s t e
# Ex pour f 0 = [ − 0 . 4 9 9 2 6 1 , −1.51369 , −1.60321 , . . . ]
# Ex pour f 1 = [ − 0 . 0 6 1 2 3 5 6 , 0 . 2 6 5 4 4 6 , 0 . 3 6 2 0 3 9 , . . . ]
data=create_data("x1")
f0=[f[0] for f in data]
f1=[f[1] for f in data]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()


###################################
############ TP1.2 ################
###################################


#
# Les donnees s o n t dans datanp ( 2 d i m e n s i o n s )
# f 0 : v a l e u r s s u r l a p r e m i e r e d i m e n s i o n
# f 1 : v a l e u r s u r l a deuxieme d i m e n s i o n
#
"""
print("Appel KMeans pour une valeur fixee de k" )
tps1=time.time()
k=3
model=cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(data)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees apres clustering Kmeans" )
plt.show()
print("nb clusters=",k,", nb iter =" , iteration, " , runtime = " , round ((tps2 - tps1) * 1000 ,2), " ms " )
"""

#####Partie 2.2
def k_means_davies(data):
    labels_kmeans=[]
    min_davies_score = 9999
    min_davies_k = 0
    for k in range(2,10):
        print("k=",k)
        tps1=time.time()
        model=cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(data)
        tps2 = time.time()
        labels = model.labels_
        #labels_kmeans.append(labels)
        davies_score = metrics.davies_bouldin_score(data, labels)
        if(davies_score<min_davies_score):
            min_davies_score=davies_score
            min_davies_k=k
        iteration = model.n_iter_
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Donnees apres clustering Kmeans")
        plt.show()
        print("nb clusters=",k,", nb iter =" , iteration, "score=", davies_score,", runtime = ", round ((tps2-tps1) * 1000, 2), " ms")
        labels_kmeans = labels
        
    print("Le meilleur nombre de clusters est :", min_davies_k, "avec un score de", min_davies_score)
    return labels_kmeans

####Partie2.3
"""Méthode kmeans fonctionne bien avec des  nuages de points. Quand on a des anneaux (par exemple rings), cela 
fonctionne moins bien"""

####Partie2.4
k_means_davies(data)

def k_medoids(data):
    max_score = -1
    max_k = 0
    labels_pred=[]
    for k in range(2,10):
        print("k=",k)
        tps1 = time.time()
        distmatrix=euclidean_distances(data)
        fp=kmedoids.fasterpam(distmatrix,k)
        tps2=time.time()
        iter_kmed = fp.n_iter
        labels_kmed = fp.labels
            
        silhouette_score= metrics.silhouette_score(data, labels_kmed, metric='euclidean')
        silhouette_score_man = metrics.silhouette_score(data, labels_kmed, metric='manhattan')
        if(silhouette_score>max_score):
            max_score=silhouette_score
            max_k=k
        
            
        #print("Loss with FasterPAM : " , fp.loss)
        plt.scatter(f0 , f1 , c=labels_kmed , s =8)
        plt.title(" Donnees apres clustering KMedoids " )
        plt.show()
        print(" nb clusters=" ,k , " , nb iter=" , iter_kmed , "score euclidean= ", silhouette_score,"score manhattan= ", silhouette_score_man," runtime = " , round ((tps2 - tps1) * 1000,2), " ms")
       
    
    print("Le meilleur nombre de clusters est :", max_k, "avec un score de", max_score)

k_medoids(data)


"""Pour la valeur de k identifiée comme la plus pertinente, comparez les résultats obtenus par le
clustering k-means et par le clustering k-medoids à l’aide d’indicateurs comme rand_score ou
mutual_information."""

def calc_rand_score(k):
    model=cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    labels_kmeans = model.labels_
    
    distmatrix=euclidean_distances(data)
    fp=kmedoids.fasterpam(distmatrix,k)
    labels_kmed = fp.labels
    
    rand_score = metrics.rand_score(labels_kmeans, labels_kmed)
    print("Le rand score est de", rand_score, "pour k =", k)


calc_rand_score(3)

"""L'impact de la metric est minime"""
    










