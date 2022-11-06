# -*- coding: utf-8 -*-
"""
TP Unsepervised Machine Learning

@author: CHAUZY Célia
@author: CHABLOZ Ellias
"""


""" Imports """
import numpy as np
import matplotlib.pyplot as plt
import time
import kmedoids
import scipy.cluster.hierarchy as shc
import pandas as pd

from scipy.io import arff
from sklearn import metrics
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances


""" Data preparation """
def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    data=[[x[0], x[1]] for x in databrut[0]]
    return data

def create_data_mystere(name):
    path='./dataset-rapport/'
    databrut=pd.read_csv(path+name+".txt", sep=" ", encoding="ISO-8859-1", skipinitialspace=True)
    data=databrut.to_numpy()
    return data

data=create_data("donut1")
f0=[f[0] for f in data]
f1=[f[1] for f in data]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()


""" K-Means """
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


""" K-Medoids """
def k_medoids(data):
    max_score = -1
    max_k = 0
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
    

""" K-Means VS K-Medoids Comparison """
def calc_rand_score(k):
    model=cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    labels_kmeans = model.labels_
    
    distmatrix=euclidean_distances(data)
    fp=kmedoids.fasterpam(distmatrix,k)
    labels_kmed = fp.labels
    
    rand_score = metrics.rand_score(labels_kmeans, labels_kmed)
    print("Le rand score est de", rand_score, "pour k =", k)
    
    
""" Dendrogramme """
def create_dendro():
    print ( "Dendrogramme 'single' donnees initiales")
    linked_mat = shc.linkage(data, 'single')
    plt.figure(figsize = (12, 12))
    shc.dendrogram(linked_mat, 
                   orientation = 'top',
                   distance_sort = 'descending',
                   show_leaf_counts = False )
    plt.show ()


""" Clustering Agglomeratif """
def hierachical(mode):
# mode 0 is distance threshold mode 1 is number of cluster
    itera = 0
    if mode==int(0) :
        max_score = -1
        best_dist = 0
        best_k = 0
        """
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
        """
    if mode==int(1):  
        max_score = -1
        best_k = 0
        
        link = ['single', 'average', 'complete', 'ward']

        for l in link :
            
            for k in range(2, 10):
            # set thringse number of clusters
                #print(k)
                tps1 = time.time()
                model = cluster.AgglomerativeClustering(linkage = l, n_clusters = k)
                model = model.fit(data)
                tps2 = time.time()
                labels = model.labels_
                
                silhouette_score= metrics.silhouette_score(data, labels, metric='euclidean')
                #silhouette_score_man = metrics.silhouette_score(datanp, labels, metric='manhattan')
                if(silhouette_score>max_score):
                    max_score=silhouette_score
                    best_k=k                
                
                leaves = model.n_leaves_
                # Affichage clustering
                plt.scatter(f0, f1, c = labels, s=8)
                plt.title(" Resultat du clustering ")
                plt.show()
                #print("iter=",itera," nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = ", round(( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
            
            print("Le meilleur k est :", best_k, "avec un score de", max_score)