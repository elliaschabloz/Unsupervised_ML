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
"""
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()
"""

""" K-Means """
"""
Indice de Davies-Bouldin : Plus l'indice est faible, meilleure est la qualité du clustering'
"""
def k_means_davies(data, dataname):
    best_labels=[]
    min_davies_score = 9999
    min_davies_k = 0
    for k in range(2,10):
        tps1=time.time()
        model=cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(data)
        tps2 = time.time()
        runtime = round((tps2-tps1) * 1000, 2)
        labels = model.labels_
        davies_score = metrics.davies_bouldin_score(data, labels)
        if(davies_score<min_davies_score):
            min_davies_score=davies_score
            min_davies_k=k
            best_labels = labels
        #iteration = model.n_iter_
        
        #print("nb clusters=",k,", nb iter =" , iteration, "score=", davies_score,", runtime = ", runtime, " ms")
        
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]    
    plt.scatter(f0, f1, c=best_labels, s=8)
    plt.title("Donnees apres clustering Kmeans")
    plt.show()    
    print(dataname," : ","Le meilleur nombre de clusters est :", min_davies_k, "avec un score de", min_davies_score)
    return best_labels, min_davies_k, min_davies_score, runtime


""" K-Medoids """
"""
Silhouette Score : Varie de -1 à 1 ; Plus le score est proche de 1, meilleur est le cluster
dist can be : "euc" or "man" for euclidean or manhattan
"""
def k_medoids(data, dataname, dist="euc"):
    best_score = -1
    best_k = 0
    best_labels = []
    for k in range(2,10):
        tps1 = time.time()
        
        if (dist == "euc") :
            distmatrix=euclidean_distances(data)
        elif (dist == "man") :
            distmatrix=manhattan_distances(data)
            
        fp=kmedoids.fasterpam(distmatrix,k)
        tps2=time.time()
        runtime = round ((tps2-tps1) * 1000, 2)
        #iter_kmed = fp.n_iter
        labels_kmed = fp.labels
            
        silhouette_score = metrics.silhouette_score(data, labels_kmed, metric='euclidean')
        if(silhouette_score>best_score):
            best_score = silhouette_score
            best_k = k
            best_labels = labels_kmed
        
        #print(" nb clusters=" ,k , " , nb iter=" , iter_kmed , "score euclidean= ", silhouette_score, " runtime = " , round ((tps2 - tps1) * 1000,2), " ms")
    
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]   
    plt.scatter(f0 , f1 , c=best_labels , s=8)
    plt.title(" Donnees apres clustering KMedoids " )
    plt.show()
    print(dataname," : ","Le meilleur nombre de clusters est :", best_k, "avec un score de", best_score)
    return best_labels, best_k, best_score, runtime

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
def hierachical(data, dataname, mode):
#mode 0 is distance threshold mode 1 is number of cluster

    f0=[f[0] for f in data]
    f1=[f[1] for f in data] 
    
    link = ['single', 'average', 'complete', 'ward']
    
    best_labels = []
    best_dist = 0
    best_score = -1
    best_k = 0
    runtime = 0
    
    best_labels_global = []
    best_score_global = -1
    best_dist_global = 0
    best_k_global = 0
    runtime_global = 0
    
    for l in link : 
        
        if mode==int(0) :
            xmin=np.min(f0)
            xmax=np.max(f0)
            xrange=np.abs(xmax-xmin)
            for dist in np.arange(xrange/1000, xrange, xrange/100.0):
                #set distance_threshold (0 ensures we compute the full tree)
           
                tps1 = time.time()
                model = cluster.AgglomerativeClustering(distance_threshold = dist, linkage =l, n_clusters = None)
                model = model.fit(data)
                tps2 = time.time()
                runtime = round ((tps2-tps1) * 1000, 2)
                labels = model.labels_
                k = model.n_clusters_
                
                if(k>=2) : 
                    silhouette_score= metrics.silhouette_score(data, labels, metric='euclidean')
                    if(silhouette_score>best_score):
                        best_score=silhouette_score
                        best_dist=dist
                        best_k=k
                        best_labels = labels
            
            plt.scatter(f0, f1, c = best_labels, s=8)
            plt.title(" Resultat du clustering agglomératif en mode Distance avec linkage=" + l)
            plt.show()
            print("(",dataname,",",l,"):","La meilleure distance est :", best_dist, "avec un score de", best_score, "et un k associé à", best_k)
                
        if mode==int(1):
    
            
                
            for k in range(2, 10):
            # set the number of clusters
                tps1 = time.time()
                model = cluster.AgglomerativeClustering(linkage = l, n_clusters = k)
                model = model.fit(data)
                tps2 = time.time()
                runtime = round ((tps2-tps1) * 1000, 2)
                labels = model.labels_
                
                silhouette_score= metrics.silhouette_score(data, labels, metric='euclidean')
                if(silhouette_score>best_score):
                    best_score=silhouette_score
                    best_k=k                
                    best_labels = labels
                
            plt.scatter(f0, f1, c = best_labels, s=8)
            plt.title(" Resultat du clustering agglomératif en mode Cluster avec linkage=" + l)
            plt.show()
            print("Le meilleur k est :", best_k, "avec un score de", best_score)
                
        #comparison between linkages
        if(best_score>best_score_global):
            best_score_global=best_score
            best_dist_global=best_dist
            best_k_global=best_k
            best_labels_global = best_labels
            runtime_global = runtime
            best_linkage = l
        
    return best_labels_global, best_k_global, best_score_global, runtime_global, best_dist_global, best_linkage


     
""" Exploitation """

#K-means 
def kmeans_df(): 
    dataframe_kmeans = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    examples = ["triangle1", "diamond9", "xclara", "donut1", "xor"]
    
    for e in examples : 
        data=create_data(e)
        f0=[f[0] for f in data]
        f1=[f[1] for f in data]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t = k_means_davies(data, e)
        
        new_entry = {"Example":e, "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}
        dataframe_kmeans = dataframe_kmeans.append(new_entry, ignore_index=True)
    return dataframe_kmeans

#print("------ Running DataFrame for K-Means ------\n")
#dataframe_kmeans = kmeans_df()
#print("------ End of DataFrame for K-Means ------\n") 


#K-medoids 
def kmedoids_df():
    
    dataframe_kmedoids = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    examples = ["triangle1", "diamond9", "xclara", "donut1", "xor"]
    
    for e in examples : 
        data=create_data(e)
        f0=[f[0] for f in data]
        f1=[f[1] for f in data]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t = k_medoids(data, e)
        
        new_entry = {"Example":e, "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}
        dataframe_kmedoids = dataframe_kmedoids.append(new_entry, ignore_index=True)
    return dataframe_kmedoids

# print("------ Running DataFrame for K-Medoids ------\n")        
# dataframe_kmedoids = kmedoids_df()
# print("\n------ End of DataFrame for K-Medoids ------\n") 


#Agglo

def agglo_df():
    
    dataframe_agglo_dist = pd.DataFrame(columns=["Example", "Linkage", "Nb of Cluster", "Distance Treshold", "Score", "Runtime (ms)"])
    dataframe_agglo_clust = pd.DataFrame(columns=["Example", "Linkage", "Nb of Cluster", "Score", "Runtime  (ms)"])
    examples = ["triangle1", "diamond9", "xclara", "donut1", "xor"]
    
    for e in examples : 
        data=create_data(e)
        f0=[f[0] for f in data]
        f1=[f[1] for f in data]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t,d,link = hierachical(data, e, 0)
        dist_entry = {"Example":e, "Linkage":link, "Nb of Clusters":k,"Distance Treshold":d, "Score":s, "Runtime (ms)":t}
        dataframe_agglo_dist = dataframe_agglo_dist.append(dist_entry, ignore_index=True)
        
        l,k,s,t,d,link = hierachical(data, 1)
        clust_entry = {"Example":e, "Linkage":link, "Nb of Clusters":k, "Score":s, "Runtime":t}
        dataframe_agglo_clust = dataframe_agglo_clust.append(clust_entry, ignore_index=True)
    return dataframe_agglo_dist, dataframe_agglo_clust
    #return dataframe_agglo_dist
    
# print("------ Running DataFrame for Agglo ------\n")        
# dataframe_agglo_dist, dataframe_agglo_clust = agglo_df()
# #dataframe_agglo_dist = agglo_df()
# print("\n------ End of DataFrame for Agglo ------\n")     
    
    
    
    