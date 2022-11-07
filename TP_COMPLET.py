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
import hdbscan

from scipy.io import arff
from sklearn import metrics
from sklearn import cluster
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors


""" Data preparation """
examples_p1 = ["triangle1", "diamond9", "xclara", "donut1", "xor"]
examples_dbscan_p1 = [["triangle1",0.5,0.1,20],
                      ["diamond9",0.08,0.005,20],
                      ["xclara",2,0.1,25],
                      ["donut1",0.002,0.0005,12],
                      ["xor",0.05,0.005,10]]

examples_p2 = ["x1","x2","x3","x4","zz1","zz2"]
example_y1 = ["y1"]

examples_dbscan_p2 = [["x1",10000,1000,20],
                      ["x2",10000,1000,20],
                      ["x3",5000,1000,20],
                      ["x4",5000,500,30],
                      ["zz1",1000,100,20],
                      ["zz2",5,1,45]]

def create_data(name):
    path='./artificial/'
    databrut=arff.loadarff(open(path+name+".arff", 'r'))
    if len(databrut) == 1 :
        data=[[x[0], x[1]] for x in databrut[0]]
    elif len(databrut) == 2 :
        data=[[x[0], x[1], x[2]] for x in databrut[0]]
    return data


def create_data_mystere(name):
    path='./dataset-rapport/'
    databrut=pd.read_csv(path+name+".txt", sep=" ", encoding="ISO-8859-1", skipinitialspace=True)
    data=databrut.to_numpy()
    return data

def create_data_list(mylist):
    datalist = []
    for d in mylist:
        datalist.append(create_data(d))
    return datalist

def create_data_list_dbscan(mylist):
    datalist = []
    for d in mylist:
        datalist.append(create_data(d[0]))
    return datalist
        
def create_data_mystere_list(mylist):
    datalist = []
    for d in mylist:
        datalist.append(create_data_mystere(d))
    return datalist

def create_data_list_mystere_dbscan(mylist):
    datalist = []
    for d in mylist:
        datalist.append(create_data_mystere(d[0]))
    return datalist
        
datalist_p1 = create_data_list(examples_p1)
datalist_p1_dbscan = create_data_list_dbscan(examples_dbscan_p1)

datalist_p2 = create_data_mystere_list(examples_p2)
datalist_y1 = create_data_mystere_list(example_y1)

datalist_dbscan_p2 = create_data_list_mystere_dbscan(examples_dbscan_p2)

# test=create_data("dense-disk-3000")
# f0=[f[0] for f in test]
# f1=[f[1] for f in test]
# f2=[f[2] for f in test]

# plt.scatter(f0, f1, f2, s=8)
# plt.title("Donnees initiales")
# plt.show()


""" K-Means """
"""
Indice de Davies-Bouldin : Plus l'indice est faible, meilleure est la qualité du clustering'
"""
def k_means_davies(data, dataname):
    best_labels=[]
    min_davies_score = 9999
    min_davies_k = 0
    best_runtime = 0
    
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
            best_runtime = runtime
        #iteration = model.n_iter_
        
        #print("nb clusters=",k,", nb iter =" , iteration, "score=", davies_score,", runtime = ", runtime, " ms")
        
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]    
    plt.scatter(f0, f1, c=best_labels, s=8)
    plt.title("Donnees apres clustering Kmeans")
    plt.show()    
    print(dataname," : ","Le meilleur nombre de clusters est :", min_davies_k, "avec un score de", min_davies_score)
    return best_labels, min_davies_k, min_davies_score, best_runtime


""" K-Medoids """
"""
Silhouette Score : Varie de -1 à 1 ; Plus le score est proche de 1, meilleur est le cluster
dist can be : "euc" or "man" for euclidean or manhattan
"""
def k_medoids(data, dataname, dist="euc"):
    best_score = -1
    best_k = 0
    best_labels = []
    best_runtime = 0
    
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
            best_runtime = runtime
        
        #print(" nb clusters=" ,k , " , nb iter=" , iter_kmed , "score euclidean= ", silhouette_score, " runtime = " , round ((tps2 - tps1) * 1000,2), " ms")
    
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]   
    plt.scatter(f0 , f1 , c=best_labels , s=8)
    plt.title(" Donnees apres clustering KMedoids " )
    plt.show()
    print(dataname," : ","Le meilleur nombre de clusters est :", best_k, "avec un score de", best_score)
    return best_labels, best_k, best_score, best_runtime

""" K-Means VS K-Medoids Comparison """
def calc_rand_score(data, k):
    model=cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    labels_kmeans = model.labels_
    
    distmatrix=euclidean_distances(data)
    fp=kmedoids.fasterpam(distmatrix,k)
    labels_kmed = fp.labels
    
    rand_score = metrics.rand_score(labels_kmeans, labels_kmed)
    print("Le rand score est de", rand_score, "pour k =", k)
    
    
""" Dendrogramme """
def create_dendro(data):
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
    best_runtime = 0
    
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
                        best_runtime = runtime
            
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
                    best_runtime = runtime
                
            plt.scatter(f0, f1, c = best_labels, s=8)
            plt.title(" Resultat du clustering agglomératif en mode Cluster avec linkage=" + l)
            plt.show()
            print("(",dataname,",",l,"):","Le meilleur k est :", best_k, "avec un score de", best_score)
                
        #comparison between linkages
        if(best_score>best_score_global):
            best_score_global=best_score
            best_dist_global=best_dist
            best_k_global=best_k
            best_labels_global = best_labels
            runtime_global = best_runtime
            best_linkage = l
        
    return best_labels_global, best_k_global, best_score_global, runtime_global, best_dist_global, best_linkage

""" DBSCAN """
def plot_k_nearest_neighbors(k, data, dataname):
    #Distances k plus proches voisins
    # Donnees dans X
    neigh = NearestNeighbors ( n_neighbors = k )
    neigh.fit(data)
    distances,indices = neigh.kneighbors(data)
    # retirer le point " origine "
    newDistances = np.asarray( [np.average(distances[i][1:]) for i in range (0,distances.shape[0])])
    trie = np.sort(newDistances)
    title=("Plus proches voisins(" + str(k) + ") pour " + dataname)
    plt.title(title)
    plt.plot(trie);
    plt.show()


def dbscan(data, dataname, min_samples, e_min, e_step, e_nb): 
    max_score = -1
    best_eps = 0
    best_labels=[]
    best_k = 0
    best_runtime = 0
    
    for e in range(0, e_nb):
        eps=e*e_step+e_min
        tps1=time.time()
        model = cluster.DBSCAN(eps=eps,  min_samples=min_samples, metric='euclidean')
        model = model.fit(data)
        tps2=time.time()
        runtime = round ((tps2-tps1) * 1000, 2)
        labels = model.labels_        
        
        silhouette_score= metrics.silhouette_score(data, labels, metric='euclidean')
        if(silhouette_score>max_score):
            max_score=silhouette_score
            best_eps=eps
            best_labels=labels
            best_runtime = runtime
            best_k = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            
    print(dataname," : ", "Le meilleur epsilon est :", best_eps,"avec ", best_k," clusters et un score de", max_score)
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]
    plt.scatter(f0, f1, c = best_labels, s=8)
    title="Resultat du clustering avec DBSCAN", best_eps
    plt.title(title)
    plt.show()
    return best_labels, best_k, max_score, best_runtime, best_eps
    
""" HDBSCAN """

def myhdbscan(data, dataname, min_size, max_size):
    max_score = -1
    best_size = 0
    best_labels=[]
    best_runtime = 0
    best_k = 0
    
    for size in range(min_size, max_size):
        tps1=time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        cluster_labels = clusterer.fit_predict(data)
        tps2=time.time()
        runtime = round ((tps2-tps1) * 1000, 2)
        silhouette_score= metrics.silhouette_score(data, cluster_labels, metric='euclidean')
        
        if(silhouette_score>max_score):
            max_score=silhouette_score
            best_size=size
            best_labels=cluster_labels
            best_runtime = runtime
            best_k = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            
    print(dataname," : ", "La meilleure taille de cluster min est :", best_size,"avec ", best_k, " clusters et un score de", max_score)
    f0=[f[0] for f in data]
    f1=[f[1] for f in data]
    plt.scatter(f0, f1, c = best_labels, s=8)
    title="Resultat du clustering avec HDBSCAN"
    plt.title(title)
    plt.show()
    return best_labels, best_k, max_score, best_runtime

""" Exploitation """

#K-means 
def kmeans_df(examples, datalist): 
    dataframe_kmeans = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    
    for i in range(len(examples)):
        f0=[f[0] for f in datalist[i]]
        f1=[f[1] for f in datalist[i]]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t = k_means_davies(datalist[i], examples[i])
        
        new_entry = {"Example":examples[i], "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}
        dataframe_kmeans = dataframe_kmeans.append(new_entry, ignore_index=True)
    return dataframe_kmeans

#print("------ Running DataFrame for K-Means ------\n")
#dataframe_kmeans = kmeans_df(examples_p1, datalist_p1)
#print("------ End of DataFrame for K-Means ------\n") 


#K-medoids 
def kmedoids_df(examples, datalist):
    
    dataframe_kmedoids = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    
    for i in range(len(examples)):
        f0=[f[0] for f in datalist[i]]
        f1=[f[1] for f in datalist[i]]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t = k_medoids(datalist[i], examples[i])
        
        new_entry = {"Example":examples[i], "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}
        dataframe_kmedoids = dataframe_kmedoids.append(new_entry, ignore_index=True)
    return dataframe_kmedoids

# print("------ Running DataFrame for K-Medoids ------\n")        
# dataframe_kmedoids = kmedoids_df(examples_p1, datalist_p1)
# print("\n------ End of DataFrame for K-Medoids ------\n") 


#Agglo

def agglo_df(examples, datalist):
    
    dataframe_agglo_dist = pd.DataFrame(columns=["Example", "Linkage", "Nb of Clusters", "Distance Treshold", "Score", "Runtime (ms)"])
    dataframe_agglo_clust = pd.DataFrame(columns=["Example", "Linkage", "Nb of Clusters", "Score", "Runtime  (ms)"])
    
    for i in range(len(examples)):
        f0=[f[0] for f in datalist[i]]
        f1=[f[1] for f in datalist[i]]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t,d,link = hierachical(datalist[i], examples[i], 0)
        dist_entry = {"Example":examples[i], "Linkage":link, "Nb of Clusters":k,"Distance Treshold":d, "Score":s, "Runtime (ms)":t}
        dataframe_agglo_dist = dataframe_agglo_dist.append(dist_entry, ignore_index=True)
        
        l,k,s,t,d,link = hierachical(datalist[i], examples[i], 1)
        clust_entry = {"Example":examples[i], "Linkage":link, "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}
        dataframe_agglo_clust = dataframe_agglo_clust.append(clust_entry, ignore_index=True)
        
    return dataframe_agglo_dist, dataframe_agglo_clust
    #return dataframe_agglo_dist
    
# print("------ Running DataFrame for Agglo ------\n")        
# dataframe_agglo_dist, dataframe_agglo_clust = agglo_df(examples_p1, datalist_p1)
# print("\n------ End of DataFrame for Agglo ------\n")     
    
    
#DBSCAN  

# Uncomment to see what we used to get our parameters for dbscan
# plot_k_nearest_neighbors(5, create_data("triangle1"),"triangle1")
# plot_k_nearest_neighbors(5, create_data("diamond9"),"diamond9")
# plot_k_nearest_neighbors(5, create_data("xclara"),"xclara")
# plot_k_nearest_neighbors(5, create_data("donut1"),"donut1")
# plot_k_nearest_neighbors(5, create_data("xor"),"xor")


def dbscan_df(examples_dbscan, datalist_dbscan): 
    dataframe_dbscan = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    
    #format : [dataset, min_samples, e_min, e_step, e_nb]
    
    
    for i in range(len(examples_dbscan)):
        f0=[f[0] for f in datalist_dbscan[i]]
        f1=[f[1] for f in datalist_dbscan[i]]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t,eps = dbscan(datalist_dbscan[i], examples_dbscan[i][0], 11, examples_dbscan[i][1],
                             examples_dbscan[i][2], examples_dbscan[i][3])
        
        new_entry = {"Example":examples_dbscan[i][0], "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}

        dataframe_dbscan = dataframe_dbscan.append(new_entry, ignore_index=True)
    return dataframe_dbscan

# print("------ Running DataFrame for Agglo ------\n") 
#dataframe_dbscan = dbscan_df(examples_dbscan_p1,datalist_p1_dbscan)
# print("\n------ End of DataFrame for Agglo ------\n") 

#HDBSCAN

def hdbscan_df(examples, datalist):
    dataframe_hdbscan = pd.DataFrame(columns=["Example", "Nb of Clusters", "Score", "Runtime (ms)"])
    
    for i in range(len(examples)): 
        f0=[f[0] for f in datalist[i]]
        f1=[f[1] for f in datalist[i]]
    
        plt.scatter(f0, f1, s=8)
        plt.title("Donnees initiales")
        plt.show()
        
        l,k,s,t = myhdbscan(datalist[i], examples[i], 5, 20)
        
        new_entry = {"Example":examples[i], "Nb of Clusters":k, "Score":s, "Runtime (ms)":t}

        dataframe_hdbscan = dataframe_hdbscan.append(new_entry, ignore_index=True)
    return dataframe_hdbscan

# print("------ Running DataFrame for hdbscan ------\n") 
# dataframe_hdbscan = hdbscan_df(examples_p1, datalist_p1)
# print("\n------ End of DataFrame for hdbscan ------\n") 

"""
PARTIE II : Données Mystères
"""
print("######### PARTIE 2 : DONNEES MYSTERES #########\n")

# temp = [["dense-disk-3000",0.5,0.01,10]]
# datalist_temp = create_data_list(["dense-disk-3000"])
# plot_k_nearest_neighbors(5, datalist_temp,temp)


# for i in range(len(examples_p2)):
#     plot_k_nearest_neighbors(5, datalist_p2[i],examples_p2[i])

# print("------ Running DataFrame for kmeans ------\n") 
# dataframe_kmeans = kmeans_df(temp, datalist_temp)
# print("\n------ End of DataFrame for kmeans ------\n") 

# print("------ Running DataFrame for kmedoids ------\n") 
# dataframe_kmedoids = kmedoids_df(examples_p2, datalist_p2)
# print("\n------ End of DataFrame for kmedoids ------\n") 

# print("------ Running DataFrame for agglo ------\n") 
# dataframe_agglo_dist, dataframe_agglo_clust = agglo_df(examples_p2, datalist_p2)
# print("\n------ End of DataFrame for agglo ------\n")  

# print("------ Running DataFrame for dbscan ------\n") 
# dataframe_dbscan = dbscan_df(temp, datalist_temp)
# print("\n------ End of DataFrame for dbscan ------\n")

# print("------ Running DataFrame for hdbscan ------\n") 
# dataframe_hdbscan = hdbscan_df(examples_p2, datalist_p2)
# print("\n------ End of DataFrame for hdbscan ------\n") 

#create_data_mystere