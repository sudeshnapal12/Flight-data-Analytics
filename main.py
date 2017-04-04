from sklearn import manifold

import pandas
from flask import Flask
from flask import render_template
import random

from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

labels = []  # clustering
random_samples = []
adaptive_samples = []
samplesize = 200
imp_ftrs = []

data_csv = pandas.read_csv('flights1987.csv', low_memory=False)
data_csv_original = pandas.read_csv('flights1987.csv', low_memory=False)
data_csv = data_csv.fillna(0)
data_csv_original = data_csv_original.fillna(0)
del data_csv['UniqueCarrier']
del data_csv['TailNum']
del data_csv['AirTime']
del data_csv['Origin']
del data_csv['Dest']
del data_csv['TaxiIn']
del data_csv['TaxiOut']
del data_csv['Cancelled']
del data_csv['CancellationCode']
del data_csv['Diverted']
del data_csv['CarrierDelay']
del data_csv['WeatherDelay']
del data_csv['NASDelay']
del data_csv['SecurityDelay']
del data_csv['LateAircraftDelay']

ftrs = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime']
scaler = StandardScaler()
data_csv[ftrs] = scaler.fit_transform(data_csv[ftrs])

@app.route("/")
def d3():
    return render_template('index.html')

# TASK 1b - find K in k-means.
def plot_kmeans_elbow():
    print("Inside Plot elbow");
    global data_csv_original
    features = data_csv_original[ftrs]

    k = range(1, 11)

    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]

    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    clust_indx = [np.argmin(kd, axis=1) for kd in k_distance]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]

    with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
    to_sum_square = np.sum(pdist(features) ** 2) / features.shape[0]
    bet_sum_square = to_sum_square - with_in_sum_square

    kidx = 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, avg_within, 'g*-')
    ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow plot of KMeans clustering')
    print("End of plotElbow")
    plt.show()

def clustering():
    plot_kmeans_elbow()
    features = data_csv[ftrs]
    k = 3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    kmeans_centres = kmeans.cluster_centers_
    labels = kmeans.labels_
    data_csv['kcluster'] = pandas.Series(labels)

# Task 1a
def random_sampling():
    # Random samples
    global data
    global data_csv
    global random_samples
    global samplesize
    features = data_csv[ftrs]
    data = np.array(features)
    rnd = random.sample(range(len(data_csv)), samplesize)
    for j in rnd:
        random_samples.append(data[j])

# Task 1a
def adaptive_sampling():
    # Adaptive samples
    global data_csv
    global adaptive_samples

    kcluster0 = data_csv[data_csv['kcluster'] == 0]
    kcluster1 = data_csv[data_csv['kcluster'] == 1]
    kcluster2 = data_csv[data_csv['kcluster'] == 2]

    size_kcluster0 = len(kcluster0) * samplesize / len(data_csv)
    size_kcluster1 = len(kcluster1) * samplesize / len(data_csv)
    size_kcluster2 = len(kcluster2) * samplesize / len(data_csv)

    sample_cluster0 = kcluster0.ix[random.sample(kcluster0.index, int(size_kcluster0))]
    sample_cluster1 = kcluster1.ix[random.sample(kcluster1.index, int(size_kcluster1))]
    sample_cluster2 = kcluster2.ix[random.sample(kcluster2.index, int(size_kcluster2))]

    adaptive_samples = pandas.concat([sample_cluster0, sample_cluster1, sample_cluster2])

def generate_eigenValues(data):
    # X_std = StandardScaler().fit_transform(data) #did it outside only
    # mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data.T)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)

    # centered_matrix = data - np.mean(data, axis=0)
    # cov = np.dot(centered_matrix.T, centered_matrix)
    # eig_values, eig_vectors = np.linalg.eig(cov)

    #Sorting Eigen Values
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    return eig_values, eig_vectors

@app.route("/pca_scree")
def scree_adaptive():
    print("Inside scree")
    # Plotting scree plot using random samples
    try:
        global adaptive_samples
        [eigenValues, eigenVectors] = generate_eigenValues(adaptive_samples[ftrs])
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(eigenValues)


def plot_intrinsic_dimensionality_pca(data, k):
    [eigenValues, eigenVectors] = generate_eigenValues(data)
    print eigenValues
    squaredLoadings = []
    ftrCount = len(eigenVectors)
    for ftrId in range(0, ftrCount):
        loadings = 0
        for compId in range(0, k):
            loadings = loadings + eigenVectors[compId][ftrId] * eigenVectors[compId][ftrId]
        squaredLoadings.append(loadings)

    print 'squaredLoadings', squaredLoadings
    # print eigenValues
    # plt.plot(eigenValues)
    # plt.show()
    return squaredLoadings

@app.route('/pca_random')
def pca_random():
    print 'in pca_random'
    data_columns = []
    try:
        global random_samples
        global imp_ftrs
        pca_data = PCA(n_components=2)
        X = random_samples
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_columns = pandas.DataFrame(X)
        # This is for tool-tip showcasing. Should have taken corresponding samples from random sampling or adaptive sampling
        # not the first 200 samples. So, needs to be changed similarly in all other functions.
        for i in range(0, 2):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]
        # We actually donot use clusterId in random sampling but is sent because otherwise Javascript will brea, because it expects 5 columns.
        data_columns['clusterid'] = data_csv['kcluster'][:samplesize]

        # data_columns['departure'] = data_csv['DepTime'][:samplesize]
        # data_columns['arrival'] = data_csv['ArrTime'][:samplesize]
        # pca_variance = pca_data.explained_variance_ratio_
        # data_columns['variance'] = pandas.DataFrame(pca_variance)[0]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/pca_adaptive')
def pca_adaptive():
    print("pca adaptive");
    data_columns = []
    try:
        global adaptive_samples
        global imp_ftrs
        X = adaptive_samples[ftrs]
        pca_data = PCA(n_components=2)
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_columns = pandas.DataFrame(X)
        for i in range(0, 2):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]
        data_columns['clusterid'] = np.nan
        x = 0
        for index, row in adaptive_samples.iterrows():
            data_columns['clusterid'][x] = row['kcluster']
            x = x + 1

            # print data_columns.head(200).to_string()
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)


@app.route('/mds_euclidean_random')
def mds_euclidean_random():
    data_columns = []
    try:
        global random_samples
        global imp_ftrs
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        for i in range(0, 3):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]
        data_columns['clusterid'] = data_csv['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)


@app.route('/mds_euclidean_adaptive')
def mds_euclidean_adaptive():
    data_columns = []
    try:
        global adaptive_samples
        global imp_ftrs
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[ftrs]
        similarity = pairwise_distances(X, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        for i in range(0, 3):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]

        data_columns['clusterid'] = np.nan
        x = 0
        for index, row in adaptive_samples.iterrows():
            data_columns['clusterid'][x] = row['kcluster']
            x = x + 1
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)


@app.route('/mds_correlation_random')
def mds_correlation_random():
    data_columns = []
    try:
        global random_samples
        global imp_ftrs
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        for i in range(0, 2):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]
        data_columns['clusterid'] = data_csv['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)


@app.route('/mds_correlation_adaptive')
def mds_correlation_adaptive():
    data_columns = []
    try:
        global adaptive_samples
        global imp_ftrs
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[ftrs]
        similarity = pairwise_distances(X, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        for i in range(0, 2):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]

        data_columns['clusterid'] = np.nan
        x = 0
        for index, row in adaptive_samples.iterrows():
            data_columns['clusterid'][x] = row['kcluster']
            x = x + 1
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)


@app.route('/scatter_matrix_random')
def scatter_matrix_random():
    try:
        global random_samples
        global imp_ftrs
        data_columns = pandas.DataFrame()
        for i in range(0, 3):
            data_columns[ftrs[imp_ftrs[i]]] = data_csv_original[ftrs[imp_ftrs[i]]][:samplesize]

        data_columns['clusterid'] = data_csv['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(data_columns)

@app.route('/scatter_matrix_adaptive')
def scatter_matrix_adaptive():
    try:
        global imp_ftrs
        data_columns = pandas.DataFrame()
        for i in range(0, 3):
            data_columns[ftrs[imp_ftrs[i]]] = adaptive_samples[ftrs[imp_ftrs[i]]][:samplesize]

        data_columns['clusterid'] = np.nan
        for index, row in adaptive_samples.iterrows():
            data_columns['clusterid'][index] = row['kcluster']
        data_columns = data_columns.reset_index(drop=True)
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

clustering()
random_sampling()
adaptive_sampling()
squared_loadings = plot_intrinsic_dimensionality_pca(data, 3)
imp_ftrs = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)

if __name__ == "__main__":
    app.run("localhost", 7777)
