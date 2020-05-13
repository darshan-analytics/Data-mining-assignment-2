import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("water-treatment.data",header=None, sep=",")
df1 = df.drop(df.columns[0], axis=1)

#print(df1)

df2 = df1.replace(to_replace = '?',value = 'Nan')
df12 = df2.replace(to_replace = 'Nan', value = df2.median(axis = 0))


df15 = df12
df12 = pd.DataFrame(df12)
df13  = df12.values.tolist()

for i in range(1,39):
    df15[i] = pd.to_numeric(df15[i], errors='ignore')

#print(df15.dtypes)
df15 = pd.DataFrame(df15)
df16 = pd.DataFrame(df15)
df16  = df16.values

#print(df16)
#print(df12)
#x = df14
#x = ','.join(str(v) for v in x)

#x = np.fromstring( x, dtype=np.float, sep=',' )
#print(x)
#print(df13)
#df15_norm = df15.apply(lambda  x: (x - x.min(axis=0)) / (x.max(axis= 0) - x.min(axis=0)))
scaler = MinMaxScaler()
df15_norm = scaler.fit_transform(df15)
df15_norm = pd.DataFrame(df15_norm)
df15_norm = df15_norm.values
#data_frame_normalized = normalize(df15, axis=0, norm='max')
#print(data_frame_normalized)
#print(df15)
#df13 = scalar.transform(df12)
#df15_norm = preprocessing.normalize(df15, norm='l2', axis=0, copy=True, return_norm=False)
#df15_norm = preprocessing.normalize(df15)
#print(df15_norm)
#df13 = pd.Dataframe(df12)
#print(df14)
#df12 = df14.replace(to_replace = "Nan", value = "")

#dfn = preprocessing.normalize(df13, norm = 'l1')

#df3 = df2.replace(to_replace = "Nan",value = df1.median(axis=1))
#df4 = df3.replace(to_replace = "Nan",value = "")
#print(df3)
#df5 = df4.median(axis=1)
#print(df5)



#print(df45)
####using the elbow method to find the optimal number of clusters
new = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters= i, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
    Kmeans.fit_transform(df15_norm)
    new.append(Kmeans.inertia_)
plt.plot(range(1,11),new)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('New')
plt.show()
#print(new)

#plt.plot(df16)
#plt.show()
#df15_norm = df15_norm.T

#print(df15_norm)
#applying K-Means
Kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
y_kmeans = Kmeans.fit_predict(df15_norm)
#df16 = scaler.fit_transform(df16)
#print(df16)
x_kemans = y_kmeans

#visualizing the result
#print(y_kmeans)
#x_std = df15_norm
plt.scatter(df15_norm[x_kemans == 0, 0], df15_norm[x_kemans == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(df15_norm[x_kemans == 1, 0], df15_norm[x_kemans == 1, 1], s = 100, c= 'green', label = 'Cluster 2')
plt.scatter(df15_norm[x_kemans == 2, 0], df15_norm[x_kemans == 2, 1], s = 100, c= 'blue', label = ' CLuster 3')
plt.scatter(df15_norm[x_kemans == 3, 0], df15_norm[x_kemans == 3, 1], s = 100, c= 'purple', label = ' CLuster 4')
plt.scatter(df15_norm[x_kemans == 4, 0], df15_norm[x_kemans == 4, 1], s = 100, c= 'orange', label = ' CLuster 5')
plt.scatter(df15_norm[x_kemans == 5, 0], df15_norm[x_kemans == 5, 1], s = 100, c= 'gray', label = ' CLuster 6')
plt.scatter(df15_norm[x_kemans == 6, 0], df15_norm[x_kemans == 6, 1], s = 100, c= 'black', label = ' CLuster 7')
#plt.scatter(df16[y_kmeans == 7, 0], df16[y_kmeans == 7, 1], s = 100, c= 'pink', label = ' CLuster 8')
#plt.scatter(df15_norm[y_kmeans == 0, 0], df15_norm[y_kmeans == 8, 1], s = 100, c= 'red', label = ' CLuster 1')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.title('NORMAL K-MEAN clusters')
plt.show()

scalar = preprocessing.MinMaxScaler()
df21 = scalar.fit_transform(df15_norm)
#df21 = df15_norm

data1 = pd.DataFrame(df21)
data = data1.values
SAMPLE_SIZE = 0.5
RANDOM_STATE = 42
NUM_CLUSTERS = 5  # k
NUM_ITER = 3          # n
NUM_ATTEMPTS = 3     # m
data_sample = data1.sample(frac=SAMPLE_SIZE, random_state=RANDOM_STATE, replace=False)
data_sample.shape

from sklearn.cluster import KMeans
km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)#, verbose=1)
km.fit(data_sample)
#print('Pre-clustering metrics')
#print('----------------------')
#print('Inertia:', km.inertia_)
#print('Centroids:', km.cluster_centers_)
final_cents = []
final_inert = []

for sample in range(NUM_ATTEMPTS):
    #print('\nCentroid attempt: ', sample)
    km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)  # , verbose=1)
    km.fit(data_sample)
    inertia_start = km.inertia_
    intertia_end = 0
    cents = km.cluster_centers_

    for iter in range(NUM_ITER):
        km = KMeans(n_clusters=NUM_CLUSTERS, init=cents, max_iter=1, n_init=1)
        km.fit(data_sample)
        #print('Iteration: ', iter)
        #print('Inertia:', km.inertia_)
        #print('Centroids:', km.cluster_centers_)
        inertia_end = km.inertia_
        cents = km.cluster_centers_
    final_cents.append(cents)
    final_inert.append(inertia_end)
    print('Difference between initial and final inertia: ', inertia_start - inertia_end)

# Get best centroids to use for full clustering
best_cents = final_cents[final_inert.index(min(final_inert))]
best_cents
km_full = KMeans(n_clusters=NUM_CLUSTERS, init=best_cents, max_iter=100, verbose=1, n_init=1)
n1 = km_full.fit_predict(data)
#print(n1)
km_naive = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=100, verbose=1, n_init=1)
n2 = km_naive.fit_predict(data)
outF = open("KmeansOutput.txt", "w")

i = 1
for i in range(1, 528):
    outF.write(str(i))
    outF.write("\t")
    outF.write(str(n2[i - 1]))
    outF.write("\n")
    i = + 1
outF.close()

n3 = n2
#print(n2)
'''plt.scatter(data[n1 == 0, 0], data[n1 == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(data[n1 == 1, 0], data[n1 == 1, 1], s = 100, c= 'green', label = 'Cluster 2')
plt.scatter(data[n1 == 2, 0], data[n1 == 2, 1], s = 100, c= 'blue', label = ' CLuster 3')
plt.scatter(data[n1 == 3, 0], data[n1 == 3, 1], s = 100, c= 'purple', label = ' CLuster 4')
plt.scatter(data[n1 == 4, 0], data[n1 == 4, 1], s = 100, c= 'orange', label = ' CLuster 5')#
plt.scatter(data[n1 == 5, 0], data[n1 == 5, 1], s = 100, c= 'gray', label = ' CLuster 6')
plt.scatter(data[n1 == 6, 0], data[n1 == 6, 1], s = 100, c= 'black', label = ' CLuster 7')
#plt.scatter(df16[y_kmeans == 7, 0], df16[y_kmeans == 7, 1], s = 100, c= 'pink', label = ' CLuster 8')
#plt.scatter(df15_norm[y_kmeans == 0, 0], df15_norm[y_kmeans == 8, 1], s = 100, c= 'red', label = ' CLuster 1')
plt.title('with n1 kmean real')
plt.show()'''
print('Naive -K modified takes less iterations to find the centroid')
plt.scatter(data[n3 == 0, 0], data[n3 == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(data[n3 == 1, 0], data[n3 == 1, 1], s = 100, c= 'green', label = 'Cluster 2')
plt.scatter(data[n3 == 2, 0], data[n3 == 2, 1], s = 100, c= 'blue', label = ' CLuster 3')
plt.scatter(data[n3 == 3, 0], data[n3 == 3, 1], s = 100, c= 'purple', label = ' CLuster 4')
plt.scatter(data[n3 == 4, 0], data[n3 == 4, 1], s = 100, c= 'orange', label = ' CLuster 5')#
plt.scatter(data[n3 == 5, 0], data[n3 == 5, 1], s = 100, c= 'gray', label = ' CLuster 6')
plt.scatter(data[n3 == 6, 0], data[n3 == 6, 1], s = 100, c= 'black', label = ' CLuster 7')
#plt.scatter(df16[y_kmeans == 7, 0], df16[y_kmeans == 7, 1], s = 100, c= 'pink', label = ' CLuster 8')
#plt.scatter(df15_norm[y_kmeans == 0, 0], df15_norm[y_kmeans == 8, 1], s = 100, c= 'red', label = ' CLuster 1')
plt.title('with - NAIVE-KMEAN ALGORITHM MODIFIED')
plt.show()


###################################################################
#####PCA ##################

X_std1 = StandardScaler().fit_transform(df15_norm)            #for pre-processing of data
#print("\n \n")
#print("NumPy covariance matrix: \n%s" %np.cov(X_std1.T))
cov1 = np.cov(X_std1.T)                               #get covariance of data

eigan_values1, eigan_vectors1 = np.linalg.eig(cov1)

#print('Eigenvectors For Dataset One\n%s' %eigan_vectors1)
#print('\nEigenvalues For Dataset one\n%s' %eigan_values1)

cor11 = np.corrcoef(X_std1.T)                 #get correaltion between data

eigan_values1, eigan_vectors1 = np.linalg.eig(cor11)

#print('Eigenvectors For Dataset One\n%s' %eigan_vectors1)
#print('\nEigenvalues For Dataset one\n%s' %eigan_values1)

u,s,v = np.linalg.svd(X_std1.T)          #doing SVD on dataset for singular value decomposition
#print("\n Singular directional data for dataset one")
#print(u)
for ev in eigan_vectors1:                             #to determine data is single directional
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#print('\nEverything ok!')
                                                # Make a list of (eigenvalue, eigenvector) tuples
eigan_pairs1 = [(np.abs(eigan_values1[i]), eigan_vectors1[:,i]) for i in range(len(eigan_values1))]

                                                        # Sort the (eigenvalue, eigenvector) tuples from high to low
eigan_pairs1.sort()
eigan_pairs1.reverse()


                                                        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
#print('\nEigenvalues in descending order:')
'''for i in eigan_pairs1:
    print(i[0])'''

tot1 = sum(eigan_values1)
var_exp1 = [(i / tot1)*100 for i in sorted(eigan_values1, reverse=True)]
cum_var_exp1 = np.cumsum(var_exp1)
#print("\nvariance by each component \n", var_exp1)       #finding the variance of data
#print(1000 * '_')
#print("\nsum of contribution",cum_var_exp1)                 #finding the total contribution for data
#print("\n\n Taking 8 components for PCA as it gave nearly 75% data\n")
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(38), var_exp1, alpha=0.5, align='center',
            label='individual explained variance')



    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title("Principal components vs Variance ratio for dataset one")
    plt.tight_layout()
    plt.show()
# Fit the PCA and transform the data
pca = PCA(n_components=4)
reduced1 = pca.fit_transform(X_std1)      #fitting the standard data
#print("\nSIZE of DATASET ONE after PCA feature extraction")
#print(reduced1.shape)
#print("\nReduced dataset 1 after PCA")
#print(reduced1)
plt.figure(figsize=(12,8))
plt.title('PCA Components for DATASET ')
j= 1
plt.scatter(reduced1[:,0],reduced1[:,1])

plt.scatter(reduced1[:,1],reduced1[:,2])

#plt.scatter(reduced1[:,2],reduced1[:,0])

plt.scatter(reduced1[:,2],reduced1[:,3])
plt.scatter(reduced1[:,3],reduced1[:,0])

plt.show()

new1 = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters= i, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
    Kmeans.fit_transform(reduced1)
    new1.append(Kmeans.inertia_)
plt.plot(range(1,11),new1)
plt.title('The elbow method')
plt.xlabel('Number of clusters after PCA')
plt.ylabel('New1')
plt.show()
#print(new1)


outF = open("KmeansOutputafterPCA.txt", "w")


Kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
y_kmeans = Kmeans.fit_predict(reduced1)
i = 1
for i in range(1, 528):
    outF.write(str(i))
    outF.write("\t")
    outF.write(str(y_kmeans[i - 1]))
    outF.write("\n")
    i = + 1
outF.close()
#x_kemans = Kmeans.fit_predict(df16)
plt.scatter(reduced1[y_kmeans == 0, 0], reduced1[y_kmeans == 0, 1], s = 100, c= 'red', label = 'Cluster 1')
plt.scatter(reduced1[y_kmeans == 1, 0], reduced1[y_kmeans == 1, 1], s = 100, c= 'green', label = 'Cluster 2')
plt.scatter(reduced1[y_kmeans == 2, 0], reduced1[y_kmeans == 2, 1], s = 100, c= 'blue', label = ' CLuster 3')
plt.scatter(reduced1[y_kmeans == 3, 0], reduced1[y_kmeans == 3, 1], s = 100, c= 'purple', label = ' CLuster 4')
plt.scatter(reduced1[y_kmeans == 4, 0], reduced1[y_kmeans == 4, 1], s = 100, c= 'orange', label = ' CLuster 5')
plt.scatter(reduced1[y_kmeans == 5, 0], reduced1[y_kmeans == 5, 1], s = 100, c= 'gray', label = ' CLuster 6')
plt.scatter(reduced1[y_kmeans == 6, 0], reduced1[y_kmeans == 6, 1], s = 100, c= 'black', label = ' CLuster 7')
#plt.scatter(df16[y_kmeans == 7, 0], df16[y_kmeans == 7, 1], s = 100, c= 'pink', label = ' CLuster 8')
#plt.scatter(df15_norm[y_kmeans == 0, 0], df15_norm[y_kmeans == 8, 1], s = 100, c= 'red', label = ' CLuster 1')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.title('clusters after PCA')
plt.show()









