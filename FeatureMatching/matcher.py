########################
# NOTES
########################
#remove local barrel effect
#remove lens distortion

from dataclasses import dataclass
import math
import cv2
from features import convert_kp_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import Visualization
import ClusterTree


########################
# MATCH DATASTRUCT
########################
@dataclass
class Match:
    """ Holds a match object between two vectors """
    def __init__(self, v1, v2, i1, i2, distance, angle=None):
        assert len(v1) == len(v2), "ERROR: Tried to make a match struct with vectors of different length"
        self.v1 = v1
        self.v2 = v2
        self.i1 = i1
        self.i2 = i2
        self.distance = distance
        self._angle = angle


    def angle(self):
        """
        Angle between the matches in terms of Degrees north of east.
        Will calculate if has not been calculated yet
        """
        if not(self._angle is None):
            return self._angle
        #calculate angle between
        dx, dy = self.v2[0] - self.v1[0], self.v2[1] - self.v1[1]
        a = ((dx**2) + (dy**2))**0.5 # length of the difference vector
        b = 1
        c = (((dx-1)**2) + (dy**2))**0.5 #distance between <1,0> and difference vector
        radians = math.acos((a**2 + b**2 - c**2 )/ (2 * a * b))
        gamma = radians * ( 180.0 / math.pi )
        if dy < 0:
            gamma = 360 - gamma

        self._angle = gamma
        return gamma



######################
# MATCHER OBJECT
######################
class Matcher:

    def __init__(self, bucket_size=40):
        self.X_df = None  # original data values, read in as a dataframe
        self.X = None # numpy array of the original values
        self.X_prime = None #X after scale and PCA
        self.scaler = StandardScaler() #object for scaling
        self.pca = PCA(n_components=4) #object for pca
        self.cluster = None # cluster object
        self.cluster_map = dict()  # dictionary storing the adjusted clusters by their label
        self.bucket_size = bucket_size # how many keypoints per cluster



    #################################
    # grab sift features of an image
    #################################
    def load_image_features(self, path):
        print('Reading image...')
        picture = cv2.imread(path)
        print('Extracting SIFT features...')
        sift = cv2.SIFT_create()
        g1, l1 = sift.detectAndCompute(picture, None)
        print('Converting SIFT features to array...')
        g1 = convert_kp_to_array(g1)
        self.X = g1  # what data I want to work with is in the variable X

        #convert to dataframe
        print('Constructing dataframe...')
        col_names = ['x', 'y', 'response', 'angle', 'size', 'octave']
        self.X_df = pd.DataFrame(g1, columns=col_names)

        print('Done!')
        return True

    ################################################
    # Fit the scaler, PCA, and Cluster objects
    # Transform the data.
    ################################################
    def fit(self):
        #make sure the data is all there
        if self.X is None:
            return False

        # scale and pca
        print('Scaling the data...')
        self.X_prime = self.scaler.fit_transform(self.X) #scale the data
        print('Performing PCA...')
        self.X_prime = self.pca.fit_transform(self.X_prime) # move into principle components

        #add principle components and labels to dataframe
        print('Adding principle components to dataframe...')
        column_names = [f'pc{i+1}' for i in range(self.pca.n_components)]
        pc_df = pd.DataFrame(self.X_prime, columns=column_names)
        print('Modifying Dataframe...')
        self.X_df = pd.concat([self.X_df, pc_df], axis=1)


        # clustering
        print('Clustering...')
        n_clusters = len(self.X) // self.bucket_size  # calculate number of clusters
        self.cluster = BisectingKMeans(n_clusters=n_clusters) #create cluster object
        self.cluster.fit(self.X_prime)
        print('Adding Labels to Dataframe...')
        self.X_df['label'] = self.cluster.labels_
        print('Done!')
        return True

    ################################################
    # Store keypoints by cluster
    ################################################
    def create_cluster_map(self):
        print('Creating Cluster Buckets...')
        #iterate over the keypoints and assign to cluster bucket
        for i in range(self.cluster.n_clusters):
            self.cluster_map[i] = self.X_df[self.X_df.label == i]

        print('keypoints with label 12:')
        print(self.cluster_map[12])


    ################################################
    # Find match for given keypoint vector
    ################################################
    def find_match(self, keypoint):
        #scale and convert to pca space
        keypoint = self.scaler.transform(keypoint)
        keypoint = self.pca.transform(keypoint)
        keypoint = np.array(keypoint)
        print(keypoint)

        print(self.cluster.cluster_centers_)
        #find label/bucket/cluster
        label = self.cluster.predict(keypoint)

        #









########################
# Euclidean Distance
########################
def euclid_dist(v1, v2):
    """
    :param v1: vector of data
    :param v2: second vector of data, same length
    :return: euclidean distance between the two vectors
    """
    assert len(v1) == len(v2), "ERROR: can't find euclidean distance between vectors of different length"
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i])**2
    return sum**0.5




########################
# Find A Single Match
########################
def match(v, A):
    """
    O(n)
    :param v: A single vector of data. Represents a keypoint in this case
    :param A: An array of vectors. Each vector is of length == len(v)
    :return: index of the best match to v in A.
    """
    best_dist = euclid_dist(v, A[0])
    index = 0
    for i in range(len(A)):
        dist = euclid_dist(A[i], v)
        if dist < best_dist:
            best_dist = dist
            index = i
    return index, best_dist



########################
# Cross Check
########################
def cross_check(v, A, dist, tolerance=0.02):
    """
    O(n)
    :param v: vector to verify
    :param A: An array of vectors
    :param dist: The best distance to be tested for
    :return: True if no vector in A has a distance smaller than dist. Comparison is done within a tolerance.
    """
    for i in range(len(A)):
        if euclid_dist(v, A[i]) < (dist -(dist * tolerance)):
            return False
    return True




########################
# FIND ALL MATCHES
########################
def all_matches(A, B):
    """
    :param A: array of vectors
    :param B: another array of vectors
    :return: a list of matches between A and B
    """
    matches = []
    for i in range(len(A)):
        match_index, distance = match(A[i], B)
        if match(B[match_index], A)[0] == i:  #cross check
            matches.append(Match(A[i], B[match_index], i, match_index, distance))
    return matches




def test_matches():
    # All the images
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')

    # keypoints
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(screenshot, None)
    g2, l2 = sift.detectAndCompute(picture, None)
    g3, l3 = sift.detectAndCompute(obstructed, None)

    #transform to array
    g1 = convert_kp_to_array(g1)
    g2 = convert_kp_to_array(g2)
    g3 = convert_kp_to_array(g3)

    #grab only the x and y values
    g1 = g1[:, :2]  #arr[row_start:row_end, col_start:col_end]
    g2 = g2[:, :2]
    g3 = g3[:, :2]

    #plot the first one
    x = g3[:, 0].reshape(-1)
    y = g3[:, 1].reshape(-1)
    #print(len(x))
    #print(len(y))
    plt.scatter(x, y, s=1)
    plt.show()


    """#plot the second one
    x = g2[:, 0].reshape(-1)
    y = g2[:, 1].reshape(-1)
    print(len(x))
    print(len(y))
    plt.scatter(x, y, s=1)
    plt.show()"""

    clustering = AgglomerativeClustering(n_clusters=len(g1)//10).fit(g1)
    print(clustering.labels_)
    plt.hist(clustering.labels_, bins=10)
    plt.show()







    #x = [1, 2, 3, 4, 5, 6, 7, 8]
    #y = [2, 3, 1, 3, 1, 4, 2, 3]
    #plt.scatter(x, y)
    #plt.show()



###################################################
#Helper Function For Plotting The Variances FOR PCA
###################################################
def plot_pca_variance(data, n_components=None):
    # calculate principle components
    pc = PCA(n_components=n_components)
    pc.fit(data)  # try with SCALED data instead

    # plot explained variance
    plt.bar(range(1, pc.n_components_ + 1), pc.explained_variance_ratio_, align='center', label='Explained Variance')

    # also plot cumulative variance
    cumulative_variance = []
    total = 0
    for i in range(pc.n_components_):
        total += pc.explained_variance_ratio_[i]
        cumulative_variance.append(total)
    plt.step(range(1, pc.n_components_ + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance',
             color='red')

    # clean up and display the plot
    plt.xticks(range(1, pc.n_components_ + 1), range(1, pc.n_components_ + 1))
    for i in range(pc.n_components_):
        text_label = str(round(100 * pc.explained_variance_ratio_[i], 2)) + '%'
        plt.text(i + 1, pc.explained_variance_ratio_[i], text_label, ha='center')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('\'K\'-th Principle Component')
    plt.legend(loc='center right')
    plt.show()

##########################
# PCA ON SIFT DIMENSIONS
##########################
def reduce_sift_dimensions():
    #load data
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(screenshot, None)
    g2, l2 = sift.detectAndCompute(picture, None)
    g1 = convert_kp_to_array(g1)
    g2 = convert_kp_to_array(g2)
    X = g1  # here is where I chose what data I want to work with

    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #pca variance plot
    plot_pca_variance(X, n_components=6)

################################################################
# Helper Function for graphing cluster validity given a dataset
################################################################
def cluster_validity(X, title, clusters_range=range(2, 8)):
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    # davies is "ratio between the cluster scatter and the cluster's separation, lower is better".
    # silhouette, higher is better it seems, range is -1 to +1


    #Perform PCA to extract first 2 principle components
    #X = PCA(n_components=2).fit_transform(X)
    # cluster and append scores
    davies_values = []
    silhouette_values = []
    for i in clusters_range:
        km = KMeans(n_clusters=i)
        y_pred = km.fit_predict(X)
        d_score = davies_bouldin_score(X, y_pred)
        s_score = silhouette_score(X, y_pred)
        davies_values.append(d_score)
        silhouette_values.append(s_score)

    #visualize, make the graphs
    print('cluster range:', clusters_range)
    print('davies:', davies_values)
    print('sil:', silhouette_values)
    plt.plot(clusters_range, davies_values, color='blue', label='davies score')
    plt.plot(clusters_range, silhouette_values, color='red', label='silhouette score')
    plt.xlabel('Value of K')
    plt.ylabel('Validity Score')

    plt.legend()
    plt.title(title)
    plt.show()




#############################
# CLUSTERING GLOBAL FEATURES
#############################
def cluster_global_keypoints():
    """
    This function will read in an image of a picture of the dashboard unobstructed.
    The keypoints will then  be clustered based on their global features
    """
    #grab data
    picture = cv2.imread('images/dash2.png')
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(picture, None)
    g1 = convert_kp_to_array(g1)
    X = g1 # what data I want to work with is in the variable X

    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # cluster the data
    cluster = AgglomerativeClustering(n_clusters=10).fit(X)

    #prepare seperate colors
    color_range = 'bgrcmykw'
    colors = [color_range[label%len(color_range)] for label in cluster.labels_]
    print(colors)

    #verify the length of everything is the same
    assert len(colors) == len(cluster.labels_)
    assert len(X) == len(colors)
    assert len(cluster.labels_) == len(colors)

    #make the plot
    x = X[:, 0].reshape(-1) # flatten x values
    y = X[:, 1].reshape(-1) # flatten y values
    plt.scatter(list(x), list(y), c=colors, s=1)
    plt.show()




######################################################
# CLUSTER GLOBAL ONLY X AND Y
######################################################
def cluster_xy():
    # grab data
    picture = cv2.imread('images/dash2.png')
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(picture, None)
    g1 = convert_kp_to_array(g1)
    X = g1  # what data I want to work with is in the variable X

    #keep only x and y values from the data
    X = X[:, 0:2]
    print(X)

    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # cluster the data
    #cluster = AgglomerativeClustering(n_clusters=5).fit(X)
    cluster = KMeans(n_clusters=10, random_state=0).fit(X)

    # prepare seperate colors
    color_range = 'bgrcmykw'
    colors = [color_range[label % len(color_range)] for label in cluster.labels_]
    print(colors)

    # verify the length of everything is the same
    assert len(colors) == len(cluster.labels_)
    assert len(X) == len(colors)
    assert len(cluster.labels_) == len(colors)

    # make the plot
    x = X[:, 0].reshape(-1)  # flatten x values
    y = X[:, 1].reshape(-1)  # flatten y values
    plt.scatter(list(x), list(y), c=colors, s=1)
    plt.show()

    #perform cluster validity
    cluster_validity(X, 'KMeans cluster validity', clusters_range=range(2, 10))




##########################################################
# CLUSTERING TEST 4
# scale the data
# move global into 4 principle components
# Agglomeratively Cluster global features to 100 clusters
# visualize the data
##########################################################
def cluster_test4():
    # grab data
    picture = cv2.imread('images/dash2.png')
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(picture, None)
    g1 = convert_kp_to_array(g1) # labels = ['x', 'y', 'response', 'angle', 'size', 'octave']
    X = g1  # what data I want to work with is in the variable X
    x_prime, y_prime = X[:, 0].reshape(-1), X[:, 1].reshape(-1)

    #scale data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #move into 4 principle components
    X = PCA(n_components=4).fit_transform(X)

    #Cluster to 100 clusters
    cluster = AgglomerativeClustering(n_clusters=100).fit(X)

    # prepare seperate colors
    colors = [label / 100 for label in cluster.labels_]

    # make the plot
    plt.scatter(list(x_prime), list(y_prime), c=colors, s=1)
    plt.show()
    data = {
        'x': x_prime,
        'y': y_prime,
        'label': cluster.labels_
    }
    df = pd.DataFrame(data)
    print(df)



############################################################
# THIS FUNCTION CASUSES THE CRASH
############################################################
def test_matcher_object():
    matcher = Matcher()
    matcher.load_image_features('images/dash2.png')
    matcher.fit()
    matcher.create_cluster_map()
    matcher.find_match([[500,500,0,15,0,10000]])





############################################################
# - build match tree on image
# - find kps in other img
# - find matches for all kps in other img
# - Visualize all matches
############################################################
def draw_all_matches():
    #build match tree on image
    tree = ClusterTree.ClusterTree()
    tree.load_image_features('images/dash2.png')
    tree.process_raw_data()
    tree.create_tree()

    #find kps in other image
    other = cv2.imread('images/dash3.png')
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(other, None)
    g1 = convert_kp_to_array(g1) # labels = ['x', 'y', 'response', 'angle', 'size', 'octave']
    X = g1  # what data I want to work with is in the variable X


    #find matches
    matches = []
    for i in range(len(X)):
        #print(X[i], "       ", end='')
        match = tree.find_match(X[i])
        #print(match)
        matches.append(match)

    #visualize matches
    #need to only grab the x and y for each one
    A = [(X[i][0], X[i][1]) for i in range(len(X))]
    B = [(features[0], features[1]) for features, _ in matches]


    Visualization.draw_matches(A, B, cv2.imread('images/dash2.png'), cv2.imread('images/dash3.png'))






if __name__ == '__main__':
    draw_all_matches()











