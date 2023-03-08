import cv2
import math
import numpy as np
import pandas as pd
from features import convert_kp_to_array
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import Visualization


############################################
# GLOBALS
############################################
MAX_ITERATIONS = 30  # maximum number of iterations for K-Means clustering
MIN_TUPLES_PER_CLUSTER = 40  # leaf nodes will stop being generated if the number of keypoints is less than this

############################################
# Leaf Node in the cluster tree
############################################
class CLeaf:
    def __init__(self, processed_data, original_data):
        assert len(processed_data) == len(original_data)
        self.X = original_data  # orginal data. not scaled, not reduced in dimensions
        self.X_prime = processed_data # data scaled and reduced dimensions

    #############################################
    # Brute Force Matching
    #############################################
    def search(self, tuple):
        assert tuple.shape[1] == self.X_prime.shape[1]  # ensure correct number of attributes are being compared
        best_match = self.X[0] #best match
        best_dist = math.dist(tuple[0], self.X_prime[0]) # best distance
        for i in range(len(self.X_prime)):
            d = math.dist(tuple[0], self.X_prime[i])
            if d < best_dist:
                best_match = self.X[i]
                best_dist = d
        return best_match, best_dist




############################################
# A single node in the cluster tree
############################################
class CNode:
    def __init__(self):
        self.cluster = KMeans(n_clusters=2, n_init='auto', max_iter=MAX_ITERATIONS) # cluster object
        self.left = None # left child
        self.right = None # right child


    #############################################
    # Performs clustering and builds child nodes
    #############################################
    def build(self, X, original_data):
        self.cluster.fit(X) #Fit the cluster object at this node
        left = [] #tuples in cluster 0
        right = [] # tuples in cluster 1
        left_original = [] # tuples from the original data in cluster 0
        right_original = []  # tuples from the original data in cluster 1
        for i in range(len(self.cluster.labels_)):  #sort tuples by cluster
            label = self.cluster.labels_[i]
            if label == 0:
                left.append(X[i])
                left_original.append(original_data[i])
            else:
                right.append(X[i])
                right_original.append(original_data[i])

        assert len(right) == len(right_original)
        assert len(left) == len(left_original)


        #create left and right nodes
        if len(left) > MIN_TUPLES_PER_CLUSTER or len(right) > MIN_TUPLES_PER_CLUSTER:
            self.left = CNode()
            self.left.build(np.array(left), np.array(left_original))
            self.right = CNode()
            self.right.build(np.array(right), np.array(right_original))
        else: # create leaf nodes
            self.left = CLeaf(np.array(left), np.array(left_original))
            self.right = CLeaf(np.array(right), np.array(right_original))

        return True


    #############################################
    # DFS
    #############################################
    def search(self, tuple):
        print(tuple)
        assert self.left is not None and self.right is not None  # make sure child nodes exist
        predicted_label = self.cluster.predict(tuple)[0]
        if predicted_label == 0:
            return self.left.search(tuple)
        else:
            return self.right.search(tuple)






######################
# Cluster Tree Object
######################
class ClusterTree:
    def __init__(self):
        self.root = None # root node, e.i. the first CNode object in the tree
        self.X_df = None  # original data values, read in as a dataframe
        self.X = None # numpy array of the original values
        self.X_prime = None #X after scale and PCA
        self.scaler = StandardScaler() #object for scaling
        self.pca = PCA(n_components=4) #object for pca


    #################################
    # grab sift features of an image
    #################################
    def load_image_features(self, path):
        picture = cv2.imread(path)
        sift = cv2.SIFT_create()
        g1, l1 = sift.detectAndCompute(picture, None)
        g1 = convert_kp_to_array(g1)
        self.X = g1  # what data I want to work with is in the variable X
        col_names = ['x', 'y', 'response', 'angle', 'size', 'octave']
        self.X_df = pd.DataFrame(g1, columns=col_names)
        return True


    #################################
    # Scale and PCA initial data
    #################################
    def process_raw_data(self):
        assert self.X is not None  #make sure raw data exists
        self.X_prime = self.scaler.fit_transform(self.X)  # scale
        self.X_prime = self.pca.fit_transform(self.X_prime) # reduce dimensions
        return True


    #################################
    # Build the tree structure
    #################################
    def create_tree(self):
        print('Creating Cluster Tree...')
        assert self.X_prime is not None # ensure that the transformed data has been created
        self.root = CNode() #create the first root node
        self.root.build(self.X_prime, self.X) #builds the root node
        print('Done Building Cluster Tree.')
        return True


    #################################
    # find closest match
    #################################
    def find_match(self, tuple):
        tuple = np.array(tuple).reshape(1, -1) # convert to array
        tuple = self.scaler.transform(tuple) # scale
        tuple = self.pca.transform(tuple) # reduce dimension
        return self.root.search(tuple)

    ##################################################################
    # Find Closest Matches, N keypoints
    ##################################################################
    def find_matches(self, tuples):
        matches = []
        for tuple in tuples:
            matches.append(self.find_match(tuple))
        return np.array(matches)




def test():
    tree = ClusterTree()
    tree.load_image_features('images/dash2.png')
    tree.process_raw_data()
    tree.create_tree()

    matches = tree.find_matches(kps)
    print("match:", matches)

