import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy

img = cv2.imread("data/traffic_signal.png")

#Flattening the image
flattened_img = img.reshape((-1,3))

#Function that calculates Euclidean distance
def euclidean(p, q):
    n_dim = len(p)
    return math.sqrt(sum([(p[i] - q[i]) ** 2 for i in range(n_dim)]))

#Function that ouputs the centroids after convergence and a list of clusters of pixels.
def k_means(flattened_img, k):
    centroids = []
    cluster = [[],[],[],[]]
    cluster_pixels = [[],[],[],[]]
    for i in range(k):
        centroids.append(flattened_img[random.randrange(len(flattened_img))])
    a = 0   
    
    processed_img = copy.deepcopy(flattened_img)
    
    while True:
        distance = []
        for i in range(len(flattened_img)):
            distance = [euclidean(flattened_img[i], j) for j in centroids]
            cluster[distance.index(min(distance))].append(flattened_img[i])
            cluster_pixels[distance.index(min(distance))].append(i)
        centroids = []    
        
        for i in range(k):
            centroids.append(np.mean(cluster[i], axis = 0).astype(int))
            
        if a == 1:
            if np.array_equal(centroids, previous_centroids):
                break 
        a = 1
        previous_centroids = copy.deepcopy(centroids)
        cluster = [[],[],[],[]]  
        cluster_pixels = [[],[],[],[]]
        
    return centroids, cluster_pixels

a, b = k_means(flattened_img, 4)

#Saving the ouput in processed image
processed_img = np.zeros((len(flattened_img),3),  dtype=np.uint8)
for i in range(len(a)):
    for j in range(len(b)):
        processed_img[b[i][j]] = a[i]

processed_img = processed_img.reshape(img.shape)
cv2.imwrite('results/traffic_signal.png', processed_img)






