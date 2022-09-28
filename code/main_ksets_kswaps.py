# Usage of k-sets and k-swaps algorithms
# Paper: "K-sets and k-swaps algorithms for clustering sets", Mohammad Rezaei and Pasi Franti, 2022

from data_utilities import data_utility
from clustering_eval import clustering_evaluation
from clustering_algorithms import ksets, kswaps
import random
import numpy as np

DISTANCE_METRIC = 'Jaccard' # Jaccard, Cosine
METHOD = 'kswaps'

DATA_PATH = '../datasets/artificial/data_1200_200_16_5_1.txt'
GT_LABELS_PATH = '../datasets/artificial/gt_labels_1200_200_16_5_1.txt' # ground truth labels
K = 16 # Number of clusters
MAX_HIST_LENGTH = 20 # if zero no limitation
RS_ITER = 300 # number of iterations for kswaps
rand_seed = 10

labels_gt = np.loadtxt(GT_LABELS_PATH, dtype="float", delimiter='\n').astype(int)

# Read original data and ground truth labels
data = data_utility.read_file(DATA_PATH)
labels_gt = np.loadtxt(GT_LABELS_PATH, dtype="float", delimiter='\n').astype(int)

np.random.seed(rand_seed)
random.seed(rand_seed)

if METHOD == 'ksets':
    labels, khist, sdh = ksets.cluster(data, K, MAX_HIST_LENGTH)
if METHOD == 'kswaps':
    labels, khist, sdh = kswaps.cluster(data, K, MAX_HIST_LENGTH, RS_ITER)

ari = clustering_evaluation.ARI(labels, labels_gt)
print('ARI = ', ari, '\n')
