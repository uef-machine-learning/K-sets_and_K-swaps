# K-sets_and_K-swaps
K-sets and k-swaps are two clustering algorithms for sets data. K-sets is designed based on the principles of classical k-means so that it repeats the assignment and update steps until convergence. K-swaps adopt the idea into random swap algorithm, which is a wrapper around the k-means that avoids local minima. The mean of cluster is defined as histogram of the elements of the data in the cluster. Both algorithms performs better when the number of elements in each histogram is limited, for example to 20.

## Preparing input sets data

Prepare sets data in a pure text file where each row is a set. The elements of each set are treated as string data. For example, the data in the text file as:

	A DC B G HJ
	DC G F HJ
	A B HJ K S

means that there are three sets where the five elements of the first set are ‘A’, ‘DC’, ‘B’, ‘G’, and ‘HJ’. We have provided the function read_file to read this text file and prepare suitable data for k-sets and k-swaps function.

## Input parameters of k-sets and k-swaps functions
  
**data** Input sets data, array of n sets

**K** Number of clusters

**MAX_HIST_LENGTH** Maximum length of each mean histogram

**RS_ITER**  Number of iterations for k-swaps

## Output parameters of k-sets and k-swaps functions

**labels** data labels after clustering; values from 0 to K-1

**khist** cluster representative; mean histogram of clusters

**sdh** sum of distances to representative histograms

## How to run the code

Run main_ksets_kswaps.py

Important lines to pay attention to (and to tweak parameters):

	from data_utilities import data_utility

	from clustering_algorithms import ksets, kswaps

	DATA_PATH = './dataset1.txt'

	K = 16

	MAX_HIST_LENGTH = 20

	RS_ITER = 300

	data = data_utility.read_file(DATA_PATH)

	labels1, khist1, sdh1 = ksets.cluster(data, K, MAX_HIST_LENGTH)

	labels2, khist2, sdh2 = kswaps.cluster(data, K, MAX_HIST_LENGTH, RS_ITER)
	
## References

[1] M. Rezaei, P. Fränti. “K-sets and k-swaps algorithms for clustering sets”, submitted to Pattern Recognition Journal.
