import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from scipy.spatial import distance
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Experiment 02')
parser.add_argument('-k', dest='n_clusters', type=int, required=True)

N_NEIGHBORS = 10

def init_dataset():
	# Read csv
    df = pd.read_csv('bags.csv')

    return df

def clustering(data, n_clusters):

	# Create model
	km = KMeans(n_clusters=n_clusters, n_jobs=8)

	print("Clustering data with %s clusters..." % n_clusters)

	t0 = time()
	km.fit(data)
	tf = time()
	print("Done in %0.3fs" % (tf - t0))

	print("Inertia: %0.4f" % km.inertia_)

	return km

def analysis(data, km):
	# Load tweets
	df_tweets = pd.read_csv('health.txt', delimiter='|')

	# Number of clusters
	n_clusters = km.cluster_centers_.shape[0]

	for k in range(0, 3):
		# Select cluster for analysis
		idx = random.randint(0, n_clusters-1)

		print("Cluster: %d" % idx)

		# Filter points by label
		points = np.where(km.labels_ == idx)[0]

		# Find medoid
		min_idx = 0
		min_dist = distance.euclidean(km.cluster_centers_[idx,:], data.iloc[points[0],:])
		for i in range(1, len(points)):
			dist = distance.euclidean(km.cluster_centers_[idx,:], data.iloc[points[i],:])
			if dist < min_dist:
				min_idx = points[i]
				min_dist = dist

		medoid_idx = min_idx
		print("\tMedoid: " + df_tweets.iloc[medoid_idx, 2])

		# Find closest neighbors
		dist = []
		for i in range(0, len(points)):
			if medoid_idx != points[i]:
				dist.append(distance.euclidean(data.iloc[medoid_idx,:], data.iloc[points[i],:]))

		idxs = np.argsort(dist)
		for i in range(0, N_NEIGHBORS):
			print("\tNeighbor " + str(i) + ": " + df_tweets.iloc[points[idxs[i]], 2])

	# Compute histogram of clusters
	hist = []
	for i in range(0, n_clusters):
		# Filter points by label
		points = np.where(km.labels_ == i)[0]
		hist.append(len(points))

	# Plot histogram
	fig = plt.figure()
	width = 1/1.5
	plt.bar(range(n_clusters), hist, width, color="blue")
	plt.xlabel("Group")
	plt.ylabel("Frequency")
	plt.show()
	fig.savefig("histogram.png")

def main():
	args = parser.parse_args()

	# Load dataset
	data = init_dataset()

	# Clustering data
	km = clustering(data, args.n_clusters)

	# Clustering analysis
	analysis(data, km)


if __name__ == '__main__':
    main()