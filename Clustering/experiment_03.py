import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from time import time
from scipy.spatial import distance
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Experiment 03')
parser.add_argument('-k', dest='n_clusters', type=int, required=True)

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

	labels = km.labels_

	# Metrics of quality
	print("Silhouette Coefficient: %0.3f" % metric.silhouette_score(data, labels, metric='euclidean'))
	print("Davies-Bouldin Index: %0.3f" % metric.davies_bouldin_score(data, labels))

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