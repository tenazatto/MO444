import argparse
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Experiment 01')
parser.add_argument('-csv', dest='csv', type=str, required=True)
parser.add_argument('-maxk', dest='max_n_clusters', type=int, required=True)
parser.add_argument('-step', dest='step', type=int, required=True)

def init_dataset(file):
	# Read csv
    df = pd.read_csv(file)

    return df

def clustering(data, max_n_clusters, step):

	cost = []
	total_time = 0.0

	clusters = list(range(0, max_n_clusters+1, step))
	clusters[0] = 2

	# For each number of cluster
	for i in range(0, len(clusters)):
		# Create model
		km = KMeans(n_clusters=clusters[i], n_jobs=8)

		print("Clustering data with %s clusters..." % clusters[i])

		t0 = time()
		km.fit(data)
		tf = time()
		total_time += (tf - t0)
		print("Done in %0.3fs" % (tf - t0))

		cost.append(km.inertia_)
		print("Inertia: %0.4f" % km.inertia_)

	# Plot cost
	fig = plt.figure()
	plt.plot(clusters, cost)
	plt.xlabel("Number of clusters")
	plt.ylabel("Cost")
	plt.show()
	fig.savefig("n_clusters.png")

	print("Elapsed total time: %0.3fs" % total_time)


def main():
	args = parser.parse_args()

	# Load dataset
	data = init_dataset(args.csv)

	# Clustering data
	clustering(data, args.max_n_clusters, args.step)


if __name__ == '__main__':
    main()