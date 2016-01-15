__author__ = 'lenovo'

import matplotlib.pyplot as plt
import numpy as np

def plotPointsOn2DPlane(x, y, color='blue'):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=color)
    ax.set_xlabel('To be added')
    ax.set_ylabel('To be added')
    fig.show()

'''
only has one dimension
'''
def plotPointsAfterPCA(x, color='blue'):
    fig, ax = plt.subplots()
    ax.scatter(x, np.zeros((1, len(x))), c=color)
    ax.set_xlabel('To be added')
    ax.set_ylabel('To be added')
    fig.show()

'''
combination
'''
def plotPCATogether(x, y, t, color1='blue', color2='red'):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('before PCA')
    ax2.set_title('after PCA')
    ax1.scatter(x, y, c=color1)
    ax2.scatter(t, np.zeros((1, len(t))), c=color2)
    fig.show()

'''
put origin data, data after pca, k-means data altogether
'''
def plotAltogether(x, y, t, km, path, color_group=[], x_name='X axis', y_name='Y axis', color1='blue', color2='blue'):
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(1, 1, 1, sharex=False, sharey=False)
    # ax2 = fig.add_subplot(1, 2, 1, sharex=False, sharey=False)
    # ax3 = fig.add_subplot(2, 1, 1, sharex=False, sharey=False)
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False)
    plt.cla()
    plt.xlabel(x_name, fontsize=15)
    plt.ylabel(y_name, fontsize=15)
    ax1.set_title('original data')
    ax2.set_title('data after PCA')
    ax3.set_title('K-Means clustering')
    ax1.scatter(x, y, c=color1)
    ax2.scatter(t, np.zeros((1, len(t))), c=color2)
    k_means_labels = km.labels_
    k_means_cluster_centers = km.cluster_centers_
    n_clusters = len(k_means_cluster_centers)
    for k, col in zip(range(n_clusters), color_group):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax3.plot(x[my_members], y[my_members], 'w', markerfacecolor=col, marker='.')
        ax3.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax3.set_xticks(())
    ax3.set_yticks(())
    plt.savefig(path+x_name+'_with_'+y_name+'.png')
    # plt.show()