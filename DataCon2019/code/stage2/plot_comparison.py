import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X_tsne =  pd.read_pickle("api_name_exinfos_call_name_tsne_data.pkl")
dbscan_y_pred = pd.read_csv("result.csv")["family_id"]
kmeans_50 = pd.read_csv("k-means_cluster=50_result.csv")["family_id"]
kmeans_100 = pd.read_csv("k-means_cluster=100_result.csv")["family_id"]
kmeans_200 = pd.read_csv("k-means_cluster=200_result.csv")["family_id"]
kmeans_250 = pd.read_csv("k-means_cluster=250_result.csv")["family_id"]
kmeans_300 = pd.read_csv("k-means_cluster=300_result.csv")["family_id"]
kmeans_400 = pd.read_csv("k-means_cluster=400_result.csv")["family_id"]
kmeans_500 = pd.read_csv("k-means_cluster=500_result.csv")["family_id"]

font = {"color": "darkred",
        "size": 25, 
        "family" : "serif"}

plt.style.use("dark_background")
plt.figure(figsize=(30, 25))

plt.subplot(3, 3, 1) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_50.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 50))
plt.title("K-means_cluster=50_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 50) 

plt.subplot(3, 3, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dbscan_y_pred.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', dbscan_y_pred.max()-dbscan_y_pred.min()))
plt.title("DBSCAN_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(dbscan_y_pred.min(), 1000)

plt.subplot(3, 3, 3) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_100.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 100))
plt.title("K-means_cluster=100_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 100) 

plt.subplot(3, 3, 4) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_200.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 200))
plt.title("K-means_cluster=200_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 200) 

plt.subplot(3, 3, 5) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c=np.ones(60000), cmap=plt.cm.get_cmap('rainbow', 1))
plt.title("origin_data_t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=[0]) 
cbar.set_label(label='color bar', fontdict=font)

plt.subplot(3, 3, 6) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_250.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 250))
plt.title("K-means_cluster=250_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 250) 

plt.subplot(3, 3, 7) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_300.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 300))
plt.title("K-means_cluster=300_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 300) 

plt.subplot(3, 3, 8) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_400.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 400))
plt.title("K-means_cluster=400_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 400) 

plt.subplot(3, 3, 9) 
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_500.values, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 500))
plt.title("K-means_cluster=500_t-SNE", fontdict=font)
cbar = plt.colorbar() 
cbar.set_label(label='family id', fontdict=font)
plt.clim(0, 500) 

plt.tight_layout()
plt.savefig("K-means_and_DBSCAN_cluster_comparison.jpg")