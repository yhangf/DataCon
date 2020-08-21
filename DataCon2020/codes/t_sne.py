from sklearn.manifold import TSNE

def plot_t_sne(train_tfidf_features):
    X_tsne = TSNE(n_components=2, perplexity=300, random_state=42).fit_transform(train_tfidf_features)
    font = {"size": 13, 
            "family" : "serif"}
    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, 
                    cmap=plt.cm.get_cmap('rainbow', 2))
        ax.set_title("Features Visualization", fontdict=font)
        ax.set_ylim([-80, 81])
        ax.set_xlim([-82, 81])