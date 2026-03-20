print("Importing umap....")
import umap
import numpy as np
import plotly.graph_objects as go
print("Importing GUIs....")
from sklearn.cluster import KMeans
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class Map:
    def __init__(self, num_clusters=6, n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42):
        self.labels = None
        self.pca = None
        self.random_state = random_state
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=self.random_state)
        self.num_clusters = num_clusters
        self.base_embedding = None  # The original unsteered state
        self.prev_embedding = None  # The state from the prompt exactly before this one
        self.embedding = None

    def setup(self, weights):
        # Standardize and add tiny noise to prevent NaN during PCA
        weights = np.nan_to_num(weights) + np.random.normal(0, 1e-7, weights.shape)
        weights = normalize(weights, axis=1, norm='l2')
        weights = weights + np.random.normal(0, 1e-7, weights.shape)
        
        n_samples, n_features = weights.shape
        n_comp = min(n_samples, n_features, 50)
        
        if self.base_embedding is None:
            # 1. INITIAL BASELINE: Define the world
            # Initialize and FIT the PCA model once
            self.pca = PCA(n_components=n_comp)
            pca_reduced = self.pca.fit_transform(weights)
            
            self.embedding = self.reducer.fit_transform(pca_reduced)
            self.base_embedding = self.embedding
            self.prev_embedding = self.embedding
            
            # Cluster ONCE on the baseline to keep head-colors consistent
            kmeans = KMeans(n_clusters=self.num_clusters, n_init="auto", random_state=self.random_state)
            self.labels = kmeans.fit_predict(self.embedding)
        else:
            # 2. INTERVENTION: Move the heads within the existing world
            self.prev_embedding = self.embedding 
            
            # USE TRANSFORM ONLY for both PCA and UMAP
            pca_reduced = self.pca.transform(weights)
            self.embedding = self.reducer.transform(pca_reduced)
            
    def plot(self):
        
        
        fig = go.Figure()
        for i in range(self.num_clusters):
            mask = self.labels == i
            head_labels = [f"L{l} H{h}" for l in range(12) for h in range(9)]

            # Inside your plot loop:
            fig.add_scatter(
                x=self.embedding[mask, 0], 
                y=self.embedding[mask, 1],
                mode='markers',
                text=[head_labels[j] for j, m in enumerate(mask) if m], # Label each dot
                hoverinfo='text',
                name=f"Cluster {i}"
            )

        fig.update_layout(title="SmolLM-135M Attention Latent Space (UMAP)", template="plotly_dark")
        fig.show()
