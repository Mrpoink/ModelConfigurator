print("Importing umap....")
import umap
import numpy as np
import plotly.graph_objects as go
print("Importing GUIs....")
from sklearn.cluster import KMeans
from IPython.display import display

class Map:
    
    def __init__(self, 
                 num_clusters = 4,
                 n_neighbors = 5,
                 min_dist=0.3,
                 metric='cosine',
                 random_state=42):
        
        self.random_state = random_state
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=self.random_state)
        self.num_clusters = num_clusters
        
    def setup(self, weights):
        
        r"""
        params: Models' Attention Weights For Initial Setup
        returns: Nothing
        """
        
        
        self.embedding = self.reducer.fit_transform(weights)
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state).fit(self.embedding)
        
        self.labels = kmeans.labels_
       
    def plot(self):
        
        
        fig = go.Figure()

        for i in range(self.num_clusters):
            mask = self.labels == i
            fig.add_scatter(
                x=self.embedding[mask, 0], 
                y=self.embedding[mask, 1],
                mode='markers',
                marker=dict(size=10, opacity=0.7),
                name=f"Section {i+1}"
            )

        fig.update_layout(title="BERT Attention Latent Space (UMAP)", template="plotly_dark")
        fig.show()
