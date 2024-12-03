import pandas as pd
import networkx as nx
import os
from sklearn.cluster import KMeans
import numpy as np

class GranularKnowledgeGraph:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def generate_aggregate_graph(self, dataset, target_column):
        try:
            df = pd.read_csv(os.path.join(self.dataset_folder, dataset))
            
            # Create aggregate graph
            G = nx.Graph()
            nodes = []
            edges = []
            for column in df.columns:
                if column != target_column:
                    correlation = df[column].corr(df[target_column])
                    importance = abs(correlation)
                    nodes.append({"data": {"id": column, "type": 'feature', "importance": importance}})
                    edges.append({"data": {"source": column, "target": target_column, "weight": importance}})
            
            nodes.append({"data": {"id": target_column, "type": 'target', "importance": 1.0}})
            
            # Create cluster graphs
            X = df.drop(columns=[target_column])
            kmeans = KMeans(n_clusters=5)  # You can adjust the number of clusters
            clusters = kmeans.fit_predict(X)
            
            cluster_graphs = {}
            for i in range(kmeans.n_clusters):
                cluster_df = df[clusters == i]
                cluster_nodes = []
                cluster_edges = []
                for column in cluster_df.columns:
                    if column != target_column:
                        correlation = cluster_df[column].corr(cluster_df[target_column])
                        importance = abs(correlation)
                        cluster_nodes.append({"data": {"id": column, "type": 'feature', "importance": importance}})
                        cluster_edges.append({"data": {"source": column, "target": target_column, "weight": importance}})
                cluster_nodes.append({"data": {"id": target_column, "type": 'target', "importance": 1.0}})
                cluster_graphs[str(i)] = {"nodes": cluster_nodes, "edges": cluster_edges}
            
            return {
                'aggregateData': {"nodes": nodes, "edges": edges},
                'clusterData': cluster_graphs
            }
        except Exception as e:
            print(f"Error generating aggregate granular graph: {str(e)}")
            return None

    def get_row_indices(self, dataset):
        try:
            df = pd.read_csv(os.path.join(self.dataset_folder, dataset))
            return list(range(len(df)))
        except Exception as e:
            print(f"Error fetching row indices: {str(e)}")
            return []