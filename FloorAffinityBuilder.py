import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class FloorPlanAffinityBuilder:
    """Build custom affinity matrices for floor plan features"""
    
    def __init__(self, feature_df, feature_names):
        self.feature_df = feature_df
        self.feature_names = feature_names
        self.feature_groups = self._categorize_features()
        
    def _categorize_features(self):
        """Categorize features by type"""
        groups = {
            'spatial': [f for f in self.feature_names if 'avg_x' in f or 'avg_y' in f],
            'size': [f for f in self.feature_names if 'area' in f and 'ratio' not in f],
            'proportion': [f for f in self.feature_names if 'aspect' in f or 'ratio' in f],
            'topology': [f for f in self.feature_names if 'count' in f],
            'bigram': [f for f in self.feature_names if 'bi_' in f],
            'trigram': [f for f in self.feature_names if 'tri_' in f]
        }
        return groups
    
    def compute_rbf_affinity(self, X, gamma='auto'):
        """Standard RBF (Gaussian) kernel affinity"""
        if gamma == 'auto':
            # Use median heuristic
            distances = pairwise_distances(X, metric='euclidean')
            gamma = 1.0 / np.median(distances)**2
        
        affinity = rbf_kernel(X, gamma=gamma)
        return affinity
    
    def compute_weighted_feature_affinity(self, X, feature_weights=None):
        """Compute affinity with weighted feature groups"""
        
        if feature_weights is None:
            # Default weights emphasizing topology and adjacency
            feature_weights = {
                'spatial': 0.15,
                'size': 0.15,
                'proportion': 0.20,
                'topology': 0.25,
                'bigram': 0.15,
                'trigram': 0.10
            }
        
        # Compute affinity for each feature group
        group_affinities = {}
        
        for group_name, features in self.feature_groups.items():
            if features:  # If group has features
                # Get indices of these features
                indices = [self.feature_names.index(f) for f in features if f in self.feature_names]
                if indices:
                    X_group = X[:, indices]
                    # Normalize group features
                    X_group_norm = StandardScaler().fit_transform(X_group)
                    # Compute group affinity
                    group_affinities[group_name] = rbf_kernel(X_group_norm)
        
        # Weighted combination
        final_affinity = np.zeros((X.shape[0], X.shape[0]))
        total_weight = 0
        
        for group_name, affinity in group_affinities.items():
            weight = feature_weights.get(group_name, 0.1)
            final_affinity += weight * affinity
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            final_affinity /= total_weight
            
        return final_affinity
    
    def compute_adjacency_pattern_affinity(self, X):
        """Affinity based on shared adjacency patterns"""
        
        # Extract adjacency features (bigrams and trigrams)
        adjacency_features = self.feature_groups['bigram'] + self.feature_groups['trigram']
        indices = [self.feature_names.index(f) for f in adjacency_features if f in self.feature_names]
        
        if not indices:
            return np.eye(X.shape[0])
        
        X_adj = X[:, indices]
        
        # Binary encoding: presence/absence of adjacency patterns
        X_binary = (X_adj > 0).astype(float)
        
        # Jaccard similarity for binary patterns
        intersection = X_binary @ X_binary.T
        row_sums = X_binary.sum(axis=1)
        union = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - intersection
        
        # Avoid division by zero
        union[union == 0] = 1
        jaccard_sim = intersection / union
        
        return jaccard_sim
    
    def compute_spatial_configuration_affinity(self, X):
        """Affinity based on spatial configuration similarity"""
        
        spatial_features = self.feature_groups['spatial']
        if not spatial_features:
            return np.eye(X.shape[0])
        
        indices = [self.feature_names.index(f) for f in spatial_features if f in self.feature_names]
        X_spatial = X[:, indices]
        
        # Normalize spatial coordinates
        X_spatial_norm = StandardScaler().fit_transform(X_spatial)
        
        # Use cosine similarity for spatial configuration
        # This captures similar layouts regardless of absolute position
        spatial_affinity = cosine_similarity(X_spatial_norm)
        
        # Convert negative similarities to 0
        spatial_affinity = np.maximum(spatial_affinity, 0)
        
        return spatial_affinity
    
    def compute_hybrid_affinity(self, X, method_weights=None):
        """Combine multiple affinity computation methods"""
        
        if method_weights is None:
            method_weights = {
                'rbf': 0.3,
                'weighted_features': 0.3,
                'adjacency_patterns': 0.25,
                'spatial_config': 0.15
            }
        
        affinities = {
            'rbf': self.compute_rbf_affinity(X),
            'weighted_features': self.compute_weighted_feature_affinity(X),
            'adjacency_patterns': self.compute_adjacency_pattern_affinity(X),
            'spatial_config': self.compute_spatial_configuration_affinity(X)
        }
        
        # Weighted combination
        hybrid_affinity = np.zeros((X.shape[0], X.shape[0]))
        
        for method, weight in method_weights.items():
            if method in affinities:
                hybrid_affinity += weight * affinities[method]
        
        return hybrid_affinity


class SpectralClusteringAnalyzer:
    """Analyze floor plans using spectral clustering with custom affinities"""
    
    def __init__(self, X, feature_names):
        self.X = X
        self.feature_names = feature_names
        self.affinity_builder = FloorPlanAffinityBuilder(None, feature_names)
        self.results = {}
        
    def evaluate_affinity_matrix(self, affinity, name=""):
        """Evaluate properties of affinity matrix"""
        
        # Check connectivity
        n_components, labels = connected_components(
            csr_matrix(affinity), directed=False, return_labels=True
        )
        
        # Compute statistics
        stats = {
            'name': name,
            'connected_components': n_components,
            'min_value': affinity.min(),
            'max_value': affinity.max(),
            'mean_value': affinity.mean(),
            'sparsity': (affinity < 0.01).sum() / affinity.size,
            'diagonal_mean': np.diag(affinity).mean()
        }
        
        # Visualize affinity matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(affinity[:500, :500], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Affinity Matrix - {name} (first 500 samples)')
        plt.tight_layout()
        plt.show()
        
        return stats
    
    def compare_clustering_methods(self, n_clusters=10, affinity_methods=None):
        """Compare different affinity methods with K-means"""
        
        if affinity_methods is None:
            affinity_methods = {
                'rbf': lambda X: self.affinity_builder.compute_rbf_affinity(X),
                'weighted': lambda X: self.affinity_builder.compute_weighted_feature_affinity(X),
                'adjacency': lambda X: self.affinity_builder.compute_adjacency_pattern_affinity(X),
                'spatial': lambda X: self.affinity_builder.compute_spatial_configuration_affinity(X),
                'hybrid': lambda X: self.affinity_builder.compute_hybrid_affinity(X)
            }
        
        results = {}
        
        # K-means baseline
        print("Running K-means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(self.X)
        
        results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(self.X, kmeans_labels, sample_size=5000),
            'calinski_harabasz': calinski_harabasz_score(self.X, kmeans_labels),
            'inertia': kmeans.inertia_
        }
        
        # Spectral clustering with different affinities
        for method_name, affinity_func in affinity_methods.items():
            print(f"\nRunning Spectral clustering with {method_name} affinity...")
            
            # Compute affinity
            affinity = affinity_func(self.X)
            
            # Evaluate affinity
            affinity_stats = self.evaluate_affinity_matrix(affinity, method_name)
            
            # Run spectral clustering
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            
            try:
                spectral_labels = spectral.fit_predict(affinity)
                
                results[f'spectral_{method_name}'] = {
                    'labels': spectral_labels,
                    'silhouette': silhouette_score(self.X, spectral_labels, sample_size=5000),
                    'calinski_harabasz': calinski_harabasz_score(self.X, spectral_labels),
                    'affinity_stats': affinity_stats
                }
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results[f'spectral_{method_name}'] = {'error': str(e)}
        
        self.results = results
        return results
    
    def visualize_comparison(self, embedding_2d):
        """Visualize different clustering results"""
        
        n_methods = len(self.results)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (method, result) in enumerate(self.results.items()):
            if 'error' not in result:
                ax = axes[idx]
                scatter = ax.scatter(
                    embedding_2d[:, 0], 
                    embedding_2d[:, 1],
                    c=result['labels'],
                    cmap='tab10',
                    alpha=0.6,
                    s=5
                )
                ax.set_title(f'{method}\nSilhouette: {result["silhouette"]:.3f}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cluster_stability(self, affinity_method='hybrid', n_clusters=10, n_iterations=10):
        """Analyze stability of spectral clustering"""
        
        print(f"Analyzing stability over {n_iterations} iterations...")
        
        # Compute affinity once
        affinity = self.affinity_builder.compute_hybrid_affinity(self.X)
        
        # Run multiple times with different initializations
        all_labels = []
        
        for i in range(n_iterations):
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=i,  # Different seed each time
                n_init=1
            )
            labels = spectral.fit_predict(affinity)
            all_labels.append(labels)
        
        # Compute pairwise agreement
        n_samples = len(labels)
        agreement_matrix = np.zeros((n_samples, n_samples))
        
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if labels[i] == labels[j]:
                        agreement_matrix[i, j] += 1
                        agreement_matrix[j, i] += 1
        
        agreement_matrix /= n_iterations
        
        # Compute stability score
        stability_score = np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)])
        
        print(f"Average pairwise stability: {stability_score:.3f}")
        
        return stability_score, agreement_matrix
    
    def parameter_tuning(self, n_clusters_range=range(5, 20), gamma_range=None):
        """Tune parameters for spectral clustering"""
        
        if gamma_range is None:
            # Compute automatic gamma range
            distances = pairwise_distances(self.X[:1000], metric='euclidean')
            median_dist = np.median(distances)
            gamma_range = np.logspace(
                np.log10(0.01 / median_dist**2),
                np.log10(100 / median_dist**2),
                10
            )
        
        results = []
        
        for n_clusters in n_clusters_range:
            for gamma in gamma_range:
                # Compute affinity with specific gamma
                affinity = rbf_kernel(self.X, gamma=gamma)
                
                # Run spectral clustering
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
                
                try:
                    labels = spectral.fit_predict(affinity)
                    silhouette = silhouette_score(self.X, labels, sample_size=5000)
                    
                    results.append({
                        'n_clusters': n_clusters,
                        'gamma': gamma,
                        'silhouette': silhouette
                    })
                except:
                    pass
        
        results_df = pd.DataFrame(results)
        
        # Plot heatmap
        pivot = results_df.pivot(index='n_clusters', columns='gamma', values='silhouette')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Spectral Clustering Parameter Tuning')
        plt.xlabel('Gamma')
        plt.ylabel('Number of Clusters')
        plt.tight_layout()
        plt.show()
        
        return results_df
