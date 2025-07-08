import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from pathlib import Path

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    print("Note: 'kneed' package not available. Install with 'pip install kneed' for automatic elbow detection.")


class FloorplanCategorizer:
    def __init__(self):
        self.living_room_color = (0xEE, 0xE8, 0xAA)  # RGB values
        self.entrance_color = (255, 0, 0)  # Red for entrance
        
    def extract_features(self, image):
        """Extract comprehensive features from floorplan image"""
        # Convert to RGB if needed
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # 1. OVERALL SHAPE FEATURES
        shape_features = self._extract_shape_features(image_rgb)
        
        # 2. LIVING ROOM FEATURES
        living_features = self._extract_living_room_features(image_rgb)
        
        # 3. ENTRANCE FEATURES
        entrance_features = self._extract_entrance_features(image_rgb)
        
        return np.concatenate([shape_features, living_features, entrance_features])
    
    def _extract_shape_features(self, image):
        """Extract overall shape characteristics"""
        # Create binary mask for the overall floorplan
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find main contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Hu moments (7 invariant moments)
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h
        
        # Convexity and solidity
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Rectangularity
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # Compactness
        compactness = (perimeter ** 2) / area if area > 0 else 0
        
        return np.array([
            area, perimeter, aspect_ratio, convexity, 
            rectangularity, compactness, *hu_moments
        ])
    
    def _extract_living_room_features(self, image):
        """Extract living room specific features"""
        # Create mask for living room color
        living_mask = np.all(image == self.living_room_color, axis=-1)
        
        if not np.any(living_mask):
            return np.zeros(8)  # Return zeros if no living room found
        
        # Find living room contours
        contours, _ = cv2.findContours(living_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(8)
        
        # Get largest living room area (in case of multiple)
        living_contour = max(contours, key=cv2.contourArea)
        
        # Living room area and shape
        living_area = cv2.contourArea(living_contour)
        living_perimeter = cv2.arcLength(living_contour, True)
        
        # Total plan area
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        total_area = np.sum(binary > 0)
        
        # Relative size
        area_ratio = living_area / total_area if total_area > 0 else 0
        
        # Living room centroid
        M = cv2.moments(living_contour)
        if M['m00'] != 0:
            living_cx = M['m10'] / M['m00']
            living_cy = M['m01'] / M['m00']
        else:
            living_cx = living_cy = 0
        
        # Overall plan centroid
        plan_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if plan_contours:
            plan_contour = max(plan_contours, key=cv2.contourArea)
            plan_M = cv2.moments(plan_contour)
            if plan_M['m00'] != 0:
                plan_cx = plan_M['m10'] / plan_M['m00']
                plan_cy = plan_M['m01'] / plan_M['m00']
            else:
                plan_cx = plan_cy = 128  # Default to center
        else:
            plan_cx = plan_cy = 128
        
        # Relative position (normalized)
        rel_pos_x = (living_cx - plan_cx) / 128
        rel_pos_y = (living_cy - plan_cy) / 128
        
        # Living room compactness
        living_compactness = (living_perimeter ** 2) / living_area if living_area > 0 else 0
        
        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(living_contour)
        living_aspect_ratio = w / h if h > 0 else 1
        
        return np.array([
            area_ratio, rel_pos_x, rel_pos_y, living_compactness,
            living_aspect_ratio, living_area, living_perimeter, 
            living_cx / 256, living_cy / 256  # Normalized position
        ])
    
    def _extract_entrance_features(self, image):
        """Extract entrance location features"""
        # Find red pixels (entrance)
        entrance_mask = np.all(image == self.entrance_color, axis=-1)
        
        if not np.any(entrance_mask):
            return np.zeros(6)  # Return zeros if no entrance found
        
        # Find entrance centroid
        entrance_points = np.where(entrance_mask)
        entrance_cx = np.mean(entrance_points[1])
        entrance_cy = np.mean(entrance_points[0])
        
        # Normalize entrance position
        norm_entrance_x = entrance_cx / 256
        norm_entrance_y = entrance_cy / 256
        
        # Determine which wall/side (rough approximation)
        # 0: top, 1: right, 2: bottom, 3: left
        distances_to_edges = [
            entrance_cy,  # distance to top
            256 - entrance_cx,  # distance to right
            256 - entrance_cy,  # distance to bottom
            entrance_cx  # distance to left
        ]
        closest_wall = np.argmin(distances_to_edges)
        
        # Distance from corners
        corners = [(0, 0), (255, 0), (255, 255), (0, 255)]
        min_corner_dist = min([
            np.sqrt((entrance_cx - cx)**2 + (entrance_cy - cy)**2) 
            for cx, cy in corners
        ]) / 256  # Normalize
        
        return np.array([
            norm_entrance_x, norm_entrance_y, closest_wall,
            min_corner_dist, entrance_cx, entrance_cy
        ])
    
    def process_directory_to_csv(self, image_directory, output_csv_path, 
                                    supported_formats=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            """
            Process all images in a directory and save features to CSV
            
            Args:
                image_directory (str): Path to directory containing floorplan images
                output_csv_path (str): Path where CSV file will be saved
                supported_formats (tuple): Tuple of supported image file extensions
            """
            print(f"Processing images from directory: {image_directory}")
            
            # Get all image files
            image_files = []
            for format_ext in supported_formats:
                pattern = os.path.join(image_directory, f"*{format_ext}")
                image_files.extend(glob.glob(pattern))
                pattern = os.path.join(image_directory, f"*{format_ext.upper()}")
                image_files.extend(glob.glob(pattern))
            
            if not image_files:
                print(f"No images found in {image_directory}")
                return
            
            print(f"Found {len(image_files)} images to process")
            
            # Define column names for the CSV
            column_names = self._get_feature_column_names()
            
            # Initialize list to store all data
            all_data = []
            failed_images = []
            
            # Process each image
            for i, image_path in enumerate(image_files):
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        failed_images.append(image_path)
                        continue
                    
                    # Extract features
                    features = self.extract_features(image)
                    
                    # Create row data with filename
                    row_data = {
                        'filename': os.path.basename(image_path),
                        'filepath': image_path,
                        'image_width': image.shape[1],
                        'image_height': image.shape[0],
                        'image_channels': image.shape[2] if len(image.shape) > 2 else 1
                    }
                    
                    # Add feature values
                    for col_name, feature_value in zip(column_names, features):
                        row_data[col_name] = feature_value
                    
                    all_data.append(row_data)
                    
                    # Progress update
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    failed_images.append(image_path)
                    continue
            
            # Create DataFrame and save to CSV
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(output_csv_path, index=False)
                print(f"\nFeatures saved to: {output_csv_path}")
                print(f"Successfully processed: {len(all_data)} images")
                print(f"Failed to process: {len(failed_images)} images")
                
                if failed_images:
                    print("Failed images:")
                    for failed_img in failed_images[:10]:  # Show first 10 failed images
                        print(f"  - {failed_img}")
                    if len(failed_images) > 10:
                        print(f"  ... and {len(failed_images) - 10} more")
                
                # Display basic statistics
                print(f"\nDataset summary:")
                print(f"Total features per image: {len(column_names)}")
                print(f"CSV shape: {df.shape}")
                print(f"CSV columns: {list(df.columns)}")
                
                return df
            else:
                print("No images were successfully processed!")
                return None
    

    def _get_feature_column_names(self):
        """
        Define meaningful column names for all extracted features
        """
        # Shape features (13 features)
        shape_columns = [
            'shape_area', 'shape_perimeter', 'shape_aspect_ratio', 
            'shape_convexity', 'shape_rectangularity', 'shape_compactness',
            'shape_hu_moment_1', 'shape_hu_moment_2', 'shape_hu_moment_3',
            'shape_hu_moment_4', 'shape_hu_moment_5', 'shape_hu_moment_6',
            'shape_hu_moment_7'
        ]
        
        # Living room features (9 features)
        living_columns = [
            'living_area_ratio', 'living_rel_pos_x', 'living_rel_pos_y',
            'living_compactness', 'living_aspect_ratio', 'living_area',
            'living_perimeter', 'living_norm_pos_x', 'living_norm_pos_y'
        ]
        
        # Entrance features (6 features)
        entrance_columns = [
            'entrance_norm_pos_x', 'entrance_norm_pos_y', 'entrance_closest_wall',
            'entrance_min_corner_dist', 'entrance_pixel_x', 'entrance_pixel_y'
        ]
        
        return shape_columns + living_columns + entrance_columns
    
    def load_features_from_csv(self, csv_path):
        """
        Load previously extracted features from CSV file
        
        Args:
            csv_path (str): Path to CSV file with extracted features
            
        Returns:
            dict: Dictionary containing features and metadata
        """
        print(f"Loading features from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Extract feature columns (exclude metadata columns)
        feature_columns = self._get_feature_column_names()
        features = df[feature_columns].values
        
        # Extract metadata
        metadata = df[['filename', 'filepath', 'image_width', 'image_height', 'image_channels']].copy()
        
        print(f"Loaded {len(features)} feature vectors with {features.shape[1]} features each")
        
        return {
            'features': features,
            'metadata': metadata,
            'dataframe': df,
            'feature_columns': feature_columns
        }
    
    def categorize_from_csv(self, csv_path, n_clusters=20):
        """
        Load features from CSV and perform clustering
        
        Args:
            csv_path (str): Path to CSV file with extracted features
            n_clusters (int): Number of clusters for K-means
            
        Returns:
            dict: Clustering results with metadata
        """
        # Load features
        data = self.load_features_from_csv(csv_path)
        features = data['features']
        metadata = data['metadata']
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Keep 95% variance
        features_pca = pca.fit_transform(features_scaled)
        
        print(f"Reduced features from {features.shape[1]} to {features_pca.shape[1]} dimensions")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)
        
        # Add cluster labels to metadata
        metadata_with_clusters = metadata.copy()
        metadata_with_clusters['cluster_id'] = cluster_labels
        
        # Hierarchical clustering for comparison
        linkage_matrix = linkage(features_pca, method='ward')
        
        return {
            'features': features,
            'features_scaled': features_scaled,
            'features_pca': features_pca,
            'cluster_labels': cluster_labels,
            'metadata': metadata_with_clusters,
            'kmeans_model': kmeans,
            'scaler': scaler,
            'pca': pca,
            'linkage_matrix': linkage_matrix,
            'feature_columns': data['feature_columns']
        }
    
    def categorize_floorplans(self, images, n_clusters=20):
        """Categorize all floorplans"""
        print("Extracting features from all floorplans...")
        features = []
        
        for i, image in enumerate(images):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(images)} images")
            
            feature_vector = self.extract_features(image)
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Keep 95% variance
        features_pca = pca.fit_transform(features_scaled)
        
        print(f"Reduced features from {features.shape[1]} to {features_pca.shape[1]} dimensions")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)
        
        # Hierarchical clustering for comparison
        linkage_matrix = linkage(features_pca, method='ward')
        
        return {
            'features': features,
            'features_scaled': features_scaled,
            'features_pca': features_pca,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans,
            'scaler': scaler,
            'pca': pca,
            'linkage_matrix': linkage_matrix
        }
    
    def analyze_clusters(self, results):
        """Analyze the characteristics of each cluster"""
        features = results['features']
        labels = results['cluster_labels']
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            
            # Calculate cluster statistics
            cluster_analysis[cluster_id] = {
                'count': np.sum(cluster_mask),
                'mean_features': np.mean(cluster_features, axis=0),
                'std_features': np.std(cluster_features, axis=0),
                'representative_indices': np.where(cluster_mask)[0][:5]  # First 5 examples
            }
        
        return cluster_analysis
    

    def evaluate_clustering_metrics(self, csv_path, k_range=None, plot_results=True, 
                                  figsize=(15, 12), save_plots=None):
        """
        Evaluate clustering performance across different k values using multiple metrics.
        Helps choose optimal number of clusters using elbow method and other criteria.
        
        Args:
            csv_path (str): Path to CSV file with extracted features
            k_range (list): Range of k values to test. Default: range(2, 21)
            plot_results (bool): Whether to create plots
            figsize (tuple): Figure size for plots
            save_plots (str): Path to save plots (optional)
            
        Returns:
            dict: Dictionary containing all metrics and recommendations
        """
        print("Loading features and preparing data for clustering evaluation...")
        
        # Load features
        data = self.load_features_from_csv(csv_path)
        features = data['features']
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_scaled)
        
        print(f"Using {features_pca.shape[1]} PCA components for evaluation")
        
        # Set default k range
        if k_range is None:
            k_range = range(2, min(21, len(features_pca) // 10))  # Don't exceed reasonable limits
        
        print(f"Testing k values: {list(k_range)}")
        
        # Initialize metrics storage
        metrics = {
            'k_values': list(k_range),
            'inertia': [],           # Within-cluster sum of squares (lower is better)
            'silhouette': [],        # Silhouette score (higher is better, -1 to 1)
            'davies_bouldin': [],    # Davies-Bouldin index (lower is better)
            'calinski_harabasz': [], # Calinski-Harabasz index (higher is better)
            'gap_statistic': [],     # Gap statistic (higher is better)
            'dunn_index': []         # Dunn index (higher is better)
        }
        
        # Test each k value
        for k in k_range:
            print(f"Evaluating k={k}...")
            
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            
            # Calculate metrics
            metrics['inertia'].append(kmeans.inertia_)
            
            # Silhouette score
            if k > 1:
                sil_score = silhouette_score(features_pca, cluster_labels)
                metrics['silhouette'].append(sil_score)
            else:
                metrics['silhouette'].append(0)
            
            # Davies-Bouldin index
            db_score = davies_bouldin_score(features_pca, cluster_labels)
            metrics['davies_bouldin'].append(db_score)
            
            # Calinski-Harabasz index
            ch_score = calinski_harabasz_score(features_pca, cluster_labels)
            metrics['calinski_harabasz'].append(ch_score)
            
            # Gap statistic (simplified version)
            gap_stat = self._calculate_gap_statistic(features_pca, cluster_labels, k)
            metrics['gap_statistic'].append(gap_stat)
            
            # Dunn index (simplified version for large datasets)
            dunn_idx = self._calculate_dunn_index_fast(features_pca, cluster_labels)
            metrics['dunn_index'].append(dunn_idx)
        
        # Find optimal k values using different methods
        recommendations = self._find_optimal_k(metrics)
        
        # Create plots if requested
        if plot_results:
            self._plot_clustering_metrics(metrics, recommendations, figsize, save_plots)
        
        # Return comprehensive results
        results = {
            'metrics': metrics,
            'recommendations': recommendations,
            'scaler': scaler,
            'pca': pca,
            'features_pca': features_pca,
            'n_samples': len(features_pca),
            'n_features_original': features.shape[1],
            'n_features_pca': features_pca.shape[1]
        }
        
        # Print summary
        self._print_clustering_summary(results)
        
        return results
    
    def _calculate_gap_statistic(self, data, labels, k, n_refs=10):
        """Calculate simplified gap statistic"""
        try:
            # Calculate within-cluster dispersion for actual data
            kmeans_actual = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans_actual.fit(data)
            wk_actual = kmeans_actual.inertia_
            
            # Generate reference datasets and calculate their within-cluster dispersions
            wk_refs = []
            for _ in range(n_refs):
                # Generate random reference data with same bounds
                data_min, data_max = data.min(axis=0), data.max(axis=0)
                ref_data = np.random.uniform(data_min, data_max, data.shape)
                
                kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans_ref.fit(ref_data)
                wk_refs.append(kmeans_ref.inertia_)
            
            # Calculate gap statistic
            gap = np.log(np.mean(wk_refs)) - np.log(wk_actual)
            return gap
            
        except:
            return 0
    
    def _calculate_dunn_index_fast(self, data, labels, sample_size=1000):
        """Calculate simplified Dunn index for large datasets"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0
            
            # Sample data if too large
            if len(data) > sample_size:
                indices = np.random.choice(len(data), sample_size, replace=False)
                data_sample = data[indices]
                labels_sample = labels[indices]
            else:
                data_sample = data
                labels_sample = labels
            
            # Calculate minimum inter-cluster distance
            min_inter_dist = float('inf')
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    cluster_i = data_sample[labels_sample == unique_labels[i]]
                    cluster_j = data_sample[labels_sample == unique_labels[j]]
                    
                    if len(cluster_i) > 0 and len(cluster_j) > 0:
                        # Use centroid distance as approximation
                        dist = np.linalg.norm(np.mean(cluster_i, axis=0) - np.mean(cluster_j, axis=0))
                        min_inter_dist = min(min_inter_dist, dist)
            
            # Calculate maximum intra-cluster distance
            max_intra_dist = 0
            for label in unique_labels:
                cluster_data = data_sample[labels_sample == label]
                if len(cluster_data) > 1:
                    # Use maximum distance from centroid as approximation
                    centroid = np.mean(cluster_data, axis=0)
                    distances = np.linalg.norm(cluster_data - centroid, axis=1)
                    max_intra_dist = max(max_intra_dist, np.max(distances))
            
            # Dunn index
            if max_intra_dist > 0:
                return min_inter_dist / max_intra_dist
            else:
                return 0
                
        except:
            return 0
    
    def _find_optimal_k(self, metrics):
        """Find optimal k using multiple methods"""
        k_values = metrics['k_values']
        recommendations = {}
        
        # 1. Elbow method for inertia
        if KNEED_AVAILABLE:
            try:
                kl = KneeLocator(k_values, metrics['inertia'], curve="convex", direction="decreasing")
                recommendations['elbow_inertia'] = kl.elbow
            except:
                recommendations['elbow_inertia'] = None
        else:
            # Simple elbow detection
            recommendations['elbow_inertia'] = self._simple_elbow_detection(k_values, metrics['inertia'])
        
        # 2. Maximum silhouette score
        max_sil_idx = np.argmax(metrics['silhouette'])
        recommendations['max_silhouette'] = k_values[max_sil_idx]
        
        # 3. Minimum Davies-Bouldin index
        min_db_idx = np.argmin(metrics['davies_bouldin'])
        recommendations['min_davies_bouldin'] = k_values[min_db_idx]
        
        # 4. Maximum Calinski-Harabasz index
        max_ch_idx = np.argmax(metrics['calinski_harabasz'])
        recommendations['max_calinski_harabasz'] = k_values[max_ch_idx]
        
        # 5. Maximum gap statistic
        max_gap_idx = np.argmax(metrics['gap_statistic'])
        recommendations['max_gap_statistic'] = k_values[max_gap_idx]
        
        # 6. Maximum Dunn index
        max_dunn_idx = np.argmax(metrics['dunn_index'])
        recommendations['max_dunn_index'] = k_values[max_dunn_idx]
        
        # 7. Consensus recommendation (most common suggestion)
        valid_recommendations = [v for v in recommendations.values() if v is not None]
        if valid_recommendations:
            recommendations['consensus'] = max(set(valid_recommendations), key=valid_recommendations.count)
        else:
            recommendations['consensus'] = None
        
        return recommendations
    
    def _simple_elbow_detection(self, k_values, inertias):
        """Simple elbow detection when kneed is not available"""
        try:
            # Calculate second derivative to find elbow
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            
            # Find the point with maximum second derivative (most curved)
            elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]
            else:
                return k_values[len(k_values)//2]  # Fallback to middle value
        except:
            return k_values[len(k_values)//2]  # Fallback to middle value
    
    def _plot_clustering_metrics(self, metrics, recommendations, figsize, save_plots):
        """Create comprehensive plots for clustering evaluation"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Clustering Evaluation Metrics', fontsize=16, fontweight='bold')
        
        k_values = metrics['k_values']
        
        # 1. Inertia (Elbow Method)
        axes[0, 0].plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0, 0].set_title('Elbow Method - Inertia')
        axes[0, 0].grid(True, alpha=0.3)
        if recommendations['elbow_inertia']:
            axes[0, 0].axvline(x=recommendations['elbow_inertia'], color='red', linestyle='--', 
                              label=f'Elbow at k={recommendations["elbow_inertia"]}')
            axes[0, 0].legend()
        
        # 2. Silhouette Score
        axes[0, 1].plot(k_values, metrics['silhouette'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=recommendations['max_silhouette'], color='red', linestyle='--',
                          label=f'Max at k={recommendations["max_silhouette"]}')
        axes[0, 1].legend()
        
        # 3. Davies-Bouldin Index
        axes[0, 2].plot(k_values, metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Number of Clusters (k)')
        axes[0, 2].set_ylabel('Davies-Bouldin Index')
        axes[0, 2].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axvline(x=recommendations['min_davies_bouldin'], color='red', linestyle='--',
                          label=f'Min at k={recommendations["min_davies_bouldin"]}')
        axes[0, 2].legend()
        
        # 4. Calinski-Harabasz Index
        axes[1, 0].plot(k_values, metrics['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Index')
        axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=recommendations['max_calinski_harabasz'], color='red', linestyle='--',
                          label=f'Max at k={recommendations["max_calinski_harabasz"]}')
        axes[1, 0].legend()
        
        # 5. Gap Statistic
        axes[1, 1].plot(k_values, metrics['gap_statistic'], 'co-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Gap Statistic')
        axes[1, 1].set_title('Gap Statistic (Higher is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=recommendations['max_gap_statistic'], color='red', linestyle='--',
                          label=f'Max at k={recommendations["max_gap_statistic"]}')
        axes[1, 1].legend()
        
        # 6. Dunn Index
        axes[1, 2].plot(k_values, metrics['dunn_index'], 'yo-', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Number of Clusters (k)')
        axes[1, 2].set_ylabel('Dunn Index')
        axes[1, 2].set_title('Dunn Index (Higher is Better)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axvline(x=recommendations['max_dunn_index'], color='red', linestyle='--',
                          label=f'Max at k={recommendations["max_dunn_index"]}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(save_plots, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_plots}")
        
        plt.show()
    
    def _print_clustering_summary(self, results):
        """Print summary of clustering evaluation"""
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Dataset: {results['n_samples']} samples, {results['n_features_original']} original features")
        print(f"PCA reduced to: {results['n_features_pca']} components")
        print()
        
        print("OPTIMAL K RECOMMENDATIONS:")
        print("-" * 30)
        recs = results['recommendations']
        
        for method, k_val in recs.items():
            if k_val is not None:
                method_name = method.replace('_', ' ').title()
                print(f"{method_name:25}: k = {k_val}")
        
        print()
        if recs['consensus']:
            print(f"🎯 CONSENSUS RECOMMENDATION: k = {recs['consensus']}")
        else:
            print("⚠️  No clear consensus - consider multiple k values")
        
        print("\n" + "="*60)
