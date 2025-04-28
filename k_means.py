import numpy as np

def apply_kmeans_clustering(image, num_clusters, threshold):
    """
    parameters:
    - image: Input image as numpy array (height, width, 3)
    - num_clusters: Number of clusters (k)
    - threshold: Convergence threshold
    
    Returns:
    - segmented_image: Image with pixels replaced by their cluster centroids (in color)
    """
    # Step 1: determine no. pixels in img
    height, width, channels = image.shape
    num_pixels = height * width
    
    # Reshape img to 2D array of pixels (num_pixels x channels)
    pixels = image.reshape(num_pixels, channels).astype(np.float64)
    
    # Step 2: Randomly generate initial centroids
    indices = np.random.choice(num_pixels, num_clusters, replace=False)
    centroids = pixels[indices]
    
    # Initialize variables for convergence check
    prev_centroids = np.zeros_like(centroids)
    iterations = 0
    
    # Store cluster assignments
    labels = np.zeros(num_pixels, dtype=np.int32)
    
    # Main k-means loop
    while True:
        iterations += 1
        
        # Step 3: Vectorized assignment of pixels to nearest centroids
        # Calculate distances from all pixels to all centroids at once
        distances = np.zeros((num_pixels, num_clusters))
        for k in range(num_clusters):
            # Broadcasting to calculate distances efficiently
            distances[:, k] = np.sum((pixels - centroids[k]) ** 2, axis=1)
        
        # Assign each pixel to the closest centroid
        labels = np.argmin(distances, axis=1)
        
        # Save current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # Step 4: Recompute the centroids for each cluster
        for k in range(num_clusters):
            cluster_pixels = pixels[labels == k]
            if len(cluster_pixels) > 0:
                centroids[k] = np.mean(cluster_pixels, axis=0)
        
        # Step 5: Check for convergence - whether centroids have moved less than threshold
        centroid_shift = np.sqrt(np.sum((centroids - prev_centroids) ** 2, axis=1))
        if np.all(centroid_shift < threshold):
            print(f"K-means converged after {iterations} iterations")
            break
            
        # Safety check to prevent infinite looping
        if iterations > 100:
            print(f"Warning: Maximum iterations (100) reached before convergence")
            break
    
    # Create segmented image by replacing each pixel with its centroid color
    segmented_pixels = centroids[labels].reshape(height, width, channels)
    segmented_image = np.uint8(np.clip(segmented_pixels, 0, 255))
    
    # Return the color segmented image
    return segmented_image

def apply_kmeans(image, num_clusters_str, threshold_str):
    try:
        # Convert parameters from strings to appropriate types
        num_clusters = int(num_clusters_str)
        threshold = float(threshold_str)
        
        # Run k-means clustering
        result = apply_kmeans_clustering(image, num_clusters, threshold)
        
        return result
    
    except Exception as e:
        print(f"Error in K-means: {e}")
        return np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)  # Return color black image