import numpy as np

def apply_kmeans_clustering(image, num_clusters, threshold):
    # Step 1: determine no. pixels in img
    height, width, channels = image.shape #EX: 800,600
    num_pixels = height * width #EX: 800x600 =480000
    
    #reshape img to 2D array of pixels (num_pixels x channels)
    pixels = image.reshape(num_pixels, channels).astype(np.float64) #EX: 480000,3, each row represent a pixel, each col represent a color channel
    
    # Step 2: Randomly generate initial centroids
    indices = np.random.choice(num_pixels, num_clusters, replace=False) #initialize centroids according to num_clusters EX:5
    centroids = pixels[indices] #contain RGB values of initialized centroids
    
    #initialize variables for convergence check
    prev_centroids = np.zeros_like(centroids)
    iterations = 0
    
    #store cluster assignments
    labels = np.zeros(num_pixels, dtype=np.int32)
    
    #k-means iterative loop
    while True:
        iterations += 1
        
        # Step 3: Vectorized assignment of pixels to nearest centroids
        #calculate distances from all pixels to all centroids at once
        distances = np.zeros((num_pixels, num_clusters)) #Distances dimension = 480000 x 5
        for k in range(num_clusters):
            # broadcasting to calculate distances efficiently
            distances[:, k] = np.sum((pixels - centroids[k]) ** 2, axis=1) #no need for square root as it inc. computational cost without affect result
        
        #assign each pixel to its closest centroid
        labels = np.argmin(distances, axis=1)
        
        #save current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # Step 4: Recompute centroids(means) for each cluster
        for k in range(num_clusters):
            cluster_pixels = pixels[labels == k]
            if len(cluster_pixels) > 0:
                centroids[k] = np.mean(cluster_pixels, axis=0)
        
        # Step 5: Check for convergence by threshold
        centroid_shift = np.sqrt(np.sum((centroids - prev_centroids) ** 2, axis=1))
        if np.all(centroid_shift < threshold):
            print(f"K-means converged after {iterations} iterations")
            break
            
        #safety check to prevent infinite looping
        if iterations > 100:
            print(f"Warning: Maximum iterations (100) reached before convergence")
            break
    
    #create segmented image by replacing each pixel with its centroid color
    segmented_pixels = centroids[labels].reshape(height, width, channels)
    segmented_image = np.uint8(np.clip(segmented_pixels, 0, 255))
    
    return segmented_image

def apply_kmeans(image, num_clusters_str, threshold_str):
    try:
        #convert user parameters from strings to appropriate types
        num_clusters = int(num_clusters_str)
        threshold = float(threshold_str)
        result = apply_kmeans_clustering(image, num_clusters, threshold)
        
        return result
    
    except Exception as e:
        print(f"Error in K-means: {e}")
        return np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)  