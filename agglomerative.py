import numpy as np
from skimage.transform import resize

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def apply_agglomerative_clustering(image, num_clusters):
   
    #  Downscale 
    target_size = 60
    height, width = image.shape[:2]

    if max(height, width) > target_size:
        scale_factor = target_size / max(height, width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        from skimage.transform import resize
        image = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    original_shape = image.shape

    #  Prepare feature vectors 
    is_color = False
    if len(image.shape) == 2:
        img_flat = image.reshape((-1, 1))
    elif len(image.shape) == 3 and image.shape[2] == 3:
        img_flat = image.reshape((-1, 3))
        is_color = True
    else:
        raise ValueError("Unsupported image format.")

    num_pixels = img_flat.shape[0]

    #  Initialize each pixel as its own cluster 
    clusters = {i: [i] for i in range(num_pixels)}
    centroids = {i: img_flat[i] for i in range(num_pixels)} #feature vector of each pixel

    #  Compute initial pairwise distances 
    distance_matrix = np.full((num_pixels, num_pixels), np.inf)

    for i in range(num_pixels):
        for j in range(i+1, num_pixels):
            distance_matrix[i, j] = euclidean_distance(centroids[i], centroids[j])

    #  Agglomerative merging 
    while len(clusters) > num_clusters:
        # Find the two closest clusters
        idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        i, j = idx

        if i > j:
            i, j = j, i  # Ensure i < j

        # Merge clusters
        clusters[i].extend(clusters[j])

        # Update centroid
        centroids[i] = np.mean(img_flat[clusters[i]], axis=0)

        # Delete old cluster j
        del clusters[j]
        del centroids[j]

        # Remove j row and column from distance matrix
        distance_matrix[j, :] = np.inf
        distance_matrix[:, j] = np.inf

        # Update distances from new cluster i
        for k in clusters:
            if k != i:
                dist = euclidean_distance(centroids[i], centroids[k])
                distance_matrix[min(i, k), max(i, k)] = dist

    #  Assign cluster labels 
    label_image = np.zeros(num_pixels, dtype=int)

    for cluster_id, pixel_indices in enumerate(clusters.values()):
        for idx in pixel_indices:
            label_image[idx] = cluster_id

    label_image = label_image.reshape((image.shape[0], image.shape[1]))

    
    if is_color:
        segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    else:
        segmented_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for cluster_id in range(num_clusters):
        mask = (label_image == cluster_id)

        if is_color:
            mean_color = img_flat[mask.reshape(-1)].mean(axis=0)
            segmented_image.reshape((-1, 3))[mask.reshape(-1)] = mean_color
        else:
            mean_intensity = img_flat[mask.reshape(-1)].mean()
            segmented_image.reshape(-1)[mask.reshape(-1)] = mean_intensity

    return segmented_image
