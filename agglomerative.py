import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.transform import resize

def apply_agglomerative_clustering(image, num_clusters):
    
    # Downscale if image is too large
    max_size = 200
    height, width = image.shape[:2]

    if max(height, width) > max_size:
        scale_factor = max_size / max(height, width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        image = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    original_shape = image.shape

    # Check if grayscale or color
    is_color = False
    if len(image.shape) == 2:
        img_flat = image.reshape((-1, 1))
    elif len(image.shape) == 3 and image.shape[2] == 3:
        img_flat = image.reshape((-1, 3))
        is_color = True
    else:
        raise ValueError("Unsupported image format.")

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    labels = clustering.fit_predict(img_flat)

    # Create an empty output image
    if is_color:
        segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    else:
        segmented_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Assign mean color of each cluster
    for cluster_id in range(num_clusters):
        mask = (labels == cluster_id)

        if is_color:
            mean_color = img_flat[mask].mean(axis=0)
            segmented_image.reshape((-1, 3))[mask] = mean_color
        else:
            mean_intensity = img_flat[mask].mean()
            segmented_image.reshape(-1)[mask] = mean_intensity

    return segmented_image
