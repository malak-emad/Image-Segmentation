import numpy as np

def apply_region_growing(image, seed_points, threshold):
    
    # Determine if the image is grayscale or color
    if len(image.shape) == 2:
        is_color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        is_color = True
    else:
        raise ValueError("Unsupported image format.")

    height, width = image.shape[:2]

    # Initialize segmentation mask
    mask = np.zeros((height, width), dtype=np.uint8)  # 0 = background, 255 = region

    # Keep track of visited pixels
    visited = np.zeros((height, width), dtype=bool)

    # Create a list (queue) to manage growth
    to_visit = list(seed_points)

    # Get the pixel values at the seeds
    if is_color:
        seed_values = [image[y, x, :].astype(np.float32) for y, x in seed_points]
    else:
        seed_values = [image[y, x].astype(np.float32) for y, x in seed_points]

    # Start region growing
    while to_visit:
        y, x = to_visit.pop(0)

        if visited[y, x]:
            continue

        visited[y, x] = True
        mask[y, x] = 255

        # Check 4-connected neighbors (up, down, left, right)
        neighbors = [
            (y-1, x),
            (y+1, x),
            (y, x-1),
            (y, x+1)
        ]

        for ny, nx in neighbors:
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                if is_color:
                    pixel_value = image[ny, nx, :].astype(np.float32)
                    # Compare to all seeds and take the min distance
                    min_distance = min(np.linalg.norm(pixel_value - seed_val) for seed_val in seed_values)
                    if min_distance < threshold:
                        to_visit.append((ny, nx))
                else:
                    pixel_value = image[ny, nx].astype(np.float32)
                    min_distance = min(abs(pixel_value - seed_val) for seed_val in seed_values)
                    if min_distance < threshold:
                        to_visit.append((ny, nx))

    return mask
