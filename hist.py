import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image):
    """Computes histogram of a grayscale image."""
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist

def compute_cdf(hist):
    """Computes cumulative distribution function (CDF) from histogram."""
    cdf = hist.cumsum()
    return cdf / cdf[-1]  # Normalize

def histogram_equalization(image):
    """Manually performs histogram equalization."""
    hist = compute_histogram(image)
    cdf = compute_cdf(hist)

    # Mapping old pixel values to new ones
    equalized = np.interp(image.flatten(), np.arange(256), cdf * 255)
    
    return equalized.reshape(image.shape).astype(np.uint8)

def histogram_specification(source, reference):
    """Matches histogram of source image to reference image."""
    source_hist = compute_histogram(source)
    reference_hist = compute_histogram(reference)

    source_cdf = compute_cdf(source_hist)
    reference_cdf = compute_cdf(reference_hist)

    # Create a mapping from source to reference using nearest match
    mapping = np.zeros(256, dtype=np.uint8)
    ref_values = np.arange(256)

    for src_value in range(256):
        closest_match = np.abs(reference_cdf - source_cdf[src_value]).argmin()
        mapping[src_value] = ref_values[closest_match]

    # Apply mapping
    matched_image = mapping[source]
    return matched_image.astype(np.uint8)

def plot_histogram(image, title):
    """Plots histogram of an image."""
    hist = compute_histogram(image)
    plt.bar(np.arange(256), hist, color='gray')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Load grayscale images
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = histogram_equalization(image)

# Apply histogram specification
matched_image = histogram_specification(image, reference)

# Display images
plt.figure(figsize=(12, 6))

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plot_histogram(image, "Histogram of Original Image")

plt.subplot(3, 3, 3)
plt.imshow(reference, cmap='gray')
plt.title("Reference Image")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(3, 3, 5)
plot_histogram(equalized_image, "Histogram of Equalized Image")

plt.subplot(3, 3, 6)
plt.imshow(matched_image, cmap='gray')
plt.title("Histogram Matched Image")
plt.axis("off")

plt.subplot(3, 3, 7)
plot_histogram(matched_image, "Histogram of Matched Image")

plt.tight_layout()
plt.show()
