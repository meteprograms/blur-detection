import cv2
import numpy as np

BLUR = False

def variance_of_laplacian(image):
    # Compute the Laplacian of the image and return the variance
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blurry_parts(image_path, threshold=100):
    # Load the image from the path
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    BLUR = False
    
    # Get image dimensions
    height, width = gray.shape
    
    # Number of splits (4x4 grid means 16 regions)
    splits = 3
    h_split = height // splits
    w_split = width // splits
    
    # Analyze each region in the 4x4 grid
    for i in range(splits):
        for j in range(splits):
            # Define the region coordinates
            x_start = i * h_split
            x_end = (i + 1) * h_split
            y_start = j * w_split
            y_end = (j + 1) * w_split
            
            # Extract the region
            region = gray[x_start:x_end, y_start:y_end]
            
            # Compute variance of Laplacian (sharpness indicator)
            variance = variance_of_laplacian(region)
            region_name = f"Region ({i+1}, {j+1})"
            
            # Print the result for each region
            print(f"{region_name} variance: {variance}")
            
            # Check if the variance is below the threshold, indicating a blurry region
            if variance < threshold:
                BLUR = True
                print(f"{region_name} is blurry")

            else:
                print(f"{region_name} is sharp")

    return BLUR

# Example usage:
image_path = 'unblurred_example3.jpg'
BLUR = detect_blurry_parts(image_path)

if BLUR:
    print("***Image is blurry!***")
else:
    print("***Image is NOT blurry!***")