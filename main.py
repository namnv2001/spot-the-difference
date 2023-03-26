import cv2
import numpy as np
import random

# Load input image
img = cv2.imread('input_image.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection algorithm
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours of the edges
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Select 10 random edge-covered shapes
random_contours = random.sample(contours, 10)

# Change color inside the edge of each selected shape
for contour in random_contours:
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    color = np.random.randint(0, 256, size=3)
    img[mask == 255] = color

# Save output image
cv2.imwrite('output_image.jpg', img)
