import cv2

# Load the two images to be compared
img1 = cv2.imread('input_image.jpg')
img2 = cv2.imread('output_image.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate the absolute difference between the two grayscale images
diff = cv2.absdiff(gray1, gray2)

# Threshold the difference image to highlight the differences
thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]

# Find the contours of the differences
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw circles around the contours of the differences
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(img2, center, radius, (0, 255, 0), 2)

# Show the output image with circles around the differences
cv2.imshow("Output", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
