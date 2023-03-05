import cv2
import numpy as np

# Define the lower and upper bounds of the white color  HUE,SATURATION,VALUE
lower_bound = np.array([0, 80, 100])
upper_bound = np.array([20, 255, 255])
# Calculate the distance camera to ball
known_width = 4.3  # in cm
# in pixels (example value, replace with your actual focal length)
focal_length = 755.81
# the height of camera from ground
height = 4

# Load the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the white color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are contours, find the largest one and draw a circle around it
    if len(contours) > 0:
        ball = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Display the coordinates on the image
        cv2.putText(frame, "Coordinates: ({}, {})".format(
            center[0], center[1]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calculate the distance camera to ball
        width_in_image = 2 * radius
        d1 = (known_width * focal_length) / width_in_image

        # distance robot to ball
        d2 = (d1**2 - height**2)**0.5

        # Display the distance on the image
        cv2.putText(frame, "Distance1: {:.2f} cm".format(
            d1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Distance2: {:.2f} cm".format(
            d2), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "radius: {:.2f}".format(
            radius), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Show the result
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Result", blur)
    cv2.imshow("normal colour", frame)
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and destroy the windows
