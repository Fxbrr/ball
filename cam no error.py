import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        # Handle case when camera is not working
        print("Error: Could not capture frame from camera")
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds of orange color in HSV
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a series of morphological transformations to remove noise and fill in gaps
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if a contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # Get the (x, y) coordinates and radius of the ball
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Calculate the distance from the camera to the ball
            focal_length = 755.81  # Update this value with your camera's focal length
            ball_diameter = 4.3  # Update this value with the diameter of your ball in cm
            distance = (focal_length * ball_diameter) / (2 * radius)

            # Draw the circle and ball information on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, "Orange Ball", (int(
                x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Distance: {:.2f} cm".format(
                distance), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "radius: {:.2f}".format(
                radius), (10, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
