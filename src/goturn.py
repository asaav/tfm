import sys

import cv2


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh

bbox = None
tracking = False

video = cv2.VideoCapture(sys.argv[1])
 
# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame
ok,frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    frame = scale_image(frame, 0.8)[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    if tracking:
        ok, bbox = tracker.update(hsv)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
    # Draw bounding box
    if tracking and ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    elif tracking :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    # Display result
    cv2.imshow("Tracking", frame)
  
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('t'):
        bbox = cv2.selectROI(frame, False)
        tracker = cv2.TrackerGOTURN_create()
        tracker.init(hsv,bbox)
        tracking = True
