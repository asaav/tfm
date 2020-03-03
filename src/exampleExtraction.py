import os
import pickle
import sys
import cv2

from comargs import process_args
from PyQt5 import QtWidgets

WSIZE = 28

def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def main():
    img = None
    def mouse_events(event, x, y, flags, params):
        positive_examples = len([name for name in os.listdir('./yes') if os.path.isfile('./yes/' + name)])
        negative_examples = len([name for name in os.listdir('./no') if os.path.isfile('./no/' + name)])
        region = img[y - WSIZE//2:y+WSIZE + WSIZE//2,x-WSIZE//2:x+WSIZE + WSIZE//2]
        img_cloned = img.copy()
        if event == cv2.EVENT_MOUSEMOVE:
            cv2.rectangle(img_cloned, (x-WSIZE//2,y-WSIZE//2), (x+WSIZE + WSIZE//2, y+WSIZE + WSIZE//2), (0,255,0))
            cv2.imshow("capture", img_cloned)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imwrite('./yes/' + str(positive_examples + 1) + '.png', region)
        if event == cv2.EVENT_MBUTTONDOWN:
            cv2.imwrite('./no/' + str(negative_examples + 1) + '.png', region)
        

    args = process_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(sys.argv[1] + " couldn't be opened.",
              file=sys.stderr)
        exit(1)

    play_video = True
    ret = None

    while cap.isOpened():
        if play_video:

            # read frame
            ret, frame = cap.read()

            if ret:
                # scale image
                frame, _, _ = scale_image(frame, args.scale)
                cv2.imshow('original', frame)
        img = frame
        # end video if q is pressed or no frame was read
        key = cv2.waitKey(20)
        if (key == ord('q')) or (not ret):
            break
        elif key == ord('c'):
            play_video = False
            cv2.imshow('capture', img)
            cv2.setMouseCallback('capture', mouse_events)
        # space to pause
        elif key == ord(' '):
            play_video = not play_video
        # k to rewind 5 seconds
        elif key == ord('k'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time - 5000)
        # l to forward 5 seconds
        elif key == ord('l'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time + 5000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main()
    app.exit()
