import os
import sys

import cv2
import tensorflow as tf
import numpy as np
from comargs import tracker_args


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh),
                      interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def main():

    args = tracker_args()

    img = None
    if os.path.exists(args.model):
        model = tf.keras.models.load_model(args.model, compile=False)
        print("Model " + args.model + " successfully loaded")
    else:
        sys.exit("Model " + args.model + " does not exist")

    if not os.path.exists("hardNegatives"):
        os.mkdir("hardNegatives")

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
            ret, raw_frame = cap.read()
            if ret:
                # scale image
                frame, _, _ = scale_image(raw_frame, args.scale)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = model.predict(np.expand_dims(frame_rgb, axis=0))
                img = np.squeeze(img, axis=0)
                cv2.imshow('output', img)
                cv2.imshow('original', frame)
        img = frame
        # end video if q is pressed or no frame was read
        key = cv2.waitKey(20)
        if (key == ord('q')) or (not ret):
            break
        elif key == ord('c'):
            # Save current frame without scaling
            path, dirs, files = next(os.walk("hardNegatives"))
            img_count = len(files)
            name = os.path.join("hardNegatives", str(img_count + 1) + ".png")
            print("Saved " + name)
            cv2.imwrite(name, raw_frame)
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
    main()
