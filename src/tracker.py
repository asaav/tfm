import os.path
from os import path
import sys

import cv2
import numpy as np
import tensorflow as tf

import comargs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = None
cap = None
args = comargs.tracker_args()
if path.exists(args.model):
    model = tf.keras.models.load_model(args.model, compile=False)
    print("Model " + args.model + " successfully loaded")
else:
    sys.exit("Model " + args.model + " does not exist")

if path.exists(args.video):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(sys.argv[1] + " couldn't be opened.",
              file=sys.stderr)
        sys.exit(1)
else:
    sys.exit("Video " + args.model + " does not exist")


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh),
                      interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def main():
    img_array = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            scaled, width, height = scale_image(frame, args.scale)
            frame_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            img = model.predict(np.expand_dims(frame_rgb, axis=0))
            img = np.squeeze(img, axis=0)
            kernel = np.ones((5, 5), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img_array.append((img*255).astype(np.uint8))
            # ret, thresh1 = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)

            cv2.imshow('test', img)
            cv2.imshow('org', scaled)
        key = cv2.waitKey(20)
        if (key == ord('q')) or (not ret):
            break
        # k to rewind 5 seconds
        elif key == ord('k'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time - 5000)
        # l to forward 5 seconds
        elif key == ord('l'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time + 5000)
    h, w = img_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("test.avi", fourcc, 30.0, (w, h), False)

    for img in img_array:
        out.write(img)
    out.release()


if __name__ == "__main__":
    main()
