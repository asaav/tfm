import os
import re
import sys
import random
from ast import literal_eval as make_tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from comargs import hnmex_args

# Print only tensorflow error and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


FIELD_SIZE = 56
drawing = False
np.random.seed(42)
random.seed(42)
args = hnmex_args()


def output2original(xout, yout):
    x_ = xout * 4
    y_ = yout * 4
    return x_, y_


def rectangleContains(rect1: (tuple, tuple), rect2: (tuple, tuple)):
    (x, y) = rect1[0]
    (x_end, y_end) = rect1[1]
    (x1, y1) = rect2[0]
    (x1_end, y1_end) = rect2[1]

    return (x <= x1 < x_end) and (y <= y1 < y_end) and\
           (x < x1_end <= x_end) and (y < y1_end <= y_end)


def create_symlinks(src_path, dst_path, leave_out=None):
    os.makedirs(dst_path, exist_ok=True)  # Ensure dst_path exists
    src_files = os.listdir(src_path)

    if leave_out is not None:
        indexes = range(0, len(src_files))
        chosen = np.random.choice(indexes, len(src_files)-leave_out,
                                  replace=False)
        src_files = np.array(src_files)[chosen]
    for file in src_files:
        # write symlink in dst_path
        os.symlink(os.path.abspath(os.path.join(src_path, file)),
                   os.path.join(dst_path, file))


def save_image(image, path):
    head, basename = os.path.split(path)
    os.makedirs(head, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def get_ball_rectangle(ball_r):
    # Get 56x56 example that contains
    start_pos, end_pos = ball_r
    w = end_pos[0] - start_pos[0]
    h = end_pos[1] - start_pos[1]

    x_offset = random.randint(0, FIELD_SIZE-w)
    y_offset = random.randint(0, FIELD_SIZE-h)
    x = start_pos[0] - x_offset
    y = start_pos[1] - y_offset
    return x, y


def introduce_new_examples(path, examples, positives):
    n_examples = 0
    folders = ["train", "test", "val"]
    end_path = "yes" if positives else "no"
    for folder in folders:
        n_examples += len(os.listdir(os.path.join(args.data,
                                                  folder, end_path)))

    indexes = range(0, len(examples))

    # Choose n random examples from list with n being min of the original
    # count or the full list
    n = min(int(n_examples*.5), len(examples))
    chosen_idx = np.random.choice(indexes, n, replace=False)

    print("Picked", n, "examples from the total of", len(examples),
          "examples")

    # Split examples in test, train and validation and write them
    trainval_ex, test_ex = train_test_split(
        chosen_idx, test_size=0.1, random_state=42)

    train_ex, val_ex = train_test_split(
        trainval_ex, test_size=0.111, random_state=42)  # 0.9*0.111 = 0.1

    print("Split examples: train -", len(train_ex),
          "val -", len(val_ex),
          "test -", len(test_ex))
    print("Saving examples in " + path)

    picks = [train_ex, test_ex, val_ex]

    for subdivision, picked in zip(folders, picks):
        for elem in picked:
            filename = os.path.join(path, subdivision,
                                    end_path, 'h'+str(elem)+'.png')
            save_image(examples[elem], filename)

    # Make symlinks of the original examples in the datasets leaving out
    # as many examples as we introduced previously

    for folder, pick in zip(folders, picks):
        print("Creating symbolic links for the original", folder,
              "negatives, leaving out", len(pick), "examples")
        create_symlinks(os.path.join(args.data, folder, end_path),
                        os.path.join(path, folder, end_path),
                        len(pick))


def main():
    # Check directories passed by argument
    if os.path.exists(args.model):
        model = tf.keras.models.load_model(args.model, compile=False)
        print("Model " + args.model + " successfully loaded")
    else:
        sys.exit("Model " + args.model + " does not exist")
    if not os.path.exists(args.hardNegatives):
        sys.exit("Path" + args.hardNegatives + " directory does not exist")
    if not os.path.exists(args.data):
        sys.exit("Path" + args.data + " directory does not exist")

    it_regex = r'it(\d+)(.+?)\.'
    match = re.match(it_regex, os.path.basename(args.model))
    iteration = match.group(1)
    name = match.group(2)

    # Create directories for hard negative examples, model and iteration
    # in case they dont already exist
    if not os.path.exists("hnmData"):
        os.mkdir("hnmData")
    if not os.path.exists(os.path.join("hnmData", name)):
        os.mkdir(os.path.join("hnmData", name))
    example_path = os.path.join("hnmData", name, 'it' + iteration)
    if not os.path.exists(example_path):
        print("Creating " + example_path + " folder.")
        os.mkdir(example_path)

    print("Calculating hard negatives and positives...")
    hard_negatives = []
    hard_positives = []
    for filename in os.listdir("hardNegatives"):

        # Get ball positions from filename
        regex = re.compile(r'\d+-(\d+,\d+)-(\d+,\d+)\D+$')
        match = re.match(regex, filename)
        start_pos = make_tuple(match.group(1))
        end_pos = make_tuple(match.group(2))
        example = cv2.imread(os.path.join("hardNegatives", filename))

        # Convert input image to RGB since model works in that color space
        # and get prediction for image
        img_rgb = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
        output = model.predict(np.expand_dims(img_rgb, axis=0))
        output = np.squeeze(output, axis=0)

        # Paint rectangle for visual reference
        color = (0, 255, 0)
        example = cv2.rectangle(example, start_pos, end_pos,
                                color, 1)

        # Get contours out of positve blobs in model output
        output = (output * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        # Filter contours (positve blobs) by size
        contours = [c for c in contours if cv2.contourArea(c) > 5]
        ball_is_shape = False
        for c in contours:
            # Compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # if (cX, cY) input patch doesn't contain the ball rectangle
            # in input image, it's a false positive
            oX, oY = output2original(cX, cY)
            original_rect = ((oX, oY), (oX + FIELD_SIZE, oY + FIELD_SIZE))
            ball_r = (start_pos, end_pos)
            if not rectangleContains(original_rect, ball_r):
                # Add example to hard negatives list
                h_neg = img_rgb[oY:oY+FIELD_SIZE, oX:oX+FIELD_SIZE]
                hard_negatives.append(h_neg)
            else:
                ball_is_shape = True
        if not ball_is_shape:
            x, y = get_ball_rectangle(ball_r)
            h_pos = img_rgb[y:y+FIELD_SIZE, x:x+FIELD_SIZE]
            hard_positives.append(h_pos)

    # Once we have every example, pick a sample sized
    # as a fraction of original dataset and save it in the directory

    print("Introducing hard positives into dataset")
    introduce_new_examples(example_path, hard_positives, True)
    print("Introducing hard negatives into dataset")
    introduce_new_examples(example_path, hard_negatives, False)


if __name__ == '__main__':
    main()
