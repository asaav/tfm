import os
import re
import sys
import random
import math
import json
from ast import literal_eval as make_tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from comargs import hnmex_args

# Print only tensorflow error and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

DIST_THRESHOLD = 20
FIELD_SIZE = 56
drawing = False
np.random.seed(42)
random.seed(42)
included = {'yes': [], 'no': []}
found = {'yes': [], 'no': []}


def output2original(xout, yout):
    x_ = xout * 4
    y_ = yout * 4
    return x_, y_


def distance(rect1, rect2):
    (x, y) = rect1[0]
    (x_end, y_end) = rect1[1]
    (x1, y1) = rect2[0]
    (x1_end, y1_end) = rect2[1]

    # Compute rectangle centroids and calculate distance between them
    center_r1 = (x + x_end/2, y + y_end/2)
    center_r2 = (x1 + x1_end/2, y1 + y1_end/2)

    dist = math.sqrt((center_r2[0] - center_r1[0])**2 +
                     (center_r2[0] - center_r1[0])**2)
    return dist


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
    # Get 56x56 example that contains ball with random offset
    start_pos, end_pos = ball_r
    w = end_pos[0] - start_pos[0]
    h = end_pos[1] - start_pos[1]

    x_offset = random.randint(0, FIELD_SIZE-w)
    y_offset = random.randint(0, FIELD_SIZE-h)
    x = start_pos[0] - x_offset
    y = start_pos[1] - y_offset
    return x, y


def selecintr_new_examples(src_path, dst_path, it, examples, positives):
    n_examples = 0
    folders = ["train", "test", "val"]
    end_path = "yes" if positives else "no"
    for folder in folders:
        n_examples += len(os.listdir(os.path.join(src_path,
                                                  folder, end_path)))

    indexes = range(0, len(examples))

    # Choose n random examples from list with n being min of the original
    # count or the full list
    n = min(int(n_examples*.5), len(examples))
    chosen_idx = np.random.choice(indexes, n, replace=False)

    # Mark examples as included to avoid selecting them in future iterations
    subset = included[end_path]
    for idx in chosen_idx:
        elem = found[end_path][idx]
        if positives:
            subset.append(elem)
        else:
            subset.append(elem)

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
    print("Saving examples in " + dst_path)

    picks = [train_ex, test_ex, val_ex]

    for subdivision, picked in zip(folders, picks):
        for elem in picked:
            filename = os.path.join(
                dst_path, subdivision,
                end_path, 'it' + it + '_h' + str(elem)+'.png'
            )
            save_image(examples[elem], filename)

    # Make symlinks of the original examples in the datasets leaving out
    # as many examples as we introduced previously

    for folder, pick in zip(folders, picks):
        print("Creating symbolic links for the original", folder,
              "examples, leaving out", len(pick))
        create_symlinks(os.path.join(src_path, folder, end_path),
                        os.path.join(dst_path, folder, end_path),
                        len(pick))


def is_used(element, positive):
    frame = element['frame']
    if positive and frame in found['yes']:
        # Hard positives are a on-in-frame occurrence,
        # so if the frame was included, we can be sure without checking boxes
        return True
    elif not positive:
        # Check every form found, if any of them is within certain
        # distance of our element, we consider it as used
        for patch in found['no']:
            if (distance(element['position'], patch['position'])
                    < DIST_THRESHOLD):
                return True
    return False


def get_hard_examples(model):
    hard_positives = []
    hard_negatives = []
    for filename in os.listdir("hardNegatives"):

        # Get ball positions from filename
        regex = re.compile(r'(\d+)-(\d+,\d+)-(\d+,\d+)\D+$')
        match = re.match(regex, filename)
        frame = match.group(1)
        start_pos = make_tuple(match.group(2))
        end_pos = make_tuple(match.group(3))
        example = cv2.imread(os.path.join("hardNegatives", filename))

        # Convert input image to RGB since model works in that color space
        # and get prediction for image
        img_rgb = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
        output = model.predict(np.expand_dims(img_rgb, axis=0))
        output = np.squeeze(output, axis=0)

        # Get contours out of positve blobs in model output
        output = (output * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        # Filter contours (positve blobs) by size
        contours = [c for c in contours if cv2.contourArea(c) > 5]
        ball_is_shape = False
        for i, c in enumerate(contours):
            # Compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # if (cX, cY) input patch doesn't contain the ball rectangle
            # in input image, it's a false positive
            oX, oY = output2original(cX, cY)
            original_rect = ((oX, oY), (oX + FIELD_SIZE, oY + FIELD_SIZE))
            ball_r = (start_pos, end_pos)
            element = {"frame": frame, "position": original_rect}
            if (not rectangleContains(original_rect, ball_r)
                    and not is_used(element, positive=False)):
                # Add example to hard negatives list
                h_neg = img_rgb[oY:oY+FIELD_SIZE, oX:oX+FIELD_SIZE]
                found['no'].append(element)
                hard_negatives.append(h_neg)
            elif rectangleContains(original_rect, ball_r):
                # If any rectangle contains the ball,
                # do not include this frame as hard positive
                ball_is_shape = True
        if not ball_is_shape and not is_used(element, positive=True):
            x, y = get_ball_rectangle(ball_r)
            h_pos = img_rgb[y:y+FIELD_SIZE, x:x+FIELD_SIZE]
            found['yes'].append(element)
            hard_positives.append(h_pos)
    return hard_negatives, hard_positives


def example_extraction(model_path, dataset, referenceImages):
    global found, included
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model " + model_path + " successfully loaded")

    it_regex = r'it(\d+)(.+?)\.'
    match = re.match(it_regex, os.path.basename(model_path))
    iteration = match.group(1)
    name = match.group(2)

    # Create directories for hard negative examples, model and iteration
    # in case they dont already exist
    if not os.path.exists("hnmData"):
        os.mkdir("hnmData")
    if not os.path.exists(os.path.join("hnmData", name)):
        os.mkdir(os.path.join("hnmData", name))
    example_path = os.path.join("hnmData", name, 'it' + iteration)
    json_path = os.path.join(example_path, "included.json")

    if not os.path.exists(example_path):
        print("Creating " + example_path + " folder.")
        os.mkdir(example_path)
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            included = json.load(f)
            found = included

    print("Calculating hard negatives and positives...")
    hard_negatives, hard_positives = get_hard_examples(model)

    # Once we have every example, pick a sample sized
    # as a fraction of original dataset and save it in the directory

    print("Introducing hard negatives into dataset")
    selecintr_new_examples(dataset, example_path,
                           iteration, hard_negatives, False)
    print("Introducing hard positives into dataset")
    selecintr_new_examples(dataset, example_path,
                           iteration, hard_positives, True)

    # Dump included dict to json for future iterations
    with open(json_path, "w") as f:
        json.dump(included, f)


def main():
    args = hnmex_args()
    # Check directories passed by argument
    if not os.path.exists(args.model):
        sys.exit("Model " + args.model + " does not exist")
    if not os.path.exists(args.refs):
        sys.exit("Path " + args.refs + " directory does not exist")
    if not os.path.exists(args.data):
        sys.exit("Path " + args.data + " directory does not exist")

    example_extraction(args.model, args.data, args.refs)


if __name__ == '__main__':
    main()
