import argparse


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video to be processed")
    parser.add_argument("-s", "--scale", type=restricted_float, help="scale factor", default=1)

    return parser.parse_args()