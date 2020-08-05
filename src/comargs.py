import argparse


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video to be processed")
    parser.add_argument("-s", "--scale", type=restricted_float,
                        help="scale factor", default=1)

    return parser.parse_args()


def hardN_examples():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video to be processed")
    parser.add_argument("-s", "--scale", type=restricted_float,
                        help="scale factor", default=1)

    return parser.parse_args()


def classifier_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset location")
    parser.add_argument("-a", "--architecture",
                        choices=["AlexNet", "FCN", "MNIST"],
                        default="FCN", help="model to use")
    parser.add_argument("-b", "--batch", type=int,
                        default=64, help="batch size")
    parser.add_argument("--summarize", action="store_true",
                        help="summarize model after getting")
    return parser.parse_args()


def tracker_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video to be processed")
    parser.add_argument("-s", "--scale", type=restricted_float,
                        help="scale factor", default=1)
    parser.add_argument("-m", "--model", help="model location")
    return parser.parse_args()


def hnmex_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model location")
    parser.add_argument("-d", "--data", help="dataset location")
    parser.add_argument("-r", "--refs", help="hard negative/positive sources")
    return parser.parse_args()


def hnmin_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model location")
    parser.add_argument("-d", "--data", help="dataset location")
    return parser.parse_args()


def retrain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model location")
    parser.add_argument("-r", "--refs", help="hard negative/positive sources",
                        default="hardNegatives/")
    parser.add_argument("-i", "--iterations", default=5, type=int,
                        help="number of iterations to be done")
    parser.add_argument("-d", "--data", default="dataset/",
                        help="dataset location (first iteration)")
    return parser.parse_args()


def query_yes_no(text, default=True):
    valid = {
        "yes": True, "y": True, "ye": True,
        "no": False, "n": False
    }
    if default is None:
        prompt = " [y/n] "
    elif default:
        prompt = " [Y/n] "
    elif not default:
        prompt = " [y/N] "
    else:
        raise ValueError("Default answer is invalid: '%s'" % default)

    while True:
        choice = input(text + prompt).lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                  "(or 'y' or 'n').\n")
