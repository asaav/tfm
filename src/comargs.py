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

def classifier_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "dataset", help="dataset location")
    parser.add_argument( "-a", "--architecture", choices=["AlexNet", "FCN", "MNIST"],
                        default="FCN" ,help="model to use")
    parser.add_argument( "-i", "--input", type=int,
                        default=56 ,help="input shape (assummes square image)")
    parser.add_argument( "-b", "--batch", type=int,
                        default=64 ,help="batch size")
    parser.add_argument("--summarize", help="summarize model after getting")  
    return parser.parse_args()                      

def query_yes_no(text, default=True):
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
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
            print("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
