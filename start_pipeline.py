import sys
import getopt
from automol.pipeline import Pipeline


def main(argv):
    if not argv:
        print_usage()
        sys.exit()
    configfile = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["cfile="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-c", "--cfile"):
            configfile = arg
    if configfile:
        pipeline = Pipeline(configfile)
        pipeline.train()


def print_usage():
    print('python start_pipeline.py -c <configfile>')


if __name__ == "__main__":
    main(sys.argv[1:])
