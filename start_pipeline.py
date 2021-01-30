#!/usr/bin/python

import sys
import getopt
import os


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
            print('Config file is "{}"'.format(configfile))
    if configfile:
        print('Pipeline file exists: {}'.format(os.path.exists('automol/pipeline.py')))


def print_usage():
    print('start_pipeline.py -c <configfile>')


if __name__ == "__main__":
    main(sys.argv[1:])
