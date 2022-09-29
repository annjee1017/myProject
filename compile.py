# python compile.py -i main_master

import py_compile
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_filename", required=True,	help="filename for compile")
args = vars(ap.parse_args())

py_compile.compile(f"{(args['input_filename'])}.py", cfile=f"{(args['input_filename'])}.pyc")
