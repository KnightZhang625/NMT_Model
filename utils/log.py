# coding: utf-8

from __future__ import print_function

def log_info(message):
    print('\033[1;34m Info : {} \033[0m'.format(message))

def log_error(message):
    print('\033[1;31m Error : {} \033[0m'.format(message))

