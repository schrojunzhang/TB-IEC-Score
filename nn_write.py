#!usr/bin/env python3
# -*- coding:utf-8 -*-
# used for collect nnscore result
import os
import pandas as pd
import sys

if __name__ == '__main__':
    log_file = sys.argv[1]
    csv_file = sys.argv[2]
    with open(log_file, 'r')as f:
        lines = f.readlines()
    result = eval(lines[0].strip())
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    if os.path.exists(log_file):
        os.remove(log_file)
