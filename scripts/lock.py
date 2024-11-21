#!/usr/bin/env python3

import os
import subprocess


if __name__ == '__main__':
    os.environ['SETUPTOOLS_SCM_PRETEND_VERSION'] = '0.0.0'
    subprocess.check_call('uv lock')  # noqa: S603, S607
