#!/usr/bin/env python3
import os
import pathlib

script = pathlib.Path(__file__).with_name("train_pi05_multihost.sh")
os.execv("/bin/bash", ["bash", str(script)])
