# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:55:14 2019

@author: Elias
"""

import subprocess

subprocess.Popen('python asyn_distributed_keras.py --job_name "ps" --task_index 0', shell = True)
print ("ps");
subprocess.Popen('python asyn_distributed_keras.py --job_name "worker" --task_index 0', shell = True)
print ("worker");
subprocess.Popen('python asyn_distributed_keras.py --job_name "worker" --task_index 1', shell = True)
print ("worker 1");