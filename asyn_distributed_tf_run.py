# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:14:16 2019

@author: Elias
"""

import subprocess

subprocess.Popen('python3 asyn_distributed_tf.py --job_name "ps" --task_index 0', shell = True)
subprocess.Popen('python3 asyn_distributed_tf.py --job_name "worker" --task_index 0', shell = True)
subprocess.Popen('python3 asyn_distributed_tf.py --job_name "worker" --task_index 1', shell = True)
subprocess.Popen('python3 asyn_distributed_tf.py --job_name "worker" --task_index 2', shell = True)