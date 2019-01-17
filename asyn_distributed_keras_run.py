# -*- coding: utf-8 -*-
"""

@author: Elias
"""

import subprocess

subprocess.Popen('python asyn_distributed_keras.py --job_name "ps" --task_index 0', shell = True)
print ("ps");
subprocess.Popen('python asyn_distributed_keras.py --job_name "worker" --task_index 0', shell = True)
print ("worker");
subprocess.Popen('python asyn_distributed_keras.py --job_name "worker" --task_index 1', shell = True)
print ("worker 1");