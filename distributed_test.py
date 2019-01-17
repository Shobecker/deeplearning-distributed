# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:18:47 2019

@author: Elias
"""

import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)  # Create a session on the server.
sess.run(c)