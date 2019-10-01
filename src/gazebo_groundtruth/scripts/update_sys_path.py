#!/usr/bin/python3.6

import sys, os

rospy_install = os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'rospy_install', 'devel','lib','python3.6','dist-packages')
paths = sys.path 

#if paths[0] != rospy_install:
#    sys.path.insert(0,rospy_install)

