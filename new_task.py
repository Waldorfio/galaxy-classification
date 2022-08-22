# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:51:34 2019

@author: Walid
"""

from datetime import datetime
startTime = datetime.now()
import sys
try:
    import matplotlib.pyplot as plt
except:
    sys.exit('\tERROR: matplotlib could not be found for python3.\n\tPlease pip install matplotlib.')
import numpy as np
import shutil
import os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
INPUTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
image_width = 50 # The individual image pixel count. Assuming square shape
images = ['2df.dat']
labels = ['2dfn.dat']
root = 'C:\\Users\\Walid\\Downloads\\attempt'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def bash_cmd(string): # A function to run bash from windows
    os.system('bash -c {0}'.format(string)) 
    return

def change_dr1_para(galaxy,a,b,c):
    # Open files
    os.chdir(root)
    with open('dr1.para', 'r') as file:
        dr1 = file.readlines()
    file.close()
    # Make edits
    dr1[9] = '{} ### 1=eliptical 2=rectangular 3=rectangular + elliptical\n'.format(str(int(galaxy)))
    dr1[10] = '{}  ### the ratio of minor-to-major axis ratio (<1.0) for elliptical\n'.format(str(float(a)))
    dr1[11] = '{}  ### The length of the longer side of the rectangular (<1)\n'.format(str(float(b)))
    dr1[12] = '{}  ### The length of the shorter side of the rectangular (<1)\n'.format(str(float(c)))
    # Write out files
    with open('dr1.para', 'w+') as file:
        file.write(''.join(dr1))
    file.close()
    return

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RUNNING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
os.chdir(str(root))
print('\nCreating files...')


i = 1
images = []
labels = []

for a in np.linspace(0.4, 1.0, 36): # OVAL
    change_dr1_para('1', a, '0.8', '0.6')
    bash_cmd('./comp')
    bash_cmd('./dr1')
    shutil.copy2('2dfn.dat', '2dfn.dat.m{}'.format(i))
    bash_cmd('./dens')
    shutil.copy2('2df.dat', '2df.dat.m{}'.format(i))
    images.append('2df.dat.m{}'.format(i))
    labels.append('2dfn.dat.m{}'.format(i))
    print(str(i)+'/108')
    i = i + 1
for a in np.linspace(0.4, 0.8, 6): # RECTANGLE
    for b in np.linspace(0.4, 0.6, 6):
        change_dr1_para('2', a, b, '0.6')
        bash_cmd('./comp')
        bash_cmd('./dr1')
        shutil.copy2('2dfn.dat', '2dfn.dat.m{}'.format(i))
        bash_cmd('./dens')
        shutil.copy2('2df.dat', '2df.dat.m{}'.format(i))
        images.append('2df.dat.m{}'.format(i))
        labels.append('2dfn.dat.m{}'.format(i))
        print(str(i)+'/108')
        i = i + 1
for c in np.linspace(0.4, 0.8, 36): # OBROUND
    change_dr1_para('3', '0.8', '0.8', c)
    bash_cmd('./comp')
    bash_cmd('./dr1')
    shutil.copy2('2dfn.dat', '2dfn.dat.m{}'.format(i))
    bash_cmd('./dens')
    shutil.copy2('2df.dat', '2df.dat.m{}'.format(i))
    images.append('2df.dat.m{}'.format(i))
    labels.append('2dfn.dat.m{}'.format(i))
    print(str(i)+'/108')
    i = i + 1


os.chdir('C:\\Users\\Walid\\Dropbox\\Thesis (GENG5511)\\standalone scripts')
exec(open('combine_2df.py').read())

#os.chdir('C:\\Users\\Walid\\Dropbox\\Thesis (GENG5511)\\standalone scripts')
#exec(open('plot_images.py').read())
#os.chdir('C:\\Users\\Walid\\Dropbox\\Thesis (GENG5511)\\standalone scripts')
#exec(open('test20.py').read()) 
#os.chdir(str(root))


print((datetime.now() - startTime), ':\tFiles Created')