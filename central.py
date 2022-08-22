# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:44:39 2019

@author: Walid
"""
from datetime import datetime
startTime = datetime.now()
import os
import shutil
import sys
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    sys.exit('\tERROR: matplotlib could not be found for python3.\n\tPlease pip install matplotlib.')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
USER INPUTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
image_width = 50
steps = 99 # 6 was default


files_to_plot = ['train.dat', '2df_step1.dat', '2df_step2.dat']
current_dir = 'C:\\SSD Folder\\kenji original - Copy (5)'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CODE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
os.chdir(current_dir)
loop = True # Initialising
while loop == True:
    code_check = input('Re-calculate? (y/n): ')
    if code_check == 'y':
        def bash_cmd(string): # A function to run bash from windows
            os.system('bash -c {0}'.format(string)) 
            return

        
        " Step 1 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"
        bash_cmd('gfortran dr.f -o dr')
        bash_cmd('gfortran dens.f -o dens')
        # MAKE dr.para EDITS HERE
        bash_cmd('./dr') # Generates the positions of particles (dr.dat)
        bash_cmd('./dens') # Converts particle positions (dr.dat) to image (2df.dat)
        shutil.copy2('2df.dat', '2df_step1.dat') # Creating copy as will be overiten in step 2
        

        " Step 2 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"
        bash_cmd('gfortran dini.f -o dini') # Producing the files for runre1
        bash_cmd('gfortran gini.f -o gini')
        bash_cmd('gfortran gpar.f -o gpar')
        bash_cmd('gfortran gint1.f -o gint1')
        
        bash_cmd('./runre1') # Produces tout.dat and other files (tout.dat....)
#        bash_cmd('./runre2') # Produces tout.dat and other files (tout.dat....)
        
        bash_cmd('gfortran ddat.f -o ddat')
#        np.savetxt('ddat.para', np.array((steps,)), fmt='%i') # ddat.para edit (Defining number of steps)
        bash_cmd('./ddat') # Produces dr.dat
        bash_cmd('./dens') # Converts particle positions (dr.dat) to image (2df.dat)
        shutil.copy2('2df.dat', '2df_step2.dat')
        
        loop = False
        
    elif code_check == 'n':
        loop = False
    
    elif code_check != 'y' or code_check != 'n':
        print('\tERROR: Invalid input.')
        loop = True

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PLOTTING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('\nPlotting...\n')
# Delete Folder if Already EXISTS
if os.path.exists('{}\\images'.format(current_dir)):
    shutil.rmtree('{}\\images'.format(current_dir))
os.makedirs('{}\\images'.format(current_dir))

# Create images/respective folders
for file in files_to_plot:
    original = np.loadtxt(file)
    os.makedirs('{}\\images\\{}'.format(current_dir, file.replace('.', '_')))
    os.chdir('{}\\images\\{}'.format(current_dir, file.replace('.', '_')))
    for i in range(1, int(len(original) / (image_width**2))):
        individual = original[((i-1)*(image_width**2)):(i*(image_width**2))]
        individual = np.resize(individual,(image_width,image_width))
        plt.imsave('image{}.png'.format(str(i)), individual)
    os.chdir('..')
    os.chdir('..')
os.chdir('..')

print((datetime.now() - startTime), ':\tTotal Time')