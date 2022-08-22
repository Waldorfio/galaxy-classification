# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:44:39 2019

@author: Walid
"""
# PLEASE USE tensorflow v1.12.0. Other modules can be latest (wont matter)
from datetime import datetime
startTime = datetime.now()
import os
import shutil
import sys
import numpy as np
import keras
from glob import glob
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
import keras.callbacks
import os.path
import math
try:
    import matplotlib.pyplot as plt
except:
    sys.exit('\tERROR: matplotlib could not be found for python3.\n\tPlease pip install matplotlib.')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
USER INPUTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n_mesh = 50
total_steps = 5 # ddat.para number
epochs = 30
train_dat = 'train.dat'
train_labels = 'ntrain.dat'

root_dir = 'C:\\SSD Folder\\kenji original - Copy (3)'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def bash_cmd(string): # A function to run bash from windows
    os.system('bash -c {0}'.format(string)) 
    return

def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

def redefine_max_steps(total_steps):
    # Open files
    os.chdir(root_dir)
    with open('ddat.para', 'r') as file:
        a = file.readlines()
    file.close()
    with open('time.para', 'r') as file:
        b = file.readlines()
    file.close()
    # Make edits
    a = str(total_steps)
    b = '{}. 0.01 100 10  ### T_end,dt,nout,ntmp\n'.format(total_steps - 1)
    # Write out files
    with open('ddat.para', 'w+') as file:
        file.write(''.join(a))
    file.close()
    with open('time.para', 'w+') as file:
        file.write(''.join(b))
    file.close()
    return

def define_current_step(step_number):
    # Open file
    os.chdir(root_dir)
    with open('sfpt.para', 'r') as file:
        a = file.readlines()
    file.close()
    # Make Edits
    a[5] = '{} ### Favorite time step 1\n'.format(step_number)
    # Write out file
    with open('sfpt.para', 'w+') as file:
        file.write(''.join(a))
    file.close()
    return

def custom_gpar_edits(v_y, theta, phi): # Kenjis request to look at a variety of different models
    # Open file
    os.chdir(root_dir)
    with open('gpar.para', 'r') as file:
        a = file.readlines()
    file.close()
    # Make Edits
    a[1] = '0.0 {} 0.0\n'.format(float(v_y))
    a[2] = '{} {}\n'.format(float(theta),float(phi))
    # Write out file
    with open('gpar.para', 'w+') as file:
        file.write(''.join(a))
    file.close()
    return

def name_files():
    files_to_plot = []
    files_to_plot.append('train.dat')
    for files in glob('2df.dat.m*'):
        files_to_plot.append(str(files))
    return files_to_plot

def test20_py(images_fname, labels_fname, img_dir):
    batch_size = 200
    num_classes = 3
    nb_epoch = epochs
    img_rows, img_cols = n_mesh, n_mesh
    n_mesh2 = n_mesh * n_mesh - 1
    n_mesh3 = n_mesh * n_mesh
    input_shape = (img_rows, img_cols, 1)
    os.chdir(img_dir) # Change into the image directory to open relevant .dat file and labels
    file1_open = np.loadtxt(images_fname)
    nmodel = int(len(file1_open) / (n_mesh**2)) # Calculate the number of models in the 2df.dat file automatically
    with open(images_fname, 'r') as file:
        images = file.readlines()
    file.close()
    with open(labels_fname, 'r') as file:
        labels = file.readlines()
    file.close()
    os.chdir(root_dir) # Return back to the location of this script
    x_train = np.zeros((nmodel, n_mesh3))
    x_test = np.zeros((nmodel, n_mesh3))
    y_train = np.zeros(nmodel, dtype=np.int)
    y_test = np.zeros(nmodel, dtype=np.int)
    # For 2D density map data
    ibin = 0
    jbin = -1
    for num, j in enumerate(images):
        jbin = jbin + 1
        tm = j.strip().split()
        x_train[ibin,jbin] = float(tm[0])
        x_test[ibin,jbin] = float(tm[0])
        if jbin == n_mesh2:
            ibin += 1
            jbin =- 1
    # For morphological map
    ibin = 0
    for num, j in enumerate(labels):
        tm = j.strip().split()
        y_train[ibin] = int(tm[0])-1
        y_test[ibin] = int(tm[0])-1
        ibin += 1
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)
    #stop
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate = 0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('model.h5')
    return

def test21_py(images_fname, labels_fname, img_dir, m_num):
    img_rows, img_cols = n_mesh, n_mesh
    n_mesh2 = n_mesh * n_mesh - 1
    n_mesh3 = n_mesh * n_mesh
    os.chdir(img_dir) # Change into the image directory to open relevant .dat file and labels
    file1_open = np.loadtxt(images_fname)
    nmodel = int(len(file1_open) / (n_mesh**2)) # Calculate the number of models in the 2df.dat file automatically
    with open(images_fname, 'r') as file:
        images = file.readlines()
    file.close()
    with open(labels_fname, 'r') as file:
        labels = file.readlines()
    file.close()
    labels = np.asarray(labels, dtype=np.int64)
    os.chdir(root_dir) # Return back to the location of this script
    # load weights into new model and json file
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    os.chdir(img_dir)
    # For output the galaxy classification results
    f1 = open('test21.out', 'w+')
    x_train = np.zeros((nmodel, n_mesh3))
    x_test = np.zeros((nmodel, n_mesh3))
    y_train = np.zeros(nmodel, dtype=np.int)
    y_test = np.zeros(nmodel, dtype=np.int)
    # For 2D density map data
    ibin = 0
    jbin = -1
    for num, j in enumerate(images):
        jbin = jbin + 1
        tm = j.strip().split()
        x_train[ibin, jbin] = float(tm[0])
        x_test[ibin, jbin] = float(tm[0])
        if jbin == n_mesh2:
            ibin += 1
            jbin =- 1
    ntest = ibin
#    print(ntest)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_vec = np.zeros(3)
    y_pred = loaded_model.predict(x_test)
    f1.write(str(ntest)+'\n')
    result_array = []
    for i in range(ntest):
        for j in range(3):
            y_vec[j] = y_pred[i, j]
        y_type = np.argmax(y_vec)
        y_type_actual = y_type + 1 # This is because python indexing starts at 0, so our actual label number is + 1 this
        prob = y_vec[y_type]
#        print('i='+str(i)+'\tG-type='+str(y_type_actual)+'\tP='+str(prob))
        results = str(y_type_actual)+' '+str(y_vec[0])+' '+str(y_vec[1])+' '+str(y_vec[2])+'\n'
        f1.write(results)
        result_array.append(y_type_actual)
    result_array = np.asarray(result_array)
    comparison = np.vstack((labels, result_array)).T
    " Post-Processing Data''''''''''''''''''''''''''''''''''''''''''''''''''''"
    # Save the comparison results in an easy to read format
    os.chdir(img_dir)
    if (comparison[:,0]==comparison[:,1]).all() == True:
        print('\tTested {} => Acc. = 100.0%'.format(images_fname))
    elif (comparison[:,0]==comparison[:,1]).all() == False:
        # Finding percentage accuracy (and printing it)
        new_col = np.ones((len(comparison),1))
        for i in range(0, len(comparison)):
            new_col[i] = comparison[i,0] - comparison[i,1]
        new_col = np.asarray(new_col)
        new_comp = np.hstack((comparison, new_col))
        accuracy = 100 - ((np.count_nonzero(new_comp[:,2])/(len(new_comp))) * 100)
        accuracy = normal_round(accuracy, 2)
        # Saving result in a text file
        print('\tTested {} => Acc. = {}%'.format(images_fname,accuracy))
        
    f1.close()
    if m_num != 0:
        shutil.copy2('test21.out', 'test21.out.m{}'.format(m_num))
        shutil.copy2(str(root_dir)+'\\gpar.para.m'+str(m_num), 'gpar.para.m{}'.format(m_num))
        os.remove('test21.out')
        
    os.chdir(root_dir)
    return

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CODE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
os.chdir(root_dir)

# Run Initial Functions
redefine_max_steps(total_steps)

loop = True # Initialising
while loop == True:
    code_check = input('1. Produce Files? (y/n): ')
    if code_check == 'y':
        
        " Step 0: Running ./comp''''''''''''''''''''''''''''''''''''''''''''''"
        bash_cmd('./comp')
        
        " Step 2: Generating tout.dat as images'''''''''''''''''''''''''''''''"
        i = 1
        for v_y in (0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0):
            for theta in (0,30,60,90,120,150,180):
                for phi in (0,30,60,90,120,150,180,210,240,270,300,330,360):
                    custom_gpar_edits(v_y, theta, phi)
                    bash_cmd('./runre1') # Produces tout.dat and other files
                    define_current_step(1) # sfpt.para edit (Defining current of steps)
                    bash_cmd('./ddat') # Produces dr.dat
                    bash_cmd('./dens') # Converts particle positions (dr.dat) to image (2df.dat)
                    
                    shutil.copy2('2df.dat', '2df.dat.m{}'.format(i))
                    shutil.copy2('2dfn.dat', 'n2df.dat.m{}'.format(i))
                    shutil.copy2('gpar.para', 'gpar.para.m{}'.format(i))
                    os.remove('dr.dat')
                    os.remove('2df.dat')
                    os.remove('2dfn.dat')
                    
                    print('Model {} files produced.'.format(i))
                    i = i + 1 # Defines the model number

        
    # Finish the Loop
        loop = False
    elif code_check == 'n':
        loop = False
    elif code_check != 'y' or code_check != 'n':
        print('\tERROR: Invalid input.')
        loop = True

print((datetime.now() - startTime), ':\tProducing Files')
startTime2 = datetime.now()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
SAVING PLOTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('\nPlotting...')
files_to_plot = name_files()

# Delete Folder if Already EXISTS
if os.path.exists('{}\\images'.format(root_dir)):
    shutil.rmtree('{}\\images'.format(root_dir))
check = False
while check == False:
    try:
        os.makedirs('{}\\images'.format(root_dir))
        check = True
    except:
        os.chdir(root_dir)
        input('\tERROR: Please move out of the images directory, and close any images.\n\tPress any key to continue ...')
        check = False

# Create images/respective folders
file_dirs = []
for file in files_to_plot:
    original = np.loadtxt(file)
    dirname = '{}\\images\\{}'.format(root_dir,file.replace('.', '_'))
    os.makedirs(dirname)
    os.chdir(dirname)
    file_dirs.append(dirname)
    shutil.copy2('{}\\{}'.format(root_dir,file), '{}\\{}'.format(os.getcwd(),file))
    shutil.copy2('{}\\n{}'.format(root_dir,file), '{}\\n{}'.format(os.getcwd(),file))
    for i in range(1, int(len(original) / (n_mesh**2))):
        individual = original[((i-1)*(n_mesh**2)):(i*(n_mesh**2))]
        individual = np.resize(individual,(n_mesh,n_mesh))
        plt.imsave('image{}.png'.format(str(i)), individual)
    os.chdir('..')
    os.chdir('..')
os.chdir('..')

print((datetime.now() - startTime2), ':\tPlotting')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAINING MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
loop = True # Initialising
while loop == True:
    train_check = input('2. Train Model? (y/n): ')
    if train_check == 'y':
        test20_py(train_dat, train_labels, str(root_dir)+'\\images\\train_dat')
    # Finish the Loop
        loop = False
    elif train_check == 'n':
        loop = False
    elif train_check != 'y' or train_check != 'n':
        print('\tERROR: Invalid input.')
        loop = True

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
loop = True # Initialising
while loop == True:
    test_check = input('3. Test Model? (y/n): ')
    if test_check == 'y':
        i = 0
        for file in files_to_plot:
            try:
                m_num = file.split('m')[1]
            except IndexError:
                m_num = 0
            test21_py(file, 'n'+file, file_dirs[i], m_num)
            i = i + 1
    # Finish the Loop
        loop = False
    elif test_check == 'n':
        loop = False
    elif test_check != 'y' or test_check != 'n':
        print('\tERROR: Invalid input.')
        loop = True

print((datetime.now() - startTime), ':\tTotal Time')