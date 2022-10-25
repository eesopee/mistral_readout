#!/usr/bin/pythonimport valon_synth9 as valon_synth This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) May 23, 2016  Gordon, Sam <sbgordo1@asu.edu Author: Gordon, Sam <sbgordo1@asu.edu>
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib, time, struct
import numpy as np
import shutil
np.set_printoptions(threshold=np.nan)
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import casperfpga 
import corr
from myQdr import Qdr as myQdr
import types
import logging
#import threading
import glob  
import os
import sys
import valon_synth9 as valon_synth
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
from socket import *
from scipy import signal
#import find_kids_olimpo as fk 
import find_kids_mistral_new as fk
import subprocess
#from olimpo_pipeline import pipeline
from mistral_pipeline_dev import pipeline
import tqdm
import configuration

'''
Function used for tuning. It spans a small range of frequencies around known resonances.
'''

target_path = configuration.setupdir+'sweeps/target/' 
center_freq = configuration.LO*1.e6

if path_current: #dobbiamo dargli in ingresso current o no
	vna_path = self.setupdir+'sweeps/vna/current'
	sweep_dir = self.timestring
else:
    print "roach415: /home/mistral/src/parameters/roach415"  
    print "current: /data/mistral/setup/kids/sweeps/target/current"
    print "roach_test: /home/mistral/src/parameters/roach_test"
    print "roach_test_destination: /data/mistral/setup/kids/sweeps/target"
    vna_path = raw_input('Absolute path to VNA sweep dir ? ')
    
    self.timestring = "%04d%02d%02d_%02d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5])            
    self.add_out_of_res(vna_path + "/target_freqs.dat", self.out_of_res_tones)       
    self.calc_transfunc(vna_path + "/target_freqs.dat")

    sweep_dir = raw_input('Insert new target sweep subdir to '+self.setupdir+ '/sweeps/target/ (eg. '+self.timestring+') Press enter for defualt:')
    if(sweep_dir)=='': sweep_dir=self.timestring
try:
	self.target_freqs = np.load(vna_path + '/target_freqs_new.npy')
except IOError:
	self.target_freqs, self.amps = np.loadtxt(os.path.join(vna_path, 'target_freqs.dat'), unpack=True)



save_path = os.path.join(target_path, sweep_dir)
self.path_configuration = save_path
#self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/2.))*1.0e6	

try:
	os.mkdir(save_path)
except OSError:
        pass
command_cleanlink = "rm -f "+target_path+'current'
os.system(command_cleanlink)
command_linkfile = "ln -f -s " + save_path +" "+ target_path+'current'
os.system(command_linkfile)
np.save(save_path + '/target_freqs.npy', self.target_freqs)
self.bb_target_freqs = ((self.target_freqs*1.0e6) - center_freq)
upconvert = (self.bb_target_freqs + center_freq)/1.0e6
print "RF tones =", upconvert
self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO
print '\nTarget baseband freqs (MHz) =', self.bb_target_freqs/1.0e6
span =self.sweep_span #200.0e3   #era 400.e3             # era 1000.e3 #era 400.e3 20170803
start = center_freq - (span)  # era (span/2)
stop = center_freq + (span)   # era (span/2) 
step = self.sweep_step #1.25e3 * 2.                 # era senza   
sweep_freqs = np.arange(start, stop, step)
sweep_freqs = np.round(sweep_freqs/step)*step
print "LO freqs =", sweep_freqs
np.save(save_path + '/bb_freqs.npy',self.bb_target_freqs)
#	np.save(vna_path  + '/bb_freqs.npy',self.bb_target_freqs)
np.save(save_path + '/sweep_freqs.npy',sweep_freqs)
if write:
	if self.do_transf == True:
	#self.writeQDR(self.bb_target_freqs)
		self.writeQDR(self.bb_target_freqs, transfunc=True)
	else:
		self.writeQDR(self.bb_target_freqs)
if sweep:

    for freq in tqdm.tqdm(sweep_freqs):
	if self.v1.set_frequency(2, freq/1.0e6, 0.01):
            self.store_UDP(100,freq,save_path,channels=len(self.bb_target_freqs)) 
            self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO

print "pipline.py su "+ save_path

pipeline(save_path) #locate centers, rotations, and resonances
self.path_configuration = save_path

self.array_configuration()#includes make_format	

if do_plot:
	self.plot_targ(save_path)
return
