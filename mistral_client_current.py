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
from pathlib import Path

import configuration as conf


class roachInterface(object):

    def __init__(self):

        self.do_transf = False #provvisorio! Da sistemare.
        self.test_comb_flag = False

        '''	
        Filename
        '''

        self.timestring = "%04d%02d%02d_%02d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5])
        self.today = "%04d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2])

        string = 'ROACH dirfile_'+self.timestring

        sys.stdout.write(string+'\n')
        sys.stdout.flush()

        '''
        Important paths (all defined in the configuration.py file)
        '''

        self.folder_frequencies=conf.folder_frequencies
        self.datadir = conf.datadir
        self.setupdir = conf.setupdir
        self.transfer_function_file = conf.transfer_function_file
        self.path_configuration = conf.path_configuration

        '''
        Setting up socket
        '''

        sys.stdout.write("Setting up socket\n")
        self.s = socket(AF_PACKET, SOCK_RAW, htons(3))
        self.s.setsockopt(SOL_SOCKET, SO_RCVBUF, 8192 + 42)
        self.s.bind((conf.data_socket, 3)) #must be the NET interface. NOT PPC INTERFACE! Do not change the 3.

        '''
        Now we set up MISTRAL roach.
        '''

        self.divconst = 1. #not really necessary anymore since we only use MISTRAL and not olimpo, that needed double LO (=2 for OLIMPO)
        self.ip = conf.roach_ip
        self.center_freq = conf.LO
        self.global_attenuation= 1.0 #need to implement the class for Arduino variable attenuator!
        self.wavemax = 1.1543e-5 #what is this? Maybe we should increase it for 400 tones.

        '''
        VALON SETUP
        '''
        print("Setting up VALON")

        '''
        For some god forsaken reason, the VALON changes serial port. Here we cycle through the serial ports until it connects and sets the LO frequency correctly.
        '''
        
        # connecting to valon and variable attenuator
        self.connect_to_valon()
        self.connect_to_arduino()

        #self.nchan = len(self.raw_chan[0])

        sys.stdout.write("Activating ROACH interface\n")
        self.fpga = casperfpga.katcp_fpga.KatcpFpga(self.ip,timeout=120.)
        self.bitstream = "mistral_firmware_current.fpg"   # October 2017 from Sam Gordon
        self.dds_shift = 318 # was 305  # this is a number specific to the firmware used. It will change with a new firmware. which number for MISTRAL?

        self.dac_samp_freq = 512.0e6
        self.fpga_samp_freq = 256.0e6

        self.do_double = False #A che serve?

        self.LUTbuffer_len = 2**21
        self.dac_freq_res = 0.5*(self.dac_samp_freq/self.LUTbuffer_len)
        self.fft_len = 1024

        '''
        SAMPLING FREQUENCY IS SET HERE
        '''
        self.accum_len = (2**20) # For 244 Hz sampling
        #self.accum_len = (2**21) # For 122.07 Hz sampling
        #self.accum_len = (2**22) # For 61.014 Hz sampling

        self.accum_freq = self.fpga_samp_freq/(self.accum_len - 1)
        np.random.seed(23578)
        self.phases = np.random.uniform(0., 2.*np.pi, 2000)   #### ATTENZIONE PHASE A pi (ORA NO)


    def connect_to_valon(self):
        '''
        Function that allows the connection with the VALON by attempting to 
        connect to 10 differents USB ports

        Returns
        -------
        None.

        '''
        
        for i in range(10):
            time.sleep(0.5)
            try:
                self.valon_port = Path("/dev/") / "ttyUSB{:d}".format(i)
                sys.stdout.write("Attempting Valon connection on port "+self.valon_port.as_posix()+"\n")
                self.v1 = valon_synth.Synthesizer(self.valon_port.as_posix())
                self.v1.set_frequency(2,self.center_freq,0.01)
                self.v1.set_frequency(1, 512.,0.01)
                sys.stdout.write("Success!\n")
                sys.stdout.write("Valon connected at port "+self.valon_port.as_posix()+"\n")
                break

            except OSError:
                if i == 9:
                    print("too many failed attempts")
                    break
                
                time.sleep(0.5)
                sys.stdout.write("Failed. Attempting next port...")
                        
                pass


    def connect_to_arduino(self):
        '''
        Function that allows the connection with the Arduino attenuators by 
        attempting to connect to 10 differents ACM ports

        Returns
        -------
        None.

        '''
        
        for i in range(10):
            time.sleep(0.5)
            try:
                self.arduino_port = Path("/dev/") / "ttyACM{:d}".format(i)
                self.att = vatt.Attenuator(self.arduino_port.as_posix())
                atts = self.att.get_att()
                print("Current attenuation = ", atts)
                sys.stdout.write("Success!\n")
                sys.stdout.write("Attenuator connected at port "+self.arduino_port+"\n")
                sys.stdout.write("Setting default attenuations")
                att.set_att(1, conf.att_RFOUT)
                att.set_att(2, conf.att_RFIN)
                sys.stdout.write("(RF_OUT, RF_IN) = " + str(att.get_att()))
                break
           
            except OSError:
                if i == 9:
                    print("too many failed attempts")
                    break
                
                time.sleep(0.5)
                sys.stdout.write("Failed. Attempting next port...")
                        
                pass
            
            
            
            
            
            
            

    def array_configuration(self):	
#	self.path_configuration = '/home/data/olimpo/setup/kids/sweeps/target/current/'

        sys.stdout.write("setting freqs, centers, radii and rotations from %s \n " %self.path_configuration)
        try:
                self.cold_array_rf = np.load(self.path_configuration+'/target_freqs_new.npy')
        except IOError:
                self.cold_array_rf = np.load(self.path_configuration+'/target_freqs.npy')
        self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/self.divconst))*1.0e6
        self.centers = np.load(self.path_configuration+'/centers.npy')
        self.rotations = np.load(self.path_configuration+'/rotations.npy')
        self.radii = np.load(self.path_configuration+'/radii.npy')
        sys.stdout.write( "reading freqs, centers, rotations and radii from %s\n" %self.path_configuration)
        print 'radii', self.radii

        self.make_format(path_current = True)

    def lpf(self, zeros):
        zeros *=(2**31 - 1)
        for i in range(len(zeros)/2 + 1):
            coeff = np.binary_repr(int(zeros[i]), 32)
            coeff = int(coeff, 2)
            print 'h' + str(i), coeff       
            self.fpga.write_int('FIR_h'+str(i),coeff)
        return 

    def upload_fpg(self):
        print 'Connecting to ROACHII on ',self.ip,'...'
        t1 = time.time()
        timeout = 10
        while not self.fpga.is_connected():
                if (time.time()-t1) > timeout:
                    raise Exception("Connection timeout to roach")
        time.sleep(0.1)
        if (self.fpga.is_connected() == True):
            print 'Connection established'
            self.fpga.upload_to_ram_and_program(self.bitstream)
        else:
                print 'Not connected to the FPGA'
        print 'Uploaded', self.bitstream
        time.sleep(3)
        return

    def qdrCal(self):    
    # Calibrates the QDRs. Run after writing to QDR.      
        self.fpga.write_int('dac_reset',1)
        print 'DAC on'
        bFailHard = False
        calVerbosity = 1
        qdrMemName = 'qdr0_memory'
        qdrNames = ['qdr0_memory','qdr1_memory']
        print 'Fpga Clock Rate =',self.fpga.estimate_fpga_clock()
        self.fpga.get_system_information()
        results = {}
        for qdr in self.fpga.qdrs:
            print qdr
            mqdr = myQdr.from_qdr(qdr)
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
        print 'qdr cal results:',results
        for qdrName in ['qdr0','qdr1']:
            if not results[qdr.name]:
                print 'Calibration Failed'
                break

    # calibrates QDR and initializes GbE block
    def initialize(self):
        self.upload_fpg()
        #do we want to upload the firmware each time?

        '''
        This is the destination IP of the packets.
        '''

        self.dest_ip = 192*(2**24) + 168*(2**16) + 40*(2**8) + 2
        self.fabric_port=60000

        '''
        Setting up the Gigabit Ethernet interface
        '''

        self.fpga.write_int('GbE_tx_destip',self.dest_ip)
        self.fpga.write_int('GbE_tx_destport',self.fabric_port)
        self.fpga.write_int('downsamp_sync_accum_len', self.accum_len - 1)
        self.accum_freq = self.fpga_samp_freq / self.accum_len # FPGA clock freq / accumulation length    
        self.fpga.write_int('PFB_fft_shift', 2**9 -1)
        self.fpga.write_int('dds_shift', self.dds_shift)
        self.lpf(self.zeros)    #parametri filtro lp
        #self.save_path = '/mnt/iqstream/'
        self.qdrCal()
        self.initialize_GbE()
        print '\n************ QDR Calibrated ************'
        print '************ Packet streaming activated ************\n'

    def fft_bin_index(self, freqs, fft_len, samp_freq):
    # returns the fft bin index for a given frequency, fft length, and sample frequency
        bin_index = np.round((freqs/samp_freq)*fft_len).astype('int')
        return bin_index

    def read_mixer_snaps(self, shift, chan, mixer_out = True):
    # returns snap data for the dds mixer inputs and outputs
        self.fpga.write_int('dds_shift', shift)
        if (chan % 2) > 0: # if chan is odd
            self.fpga.write_int('DDC_snap_chan_select', (chan - 1) / 2)
        else:
            self.fpga.write_int('DDC_snap_chan_select', chan/1)
        self.fpga.write_int('DDC_snap_rawfftbin_ctrl', 0)
        self.fpga.write_int('DDC_snap_mixerout_ctrl', 0)
        self.fpga.write_int('DDC_snap_rawfftbin_ctrl', 1)
        self.fpga.write_int('DDC_snap_mixerout_ctrl', 1)
        mixer_in = np.fromstring(self.fpga.read('DDC_snap_rawfftbin_bram', 16*2**10),dtype='>i2').astype('float')
        mixer_in /= 2.0**15
        if mixer_out:
            mixer_out = np.fromstring(self.fpga.read('DDC_snap_mixerout_bram', 8*2**10),dtype='>i2').astype('float')
            mixer_out /= 2.0**14
            return mixer_in, mixer_out
        else:
            return mixer_in

    def return_shift(self, chan):
    # Returns the dds shift
        dds_spec = np.abs(np.fft.rfft(self.I_dds[chan::self.fft_len],self.fft_len))
        dds_index = np.where(np.abs(dds_spec) == np.max(np.abs(dds_spec)))[0][0]
        print 'Finding LUT shift...' 
        for i in range(self.fft_len/2):
            print i
            mixer_in = self.read_mixer_snaps(i, chan, mixer_out = False)
            I0_dds_in = mixer_in[2::8]    
            #I0_dds_in[np.where(I0_dds_in > 32767.)] -= 65535.
            snap_spec = np.abs(np.fft.rfft(I0_dds_in,self.fft_len))
            snap_index = np.where(np.abs(snap_spec) == np.max(np.abs(snap_spec)))[0][0]
            if dds_index == snap_index:
                print 'LUT shift =', i
                shift = i
                break
        return shift

    def freq_comb(self, freqs, samp_freq, resolution, random_phase = True, DAC_LUT = True, apply_transfunc = False):
    # Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
        freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
        amp_full_scale = (2**15 - 1)
        if DAC_LUT:
            print 'freq comb uses DAC_LUT'
            fft_len = self.LUTbuffer_len
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            #np.random.seed(333)
            #phase = np.random.uniform(0., 2.*np.pi, len(bins))   #### ATTENZIONE PHASE A pi (ORA NO)
            phase = self.phases[0:len(bins)]

            if apply_transfunc:
                self.amps = self.get_transfunc(path_current = True)

            else:
                if self.target_sweep_flag == True:
                    self.amps = self.amps
                if self.test_comb_flag == True:
                    self.amps = self.test_comb_amps

                    #self.amps = np.array([1.]*len(bins)) #va commentato per impostare le attenuazioni 

            ######POTENZE DIVERSE PER ARRAY DIVERSI: 200 E 350 GHz NEL DILUIZIONE 8-08-17##################	
            #tmp, self.amps = np.loadtxt(os.path.join(self.folder_frequencies, 'target_freqs.dat'), unpack=True)COMMENTO BY ALE P.
            print self.amps
            # di seguito per applicare potenza diverse per array
#	    if !self.roach2:
##		ROACH2
#	    	self.amps[]*=
#		self.amps[]*=
#	    else
#	    	self.amps[]*=
#		self.amps[]*=

            if len(bins) != len(self.amps): self.amps=1


            if not random_phase:
                phase = np.load('/mnt/iqstream/last_phases.npy') 

            self.spec = np.zeros(fft_len,dtype='complex')
            self.spec[bins] = self.amps*np.exp(1j*(phase))
            wave = np.fft.ifft(self.spec)
            waveMax = np.max(np.abs(wave))
            waveMax = self.wavemax
            print "waveMax",waveMax, "attenuation",self.global_attenuation
            I = (wave.real/waveMax)*(amp_full_scale)*self.global_attenuation  
            Q = (wave.imag/waveMax)*(amp_full_scale)*self.global_attenuation  
            #wave = signal.convolve(wave,np.hanning(3), mode = 'same')
            #	    Q = np.roll(Q,1)
        else:
            fft_len = (self.LUTbuffer_len/self.fft_len)
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            spec = np.zeros(fft_len,dtype='complex')
            amps = np.array([1.]*len(bins))
            phase = 0.
            spec[bins] = amps*np.exp(1j*(phase))
            wave = np.fft.ifft(spec)
            #wave = signal.convolve(wave,signal.hanning(3), mode = 'same')
            waveMax = np.max(np.abs(wave))
            I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
        return I, Q    

    def select_bins(self, freqs):
    # Calculates the offset from each bin center, to be used as the DDS LUT frequencies, and writes bin numbers to RAM
        bins = self.fft_bin_index(freqs, self.fft_len, self.dac_samp_freq)
        bin_freqs = bins*self.dac_samp_freq/self.fft_len
        bins[ bins < 0 ] += self.fft_len
        self.freq_residuals = freqs - bin_freqs
        #for i in range(len(freqs)):
        #	print "bin, fbin, freq, offset:", bins[i], bin_freqs[i]/1.0e6, freqs[i]/1.0e6, self.freq_residuals[i]
        ch = 0
        for fft_bin in bins:
            self.fpga.write_int('bins', fft_bin)
            self.fpga.write_int('load_bins', 2*ch + 1)
            self.fpga.write_int('load_bins', 0)
            ch += 1
        return 

    def define_DDS_LUT(self,freqs):
# Builds the DDS look-up-table from I and Q given by freq_comb. freq_comb is called with the sample rate equal to the sample rate for a single FFT bin. There are two bins returned for every fpga clock, so the bin sample rate is 256 MHz / half the fft length  
        self.select_bins(freqs)
        I_dds, Q_dds = np.array([0.]*(self.LUTbuffer_len)), np.array([0.]*(self.LUTbuffer_len))
        for m in range(len(self.freq_residuals)):
            I, Q = self.freq_comb(np.array([self.freq_residuals[m]]), self.fpga_samp_freq/(self.fft_len/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
            I_dds[m::self.fft_len] = I
            Q_dds[m::self.fft_len] = Q
        return I_dds, Q_dds

    def pack_luts(self, freqs, transfunc = False):
    # packs the I and Q look-up-tables into strings of 16-b integers, in preparation to write to the QDR. Returns the string-packed look-up-tables
        if transfunc:
                self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True, apply_transfunc = True)
        else:
                self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True)
        self.I_dds, self.Q_dds = self.define_DDS_LUT(freqs)
        self.I_lut, self.Q_lut = np.zeros(self.LUTbuffer_len*2), np.zeros(self.LUTbuffer_len*2)
        self.I_lut[0::4] = self.I_dac[1::2]         
        self.I_lut[1::4] = self.I_dac[0::2]
        self.I_lut[2::4] = self.I_dds[1::2]
        self.I_lut[3::4] = self.I_dds[0::2]
        self.Q_lut[0::4] = self.Q_dac[1::2]         
        self.Q_lut[1::4] = self.Q_dac[0::2]
        self.Q_lut[2::4] = self.Q_dds[1::2]
        self.Q_lut[3::4] = self.Q_dds[0::2]
        print 'String Packing LUT...',
        self.I_lut_packed = self.I_lut.astype('>h').tostring()
        self.Q_lut_packed = self.Q_lut.astype('>h').tostring()
        print 'Done.'
        return 

    def writeQDR(self, freqs, transfunc = False):
    # Writes packed LUTs to QDR
        if transfunc:
                self.pack_luts(freqs, transfunc = True)
        else:
                self.pack_luts(freqs, transfunc = False)

        self.fpga.write_int('dac_reset',1)
        self.fpga.write_int('dac_reset',0)
        print 'Writing DAC and DDS LUTs to QDR...',
        self.fpga.write_int('start_dac',0)
        self.fpga.blindwrite('qdr0_memory',self.I_lut_packed,0)
        self.fpga.blindwrite('qdr1_memory',self.Q_lut_packed,0)
        self.fpga.write_int('start_dac',1)
        self.fpga.write_int('downsamp_sync_accum_reset', 0)
        self.fpga.write_int('downsamp_sync_accum_reset', 1)
        print 'Done.'
        return 

    def read_QDR_katcp(self):
    # Reads out QDR buffers with KATCP, as 16-b signed integers.    
        self.QDR0 = np.fromstring(self.fpga.read('qdr0_memory', 8 * 2**20),dtype='>i2')
        self.QDR1 = np.fromstring(self.fpga.read('qdr1_memory', 8* 2**20),dtype='>i2')
        self.I_katcp = self.QDR0.reshape(len(self.QDR0)/4.,4.)
        self.Q_katcp = self.QDR1.reshape(len(self.QDR1)/4.,4.)
        self.I_dac_katcp = np.hstack(zip(self.I_katcp[:,1],self.I_katcp[:,0]))
        self.Q_dac_katcp = np.hstack(zip(self.Q_katcp[:,1],self.Q_katcp[:,0]))
        self.I_dds_katcp = np.hstack(zip(self.I_katcp[:,3],self.I_katcp[:,2]))
        self.Q_dds_katcp = np.hstack(zip(self.Q_katcp[:,3],self.Q_katcp[:,2]))
        return        


    def read_accum_snap(self):
        # Reads the avgIQ buffer. Returns I and Q as 32-b signed integers     
        self.fpga.write_int('accum_snap_accum_snap_ctrl', 0)
        self.fpga.write_int('accum_snap_accum_snap_ctrl ', 1)
        accum_data = np.fromstring(self.fpga.read('accum_snap_accum_snap_bram', 16*2**11), dtype = '>i').astype('float')
        I0 = accum_data[0::4]    
        Q0 = accum_data[1::4]    
        I1 = accum_data[2::4]    
        Q1 = accum_data[3::4]    
        I = np.hstack(zip(I0, I1))
        Q = np.hstack(zip(Q0, Q1))
        return I, Q    

    def add_out_of_res(self,path, tones):

        freqs = np.loadtxt(path, unpack=True, usecols=0)
        freqs = np.append(freqs,tones)
        freqs = np.sort(freqs)

        np.savetxt(path,np.transpose(freqs))

    def calc_transfunc(self, path):


        '''
        Da modificare:
        - Overfittare la funzione di trasferimento. O con un poly di grado superiore o con interpolazione
        - Salvare le attenuazioni in un qualche file amps.npy
        '''

        self.transfunc_parameters_file = "/home/mistral/src/mistral_readout_dev/transfunc_polyfit_coefficients.npy"



        poly_par = np.load(self.transfunc_parameters_file)

        self.baseline_attenuation = -44.8  #dBm

        freqs = np.loadtxt(path, unpack=True, usecols=(0))

        attenuations = np.poly1d(poly_par)(freqs)
        print("prima di 10**")
        print(attenuations)
        attenuations = 10.**((self.baseline_attenuation-attenuations)/20.)


        print(attenuations)
        bad_attenuations = np.argwhere(np.floor(attenuations))
        print("Checking for bad attenuations")
        print(bad_attenuations)
        if bad_attenuations.size > 0:
            print("Warning: invalid attenuations found. Setting them to 1.0 by default.")
            attenuations[bad_attenuations] = 1.0


        table = np.array([freqs,attenuations])

        np.savetxt(path,np.transpose(table)) 

        print(attenuations)

        '''
        for f in freqs:
            attenuation = np.poly1d(poly_par)(f)
            attenuation = 10 **((self.baseline_attenuation-attenuation)/20)
        ''' 


    def get_transfunc(self, path_current=False):


        nchannel = len(self.cold_array_bb)
        channels = range(nchannel)
        tf_path = self.setupdir+"transfer_functions/"
        if path_current:
                tf_dir = self.timestring
        else:
                tf_dir = raw_input('Insert folder for TRANSFER_FUNCTION (e.g. '+self.timestring+'): ')
        save_path = os.path.join(tf_path, tf_dir)
        print "save TF in "+save_path
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+tf_path+'current'
        os.system(command_cleanlink)
        command_linkfile = "ln -f -s " + save_path +" "+ tf_path+'current'
        os.system(command_linkfile)


        by_hand = raw_input("set TF by hand (y/n)?")
        if by_hand == 'y':
                transfunc = np.zeros(nchannel)+1
                chan = input("insert channel to change")
                value = input("insert values to set [0, 1]")
                transfunc[chan] = value
        else:

#    	mag_array = np.zeros((100, len(self.test_comb)))
                mag_array = np.zeros((100,nchannel))+1
                for i in range(100):
                        packet = self.s.recv(8234) # total number of bytes including 42 byte header
                        data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                        packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                        for chan in channels:
                                if (chan % 2) > 0:
                                        I = data[1024 + ((chan - 1) / 2)]    
                                        Q = data[1536 + ((chan - 1) /2)]    
                                else:
                                        I = data[0 + (chan/2)]    
                                        Q = data[512 + (chan/2)]    

                                mags = np.sqrt(I**2 + Q**2)
                                if mags ==0:
                                        mags = 1
                                print chan, mags
                                mag_array[i,chan] = mags     #[2:len(self.test_comb)+2]
                transfunc = np.mean(mag_array, axis = 0)
                transfunc = 1./ (transfunc / np.max(transfunc))

        np.save(save_path+'/last_transfunc.npy',transfunc)
        return transfunc


    def get_transfunc_old(self, path_current=False):



        tf_path = self.setupdir+"transfer_functions/"
        if path_current:
                tf_dir = self.timestring
        else:
                tf_dir = raw_input('Insert folder for TRANSFER_FUNCTION (e.g. 161020_01): ')
        save_path = os.path.join(tf_path, tf_dir)
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+tf_path+'current'
        os.system(command_cleanlink)
        command_linkfile = "ln -f -s " + save_path +" "+ tf_path+'current'
        os.system(command_linkfile)

        mag_array = np.zeros((100, len(self.test_comb)))
        for i in range(100):
                I, Q = self.read_accum_snap()
                mags = np.sqrt(I**2 + Q**2)
                mag_array[i] = mags[2:len(self.test_comb)+2]
        transfunc = np.mean(mag_array, axis = 0)
        transfunc = 1./ (transfunc / np.max(transfunc))

        np.save(save_path+'./last_transfunc.npy',transfunc)
        return transfunc

    def initialize_GbE(self):
        # Configure GbE Block. Run immediately after calibrating QDR.
        self.fpga.write_int('GbE_tx_rst',0)
        self.fpga.write_int('GbE_tx_rst',1)
        self.fpga.write_int('GbE_tx_rst',0)
        return

    def stream_UDP(self,chan,Npackets):
        self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets:
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            forty_two = (np.fromstring(packet[-16:-12],dtype = '>I'))
            pps_count = (np.fromstring(packet[-12:-8],dtype = '>I'))
            time_stamp = np.round((np.fromstring(packet[-8:-4],dtype = '>I').astype('float')/self.fpga_samp_freq)*1.0e3,3)
            packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
            if (chan % 2) > 0:
                I = data[1024 + ((chan - 1) / 2)]    
                Q = data[1536 + ((chan - 1) /2)]    
            else:
                I = data[0 + (chan/2)]    
                Q = data[512 + (chan/2)]    
            phase = np.arctan2([Q],[I])
            print forty_two, pps_count, time_stamp, packet_count, phase
            count += 1
        return 

    def stream_cosmic(self,chan,time_interval):
        accum_len = 2**12
        self.fpga.write_int('downsamp_sync_accum_len', accum_len - 1)
        accum_freq = self.fpga_samp_freq / (accum_len - 1)
        Npackets = int(time_interval * accum_freq)
        self.fpga.write_int('GbE_pps_start', 1)
        #running_sum = 0.
        #running_sum_sq = 0.
        #running_avg = 0.
        #running_std = 0.
        while 1:
                count = 0
                phases = np.zeros(Npackets)
                while count < Npackets:
                    packet = self.s.recv(8234) # total number of bytes including 42 byte header
                    data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                    #forty_two = (np.fromstring(packet[-16:-12],dtype = '>I'))
                    #pps_count = (np.fromstring(packet[-12:-8],dtype = '>I'))
                    #time_stamp = np.round((np.fromstring(packet[-8:-4],dtype = '>I').astype('float')/self.fpga_samp_freq)*1.0e3,3)
                    packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                    if (chan % 2) > 0:
                        I = data[1024 + ((chan - 1) / 2)]    
                        Q = data[1536 + ((chan - 1) /2)]    
                    else:
                        I = data[0 + (chan/2)]    
                        Q = data[512 + (chan/2)]    
                    phases[count] = np.arctan2([Q],[I])
                    #running_sum += phase
                    #running_sum_sq += phase**2 
                    #running_avg = running_sum / count
                    #running_std = (running_sum_sq / count) - running_avg**2
                    count += 1
                std10 = 3.5*np.std(phases)
                mean = np.mean(phases)
                min_phase, max_phase = np.min(phases), np.max(phases)
                if ((min_phase - mean) < std10) | ((max_phase - mean) > std10 ):
                        print 'outlier'
                        np.save('/home/olimpo/home/data/cosmic/' + str(time.time()) + '_' + str(chan) + '.npy', phases)
                print count, mean, std10
        self.fpga.write_int('downsamp_sync_accum_len', self.accum_len - 1)
        return 

    def dirfile_all_chan(self, nchannel):
        channels = range(nchannel)
        sub_folder_1 = ""
        sub_folder_2 = "dirfile_"+self.timestring+"/"
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(self.datadir, sub_folder_1, sub_folder_2)
        sys.stdout.write( "log data in "+save_path+"\n")
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+self.datadir+"dirfile_kids_current" 
        os.system(command_cleanlink)
#	command_linkfile = "ln -f -s " + save_path +" "+ self.datadir + "dirfile_kids_current"
        command_linkfile = "ln -f -s ../" + sub_folder_2 +" "+ self.datadir + "dirfile_kids_current"


        os.system(command_linkfile)
        shutil.copy(self.datadir + "/format", save_path + "/format")
        shutil.copy(self.datadir + "/format_extra", save_path + "/format_extra")
        nfo_I = map(lambda x: save_path + "/chI_" + str(x).zfill(3), range(nchannel))
        nfo_Q = map(lambda y: save_path + "/chQ_" + str(y).zfill(3), range(nchannel))
        #nfo_phase = map(lambda z: save_path + "/chP_" + str(z).zfill(3), range(nchannel))
        fo_I = map(lambda x: open(x, "ab"), nfo_I)
        fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
        #fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(save_path + "/time", "ab")
        fo_count = open(save_path + "/packet_count", "ab")	
        count = 0
#	while count < Npackets:
        try:
            while True:
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                if(len(packet) == 8234):

                    data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                    packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                    for chan in channels:
                        if (chan % 2) > 0:
                                I = data[1024 + ((chan - 1) / 2)]  
                                Q = data[1536 + ((chan - 1) /2)]      
                        else:
                                I = data[0 + (chan/2)]    
                                Q = data[512 + (chan/2)] 
                        fo_I[chan].write(struct.pack('f',I))
                        fo_Q[chan].write(struct.pack('f',Q))
                        #fo_phase[chan].write(struct.pack('f', np.arctan2([Q],[I])))
#			fo_I[chan].flush()
#			fo_Q[chan].flush()
#			fo_phase[chan].flush()
                    count += 1
                    fo_time.write(struct.pack('d', ts))
                    fo_count.write(struct.pack('I',packet_count))
#		fo_time.flush()
#		fo_count.flush()
                    if(count/20 == count/20.):
                        fo_time.flush()
                        fo_count.flush()
                        map(lambda x: (x.flush()), fo_I)
                        map(lambda x: (x.flush()), fo_Q)
                        #map(lambda x: (x.flush()), fo_phase)	
            else: print "Incomplete packet received, length ", len(packet)
        except KeyboardInterrupt:
            pass
        for chan in channels:
                fo_I[chan].close()
                fo_Q[chan].close()
                #fo_phase[chan].close()
        fo_time.close()
        fo_count.close()
        return 

    def dirfile_all_chan_packet(self, nchannel):
#	I_center = self.centers.real
#	Q_center = self.centers.imag
#	cosine = np.cos(self.rotations) 
#	sine = np.sin(self.rotations)
        channels = range(nchannel)
        sub_folder_1 = ""
        sub_folder_2 = "packet_"+self.timestring+"/"
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(self.datadir, sub_folder_1, sub_folder_2)
        sys.stdout.write( "log packets in "+save_path+"\n")
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+self.datadir+"packet_kids_current" 
        os.system(command_cleanlink)
        command_linkfile = "ln -f -s " + save_path +" "+ self.datadir + "packet_kids_current"
        os.system(command_linkfile)

        fo = open(os.path.join(save_path, 'packet'))

        count = 0
#	while count < Npackets:
        try:
            while True:
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                if(len(packet) == 8234):
                    packet_count_bin = packet[-1]+packet[-2]+packet[-3]+packet[-4]
                    data_bin = packet[42+1024:42+1024+nchannel*4/2]+packet[42+1536:42+1536+nchannel*4/2]+packet[42+0:42+0+nchannel*4/2]+packet[42+512:42+512+nchannel*4/2]+packet[-4:]  
                    # the string above contains all data in the packet, including packet number. 
                    # It can be saved on disc to save time, and then decoded. 
                    fo.write(data_bin)


                    count += 1

                    if(count/20 == count/20.):
                        fo.flush()
                else: print "Incomplete packet received, length ", len(packet)
        except KeyboardInterrupt:
            pass
        fo.close()
        return 


    def dirfile_complex(self, nchannel):
#	I_center = self.centers.real
#	Q_center = self.centers.imag
#	cosine = np.cos(self.rotations) 
#	sine = np.sin(self.rotations)
        channels = range(nchannel)
        sub_folder_1 = ""
        sub_folder_2 = "dirfile_"+self.timestring+"/"
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(self.datadir, sub_folder_1, sub_folder_2)
        sys.stdout.write( "log data in "+save_path+"\n")
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+self.datadir+"dirfile_kids_current" 
        os.system(command_cleanlink)
        #command_linkfile = "ln -f -s " + save_path +" "+ self.datadir + "dirfile_kids_current"
        command_linkfile = "ln -f -s " + sub_folder_2 +" "+ self.datadir + "dirfile_kids_current"


        os.system(command_linkfile)
        shutil.copy(self.datadir + "/format_complex", save_path + "/format")
        shutil.copy(self.datadir + "/format_extra", save_path + "/format_extra")
        self.make_format_complex()
        shutil.copy(self.datadir + "/format_complex_extra", save_path + "/format_complex_extra")
        nfo_I = map(lambda x: save_path + "/chI_" + str(x).zfill(3), range(nchannel))
        nfo_Q = map(lambda y: save_path + "/chQ_" + str(y).zfill(3), range(nchannel))
        #nfo_phase = map(lambda z: save_path + "/chP_" + str(z).zfill(3), range(nchannel))
        fo_I = map(lambda x: open(x, "ab"), nfo_I)
        fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
        #fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(save_path + "/time", "ab")
        fo_count = open(save_path + "/packet_count", "ab")	
        count = 0
#	while count < Npackets:
        try:
            while True:
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                if(len(packet) == 8234):
#		    data_bin = packet[42+1024:42+1024+nchannel*4/2]+packet[42+1536:42+1536+nchannel*4/2]+packet[42+0:42+0+nchannel*4/2]+packet[42+512:42+512+nchannel*4/2]+packet[-4:]  
                    # the string above contains all data in the packet, including packet number. 
                    # It can be saved on disc to save time, and then decoded. 
                    data_bin = packet[42:]
#           	    data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                    packet_count_bin = packet[-1]+packet[-2]+packet[-3]+packet[-4]#packet[-4:]
                    #packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                    for chan in channels:
                        if (chan % 2) > 0:   # odd channels 
#              			I_ = data[1024 + ((chan - 1) / 2)] - I_center[chan] # stored in packet[42 + 1024: 42+1024+nchannel*4/2] 
#                		Q_ = data[1536 + ((chan - 1) / 2)] - Q_center[chan] # stored in packet[42 + 1536: 42+1536+nchannel*4/2]  
#				IQ = data[1024 + ((chan - 1) / 2)] + 1j * data[1536 + ((chan - 1) / 2)]
                                I = data_bin[(1024+((chan-1) /2))*4 : (1024+((chan-1) /2))*4+4 ] 
                                Q = data_bin[(1536+((chan-1) /2))*4 : (1536+((chan-1) /2))*4+4 ] 
                        else:                # even channels
#                		I_ = data[0   + (chan/2)] - I_center[chan] # stored in packet[42 + 0:   42+  0+nchannel*4/2]  
#                		Q_ = data[    512 + (chan/2)] - Q_center[chan] # stored in packet[42 + 512: 42+512+nchannel*4/2]
                                I = data_bin[(   0+ (chan/2))*4 : (   0+(chan /2))*4+4 ]  
                                Q = data_bin[( 512+ (chan/2))*4 : ( 512+(chan /2))*4+4 ]
#			I = (cosine[chan]*I_ + sine[chan]*Q_)/self.radii[chan]
#			Q = (-sine[chan]*I_  + cosine[chan]*Q_)/self.radii[chan]
                        #I = np.real((I_+1j*Q_) * np.exp(-1j*self.rotations[chan]))/self.radii[chan]
                        #Q = np.imag((I_+1j*Q_) * np.exp(-1j*self.rotations[chan]))/self.radii[chan]

                        fo_I[chan].write(I)
                        fo_Q[chan].write(Q)
#			fo_I[chan].write(struct.pack('f',I))
#	    		fo_Q[chan].write(struct.pack('f',Q))
#			if count == 0: print count, len(Q), Q

        #		fo_phase[chan].write(struct.pack('f', np.arctan2([Q],[I])))

                    count += 1
                    fo_time.write(struct.pack('d', ts))
                    #fo_count.write(struct.pack('I',packet_count))
                    fo_count.write(packet_count_bin)
                    fo_count.flush()
                    if(count/20 == count/20.):
                        fo_time.flush()
                #	fo_count.flush()
                        map(lambda x: (x.flush()), fo_I)
                        map(lambda x: (x.flush()), fo_Q)
        #		map(lambda x: (x.flush()), fo_phase)
                else: print "Incomplete packet received, length ", len(packet)
        except KeyboardInterrupt:
            pass
        for chan in channels:
                fo_I[chan].close()
                fo_Q[chan].close()
        #	fo_phase[chan].close()
        fo_time.close()
        fo_count.close()
        return 



    def dirfile_all_chan_phase_centered(self, nchannel):
        I_center = self.centers.real
        Q_center = self.centers.imag
        cosine = np.cos(self.rotations) 
        sine = np.sin(self.rotations)
        #nchannel=57
        channels = range(nchannel)
        #print channels
        sub_folder_1 = ""
        sub_folder_2 = "dirfile_"+self.timestring+"/"
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(self.datadir, sub_folder_1, sub_folder_2)
        sys.stdout.write( "log data in "+save_path+"\n")
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+self.datadir+"dirfile_kids_current" 
        os.system(command_cleanlink)
        command_linkfile = "ln -f -s " + save_path +" "+ self.datadir + "dirfile_kids_current"
        os.system(command_linkfile)
        shutil.copy(self.datadir + "/format", save_path + "/format")
        shutil.copy(self.datadir + "/format_extra", save_path + "/format_extra")
        nfo_I = map(lambda x: save_path + "/chI_" + str(x).zfill(3), range(nchannel))
        nfo_Q = map(lambda y: save_path + "/chQ_" + str(y).zfill(3), range(nchannel))
        #nfo_phase = map(lambda z: save_path + "/chP_" + str(z).zfill(3), range(nchannel))
        fo_I = map(lambda x: open(x, "ab"), nfo_I)
        fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
        #fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(save_path + "/time", "ab")
        fo_count = open(save_path + "/packet_count", "ab")	
        count = 0
#	while count < Npackets:
        try:
            while True:
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                if(len(packet) == 8234):
#		    data_bin = packet[42+1024:42+1024+nchannel*4/2]+packet[42+1536:42+1536+nchannel*4/2]+packet[42+0:42+0+nchannel*4/2]+packet[42+512:42+512+nchannel*4/2]+packet[-4:]  
                    # the string above contains all data in the packet, including packet number. 
                    # It can be saved on disc to save time, and then decoded. 
                    data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                    packet_count_bin = packet[-1]+packet[-2]+packet[-3]+packet[-4]#packet[-4:]
                    #packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                    for chan in channels:
                        if (chan % 2) > 0:   # odd channels 
                                I_ = data[1024 + ((chan - 1) / 2)] - I_center[chan] # stored in packet[42 + 1024: 42+1024+nchannel*4/2] 
                                Q_ = data[1536 + ((chan - 1) / 2)] - Q_center[chan] # stored in packet[42 + 1536: 42+1536+nchannel*4/2]  
                        else:                # even channels
                                I_ = data[0   + (chan/2)] - I_center[chan] # stored in packet[42 + 0:   42+  0+nchannel*4/2]  
                                Q_ = data[512 + (chan/2)] - Q_center[chan] # stored in packet[42 + 512: 42+512+nchannel*4/2]
                        I = (cosine[chan]*I_ + sine[chan]*Q_)/self.radii[chan]
                        Q = (-sine[chan]*I_  + cosine[chan]*Q_)/self.radii[chan]
                        #I = np.real((I_+1j*Q_) * np.exp(-1j*self.rotations[chan]))/self.radii[chan]
                        #Q = np.imag((I_+1j*Q_) * np.exp(-1j*self.rotations[chan]))/self.radii[chan]

                        fo_I[chan].write(struct.pack('f',I))
                        fo_Q[chan].write(struct.pack('f',Q))
        #		fo_phase[chan].write(struct.pack('f', np.arctan2([Q],[I])))

                    count += 1
                    fo_time.write(struct.pack('d', ts))
                    #fo_count.write(struct.pack('I',packet_count))
                    fo_count.write(packet_count_bin)
                    fo_count.flush()
                    if(count/20 == count/20.):
                        fo_time.flush()
                #	fo_count.flush()
                        map(lambda x: (x.flush()), fo_I)
                        map(lambda x: (x.flush()), fo_Q)
        #		map(lambda x: (x.flush()), fo_phase)
                else: print "Incomplete packet received, length ", len(packet)
        except KeyboardInterrupt:
            pass
        for chan in channels:
                fo_I[chan].close()
                fo_Q[chan].close()
        #	fo_phase[chan].close()
        fo_time.close()
        fo_count.close()
        return 



    def dirfile_phase_centered(self, time_interval):
        target_path = raw_input('Absolute path to target sweep dir (e.g. /home/data/olimpo/setup/kids/sweeps/target/161020_1): ')
        self.centers = np.load(target_path + '/centers.npy')
        I_center = self.centers.real
        Q_center = self.centers.imag
        nchannel = input("Number of channels? ")
        channels = range(nchannel)
        data_path = "/home/data"
        sub_folder_1 = "noise_measurements_0806"
        sub_folder_2 = raw_input("Insert subfolder name (e.g. single_tone): ")
        Npackets = np.int(time_interval * self.accum_freq)
        #channels = np.arange(21)
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(data_path, sub_folder_1, sub_folder_2)
        os.mkdir(save_path)
        shutil.copy(data_path + "/format", save_path + "/format")
        nfo_I = map(lambda x: save_path + "/chI_" + str(x), range(nchannel))
        nfo_Q = map(lambda y: save_path + "/chQ_" + str(y), range(nchannel))
        nfo_phase = map(lambda z: save_path + "/chP_" + str(z), range(nchannel))
        fo_I = map(lambda x: open(x, "ab"), nfo_I)
        fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
        fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(save_path + "/time", "ab")
        fo_count = open(save_path + "/packet_count", "ab")	
        count = 0
        while count < Npackets:
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                for chan in channels:
                        if (chan % 2) > 0:
                                I = data[1024 + ((chan - 1) / 2)]    
                                Q = data[1536 + ((chan - 1) /2)]    
                        else:
                                I = data[0 + (chan/2)]    
                                Q = data[512 + (chan/2)]    
                        fo_I[chan].write(struct.pack('i',I))
                        fo_Q[chan].write(struct.pack('i',Q))
                        fo_phase[chan].write(struct.pack('f', np.arctan2([Q - Q_center[chan]],[I - I_center[chan]])))
                        fo_I[chan].flush()
                        fo_Q[chan].flush()
                        fo_phase[chan].flush()
                count += 1
                fo_time.write(struct.pack('d', ts))
                fo_count.write(struct.pack('L',packet_count))
                fo_time.flush()
                fo_count.flush()
        for chan in channels:
                fo_I[chan].close()
                fo_Q[chan].close()
                fo_phase[chan].close()
        fo_time.close()
        fo_count.close()
        return 

    def dirfile_avg_chan(self, time_interval):
        nchannel = input("Number of channels? ")
        channels = range(nchannel)
        data_path = "/home/olimpo/data"
        sub_folder_1 = "noise_measurements_0806"
        sub_folder_2 = raw_input("Insert subfolder name (e.g. single_tone): ")
        Npackets = np.int(time_interval * self.accum_freq)
        self.fpga.write_int('GbE_pps_start', 1)
        save_path = os.path.join(data_path, sub_folder_1, sub_folder_2)
        os.mkdir(save_path)
        shutil.copy(data_path + "/format", save_path + "/format")
        nfo_I = save_path + "/chI_avg_" + str(nchannel)
        nfo_Q = save_path + "/chQ_avg_" + str(nchannel)
        nfo_P = save_path + "/chP_avg_" + str(nchannel)
        fo_I = open(nfo_I, "ab")
        fo_Q = open(nfo_Q, "ab")
        fo_phase = open(nfo_P, "ab")
        fo_time = open(save_path + "/time", "ab")
        fo_count = open(save_path + "/packet_count", "ab")	
        count = 0
        while count < Npackets:
                I_sum = 0.
                Q_sum = 0.
                phase_sum = 0.
                ts = time.time()
                packet = self.s.recv(8234) # total number of bytes including 42 byte header
                data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
                for chan in channels:
                        if (chan % 2) > 0:
                                I = data[1024 + ((chan - 1) / 2)]    
                                Q = data[1536 + ((chan - 1) /2)]    
                        else:
                                I = data[0 + (chan/2)]    
                                Q = data[512 + (chan/2)]    
                        I_sum += I
                        Q_sum += Q
                I_avg = I_sum / nchannel
                Q_avg = Q_sum / nchannel
                phase_avg = np.arctan2([Q_avg],[I_avg]) 
                fo_I.write(struct.pack('i',I_avg))
                fo_Q.write(struct.pack('i',Q_avg))
                fo_time.write(struct.pack('d', ts))
                fo_count.write(struct.pack('L',packet_count))
                fo_time.flush()
                fo_count.flush()
                fo_phase.write(struct.pack('f', phase_avg))
                fo_I.flush()
                fo_Q.flush()
                fo_phase.flush()
                count += 1
        fo_I.close()
        fo_Q.close()
        fo_phase.close()
        fo_time.close()
        fo_count.close()
        return 

    def kst_UDP(self, chan, time_interval):
        Npackets = np.int(time_interval * self.accum_freq)
        self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        f_chan = open('/home/olimpo/home/data/ch' + str(chan), 'ab')
        f_time = open('/home/olimpo/home/data/time' + str(chan), 'ab')
        while count < Npackets:
            ts = time.time()
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            if (chan % 2) > 0:
                I = data[1024 + ((chan - 1) / 2)]    
                Q = data[1536 + ((chan - 1) /2)]    
            else:
                I = data[0 + (chan/2)]    
                Q = data[512 + (chan/2)]    
            phase = np.arctan2([Q],[I])
            f_chan.write(struct.pack('f', phase))
            f_time.write(struct.pack('d', ts))
            f_time.flush()
            f_chan.flush()
            count += 1
        f_chan.close()
        f_time.close()
        return 

    def IQ_grad(self, dark_sweep_path, plot_chan): 
        lo_freqs, I_dark, Q_dark = self.open_stored(dark_sweep_path)
        bb_freqs, delta_f = np.linspace(-200.0e6, 200.0e6, 500,retstep=True)
        #bb_freqs = np.load('/mnt/iqstream/last_bb_freqs.npy')
        channels = np.arange(len(bb_freqs))
        delta_lo = 5e3
        i_index = [np.where(np.abs(np.diff(I_dark[:,chan])) == np.max(np.abs(np.diff(I_dark[:,chan]))))[0][0] for chan in channels]
        q_index = [np.where(np.abs(np.diff(Q_dark[:,chan])) == np.max(np.abs(np.diff(Q_dark[:,chan]))))[0][0] for chan in channels]
        di_df = np.array([(I_dark[:,chan][i_index[chan] + 1] - I_dark[:,chan][i_index[chan] - 1])/(2*delta_lo) for chan in channels])
        dq_df = np.array([(Q_dark[:,chan][q_index[chan] + 1] - Q_dark[:,chan][q_index[chan] - 1])/(2*delta_lo) for chan in channels])
        I0 = np.array([I_dark[:,chan][i_index[chan]] for chan in channels])
        Q0 = np.array([Q_dark[:,chan][q_index[chan]] for chan in channels])
        rf_freqs = np.array([200.0e6 + bb_freqs[chan] for chan in channels])
        return di_df[plot_chan], dq_df[plot_chan], rf_freqs[plot_chan]

    def plot_stream_UDP(self, chan):
        dark_sweep_path = '/mnt/iqstream/vna_sweeps/scaled2'
        di_df, dq_df, rf_freq = self.IQ_grad(dark_sweep_path, chan)
        Npackets = 244
        self.fpga.write_int('GbE_pps_start', 1)
        fig = plt.figure(num= None, figsize=(18,12), dpi=80, facecolor='w', edgecolor='w')
        plt.suptitle('1s stream: Channel ' + str(chan) + ', Freq = ' + str(np.round(rf_freq/1.0e6,3)) + ' MHz', fontsize = 20)
        # channel phase
        plot1 = fig.add_subplot(211)
        plot1.set_ylabel('rad')
        line1, = plot1.plot(np.arange(Npackets), np.zeros(Npackets), 'k-', linewidth = 1)
        plt.grid()
        # df
        plot2 = fig.add_subplot(212)
        plot2.set_ylabel('Hz')
        line2, = plot2.plot(np.arange(Npackets), np.zeros(Npackets), 'b-', linewidth = 1)
        plt.grid()
        plt.xlabel('Packet #', fontsize = 20)
        plt.show(block = False)
        stop = 1.0e6
        count = 0
        phases = np.zeros(Npackets)
        delta_I = np.zeros(Npackets)
        delta_Q = np.zeros(Npackets)
        df = np.zeros(Npackets)
        chan_freq = rf_freq
        while count < stop:
                packet_count = 0
                while packet_count < Npackets:
                        packet = self.s.recv(8234) # total number of bytes including 42 byte header
                        data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                        #data = np.fromstring(packet,dtype = '<i').astype('float')
                        data /= 2.0**17
                        data /= (self.accum_len/512.)
                        ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
                        if (chan % 2) > 0:
                                I = data[1024 + ((chan - 1) / 2)]    
                                Q = data[1536 + ((chan - 1) /2)]    
                        else:
                                I = data[0 + (chan/2)]    
                                Q = data[512 + (chan/2)]    
                        phases[packet_count] = np.arctan2([Q],[I])
                        if (count and packet_count) == 0:
                                I0 = I
                                Q0 = Q
                        delta_I = I - I0 	
                        delta_Q = Q - Q0 	
                        df[packet_count] = ((delta_I * di_df) + (delta_Q * dq_df)) / (di_df**2 + dq_df**2)
                        packet_count +=1
                avg_phase = np.round(np.mean(phases),5)
                avg_df = np.round(np.mean(df[1:]))
                avg_dfbyf = avg_df / chan_freq
                plot1.set_ylim((np.min(phases) - 1.0e-3,np.max(phases)+1.0e-3))
                plot2.set_ylim((np.min(df[1:]) - 1.0e-3,np.max(df[1:])+1.0e-3))
                line1.set_ydata(phases)
                line2.set_ydata(df)
                plot1.set_title('Phase, avg =' + str(avg_phase) + ' rad', fontsize = 18)
                plot2.set_title('Delta f, avg =' + str(avg_df) + 'Hz' + ', avg df/f = ' + str(avg_dfbyf), fontsize = 18)
                plt.draw()
                count += 1
        return 

    def vna_sweep(self, write = False, sweep = False, path_current = False, do_plot = True):

        '''
        Function performing a VNA sweep to roughly identify resonances. 
        '''

        center_freq = self.center_freq*1.e6
        sweep_path = self.setupdir+"sweeps/vna/"
        if path_current:
#		self.timestring = "%04d%02d%02d_%02d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5])
                sweep_dir = self.timestring
        else:
                sweep_dir = raw_input('Insert new VNA sweep dir (e.g. 161020_01): ')
        save_path = os.path.join(sweep_path, sweep_dir)
        try:
                os.mkdir(save_path)
        except OSError:
                pass
        command_cleanlink = "rm -f "+sweep_path+'current'
        os.system(command_cleanlink)
        command_linkfile = "ln -f -s " + save_path +" "+ sweep_path+'current'
        os.system(command_linkfile)
        self.v1.set_frequency(2, center_freq/1.0e6, 0.01)
        span = self.neg_delta

        start = center_freq - (span/2)   # era (span/2)
        stop  = center_freq + (span/2)   # era (span/2)
        step  = 1.25e3        # era senza *2.0
        sweep_freqs = np.arange(start, stop, step)
        self.sweep_freqs = np.round(sweep_freqs/step)*step
        print "LO freqs =", self.sweep_freqs
        np.save(save_path + '/bb_freqs.npy',self.test_comb)
        np.save(save_path + '/sweep_freqs.npy',self.sweep_freqs)
        if write:
                self.writeQDR(self.test_comb)
        if sweep:
                for freq in tqdm.tqdm(self.sweep_freqs):
                    #print 'Sweep freq =', freq/1.0e6
                    if self.v1.set_frequency(2, freq/1.0e6, 0.01):
                        #time.sleep(1)	    	
                        self.store_UDP(100,freq,save_path,channels=len(self.test_comb)) 
                self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO
        if do_plot:
                self.plot_vna(save_path)
        #self.find_kids_olimpo.main(path)
        return 

    def target_sweep(self, write = True, sweep = False, path_current = False, do_plot=True, olimpo=False):

        '''
        Function used for tuning. It spans a small range of frequencies around known resonances.
        '''
        self.target_sweep_flag = True

        #self.setupdir = "/data/mistral/setup/kids/"
        target_path = self.setupdir+'sweeps/target/' 
        center_freq = self.center_freq*1.e6
        if path_current:
                vna_path = self.setupdir+'sweeps/vna/current'
                sweep_dir = self.timestring
        else:
            #print "Mistral:    /home/olimpo/src/parameters/Mistral_prot"
            #print "150-480: /home/olimpo/src/parameters/OLIMPO_150_480_VOLO" 
            #print "200-350: /home/olimpo/src/parameters/OLIMPO_200_350_VOLO"
            #print "ASI:    /home/olimpo/src/parameters/ASI_KIDS_300mK"
            #print "MISTRAL:    /home/olimpo/src/parameters/Mistral_prot2_opt"
            #print "finto kid:    /home/olimpo/src/parameters/prova_finto_kid"
            #print "current: /home/data/olimpo/setup/kids2/sweeps/target/current"
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

    def sweep_lo(self, Npackets_per = 100, channels = None, span = 2.0e6, save_path = '/sweeps/vna'):
        center_freq = self.center_freq*1.e6
        for freq in self.sweep_freqs:
            print 'Sweep freq =', freq/1.0e6
            if self.v1.set_frequency(2, freq/1.0e6, 0.01): 
                self.store_UDP(Npackets_per,freq,save_path,channels=channels) 
        self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO
        return

    def store_UDP(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):
        channels = np.arange(channels)
        I_buffer = np.empty((Npackets + skip_packets, len(channels)))
        Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
        #self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets + skip_packets:
            packet = self.s.recv(8234)#era 8234 # total number of bytes including 42 byte header
            if(len(packet) == 8234):
                #data = np.fromstring(packet,dtype = '<i').astype('float')
                data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
                odd_chan = channels[1::2]
                even_chan = channels[0::2]
                I_odd = data[1024 + ((odd_chan - 1) /2)]    
                Q_odd = data[1536 + ((odd_chan - 1) /2)]    
                I_even = data[0 + (even_chan/2)]    
                Q_even = data[512 + (even_chan/2)]    
                even_phase = np.arctan2(Q_even,I_even)
                odd_phase = np.arctan2(Q_odd,I_odd)
                if len(channels) % 2 > 0:
                        if len(I_odd) > 0:
                                I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
                                Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
                                I = np.hstack((I, I_even[-1]))    
                                Q = np.hstack((Q, Q_even[-1]))
                        else:
                                I = I_even[0]
                                Q = Q_even[0]
                        I_buffer[count] = I
                        Q_buffer[count] = Q
                else:
                        I = np.hstack(zip(I_even, I_odd))
                        Q = np.hstack(zip(Q_even, Q_odd))
                        I_buffer[count] = I
                        Q_buffer[count] = Q
                count += 1
            else: print "Incomplete packet received, length ", len(packet)	
        I_file = 'I' + str(LO_freq)
        Q_file = 'Q' + str(LO_freq)
        np.save(os.path.join(save_path,I_file), np.mean(I_buffer[skip_packets:], axis = 0)) 
        np.save(os.path.join(save_path,Q_file), np.mean(Q_buffer[skip_packets:], axis = 0)) 
        return 

    def plot_vna(self, path):
        plt.ion()
        plt.figure(5)
        plt.clf()
        sweep_freqs, Is, Qs = ri.open_stored(path)
        sweep_freqs = np.load(path + '/sweep_freqs.npy')
        bb_freqs = np.load(path + '/bb_freqs.npy')
        rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
        for chan in range(len(bb_freqs)):
                rf_freqs[chan] = ((sweep_freqs) + bb_freqs[chan])/1.0e6 #era sweep_freqs/2

        Q = np.reshape(np.transpose(Qs),(len(Qs[0])*len(sweep_freqs)))
        I = np.reshape(np.transpose(Is),(len(Is[0])*len(sweep_freqs)))
        mag = np.sqrt(I**2 + Q**2)
        mag /= (2**31 -1)
        mag /= ((self.accum_len - 1) / (self.fft_len/2))
        mag = 20*np.log10(mag)
        #mag = np.concatenate((mag[len(mag)/2:],mag[:len(mag)/2]))
        rf_freqs = np.hstack(rf_freqs)
        rll_chanf_freqs = np.concatenate((rf_freqs[len(rf_freqs)/2:],rf_freqs[:len(rf_freqs)/2]))
        plt.plot(rf_freqs, mag)
        plt.title('VNA sweep')
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
        return

    def plot_targ(self, path):
        plt.ion()
        plt.figure(6)
        plt.clf()
        lo_freqs, Is, Qs = ri.open_stored(path)
        lo_freqs = np.load(path + '/sweep_freqs.npy')
        bb_freqs = np.load(path + '/bb_freqs.npy')
        tt_freqs = np.load(path + '/target_freqs.npy')
        tt_freqs_new = 0
        try:
                tt_freqs_new = np.load(path + '/target_freqs_new.npy')
                indexmin = np.load(path + '/index_freqs_new.npy')
                do_plot_new = 1
                sys.stdout.write('new frequencies read \n')
        except IOError:
                do_plot_new = 0

        channels = len(bb_freqs)
        mags = np.zeros((channels,len(lo_freqs))) 
        chan_freqs = np.zeros((channels,len(lo_freqs)))
        new_targs = np.zeros((channels))
        for chan in range(channels):
                mags[chan] = np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2)
                mags[chan] /= (2**31 - 1)
                mags[chan] /= ((self.accum_len - 1) / (self.fft_len/2))
                mags[chan] = 20*np.log10(mags[chan])
                chan_freqs[chan] = (lo_freqs + bb_freqs[chan])/1.0e6

        #mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
        #bb_freqs = np.concatenate(bb_freqs[len(b_freqs)/2:],bb_freqs[:len(bb_freqs)/2]))
        #chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
        new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]
        print new_targs
        for chan in range(channels):
                plt.plot(chan_freqs[chan],mags[chan])
                if do_plot_new == 1:
                        #print new_targs[chan], tt_freqs_new[chan]
                        plt.plot(tt_freqs_new[chan], mags[chan,indexmin[chan]], 'o')
                else:
                        plt.plot(chan_freqs[chan,len(lo_freqs)], mags[chan,len(lo_freqs)/2], 'o')

        #	plt.plot(tt_freqs[chan], np.min(mags[chan]), 'o')
        plt.title('Target sweep')
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
        return

    def store_UDP_noavg(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):
        #Npackets = np.int(time_interval * self.accum_freq)
        channels = np.arange(channels)
        I_buffer = np.empty((Npackets + skip_packets, len(channels)))
        Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
        self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets + skip_packets:
            packet = self.s.recv(8192 + 42) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
            odd_chan = channels[1::2]
            even_chan = channels[0::2]
            I_odd = data[1024 + ((odd_chan - 1) / 2)]    
            Q_odd = data[1536 + ((odd_chan - 1) /2)]    
            I_even = data[0 + (even_chan/2)]    
            Q_even = data[512 + (even_chan/2)]    
            even_phase = np.arctan2(Q_even,I_even)
            odd_phase = np.arctan2(Q_odd,I_odd)
            if len(channels) % 2 > 0:
                I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
                Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
                I = np.hstack((I, I_even[-1]))    
                Q = np.hstack((Q, Q_even[-1]))    
                I_buffer[count] = I
                Q_buffer[count] = Q
            else:
                I = np.hstack(zip(I_even, I_odd))
                Q = np.hstack(zip(Q_even, Q_odd))
                I_buffer[count] = I
                Q_buffer[count] = Q
            count += 1
        I_file = 'I' + str(LO_freq)
        Q_file = 'Q' + str(LO_freq)
        np.save(os.path.join(save_path,I_file), I_buffer[skip_packets:]) 
        np.save(os.path.join(save_path,Q_file), Q_buffer[skip_packets:]) 
        return 

    def open_stored(self, save_path = None):
        files = sorted(os.listdir(save_path))
        sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
        I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
        Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
        Is = np.array([np.load(filename) for filename in I_list])
        Qs = np.array([np.load(filename) for filename in Q_list])
        return sweep_freqs, Is, Qs

    def get_kid_freqs(self, path):
        sweep_step = 1.25 # kHz
        smoothing_scale = 1500.0 # kHz
        peak_threshold = 0.4 # mag units
        spacing_threshold = 50.0 # kHz
        #find_kids_olimpo.get_kids(path, self.test_comb, sweep_step, smoothing_scale, peak_threshold, spacing_threshold)
        return

    def plot_sweep(self, bb_freqs, path):
        plot_sweep.plot_trace(bb_freqs, path)
        return

    def plot_kids(self, save_path = None, bb_freqs = None, channels = None):
        plt.ion()
        plt.figure(1)
        plt.clf()
        lo_freqs, Is, Qs = self.open_stored(save_path)
        #[ plt.plot((sweep_freqs[2:] + bb_freqs[chan])/1.0e9,10*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2))) for chan in channels]
        mags = np.zeros((channels,len(lo_freqs)))
        scaled_mags = np.zeros((channels,len(lo_freqs)))
        for chan in range(channels):
                mags[chan] = 20*np.log10(np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2))
        #for chan in range(channels):
        #	diff = 0. - np.mean(mags[chan])
        #	scaled_mags[chan] = diff + mags[chan]
                #plt.plot((lo_freqs + bb_freqs[chan])/1.0e9,scaled_mags[chan])
                plt.plot((self.sweep_freqs + bb_freqs[chan])/1.0e9,mags[chan])
        plt.ylim((np.min(mags), np.max(mags)))
        plt.xlabel('frequency (GHz)')
        plt.ylabel('[dB]')
        #plt.savefig(os.path.join(save_path,'fig.png'))
        plt.show()
        return

    def get_stream(self, chan, time_interval):
        #self.fpga.write_int('GbE_pps_start', 1)
        #self.phases = np.empty((len(self.freqs),Npackets))
        Npackets = np.int(time_interval * self.accum_freq)
        Is = np.empty(Npackets)
        Qs = np.empty(Npackets)
        phases = np.empty(Npackets)
        count = 0
        while count < Npackets:
                packet = self.s.recv(8192 + 42) # total number of bytes including 42 byte header
                data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                #ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
                # To stream one channel, make chan an argument
                if (chan % 2) > 0:
                    I = data[1024 + ((chan - 1) / 2)]    
                    Q = data[1536 + ((chan - 1) /2)]    
                else:
                    I = data[0 + (chan/2)]    
                    Q = data[512 + (chan/2)]    
                phase = np.arctan2([Q],[I])
                Is[count]=I
                Qs[count]=Q
                phases[count]=phase
                count += 1
        return Is, Qs, phases

    def get_stream(self, chan, time_interval):
        self.fpga.write_int('GbE_pps_start', 1)
        #self.phases = np.empty((len(self.freqs),Npackets))
        #save_path = raw_input('Absolute save path (e.g. /home/olimpo/home/data/python_psd/ctime) ')
        #os.mkdir(save_path)
        Npackets = np.int(time_interval * self.accum_freq)
        Is = np.empty(Npackets)
        Qs = np.empty(Npackets)
        phases = np.empty(Npackets)
        count = 0
        while count < Npackets:
                packet = self.s.recv(8192 + 42) # total number of bytes including 42 byte header
                data = np.fromstring(packet[42:],dtype = '<i').astype('float')
                #ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
                #To stream one channel, make chan an argument
                if (chan % 2) > 0:
                    I = data[1024 + ((chan - 1) / 2)]    
                    Q = data[1536 + ((chan - 1) /2)]    
                else:
                    I = data[0 + (chan/2)]    
                    Q = data[512 + (chan/2)]    
                phase = np.arctan2([Q],[I])
                Is[count]=I
                Qs[count]=Q
                phases[count]=phase
                count += 1
        #np.save(save_path + '/phase_vals.npy', phases)
        #np.save(save_path + '/single_tone_I.npy', Is)
        #np.save(save_path + '/single_tone_Q.npy', Qs)
        return Is, Qs, phases

    def plotPSD(self, chan, time_interval):
        LO_freq = (self.center_freq/self.divconst)*1.0e6
        Npackets = np.int(time_interval * self.accum_freq)
        plot_range = (Npackets / 2) + 1
        figure = plt.figure(num= None, figsize=(20,12), dpi=100, facecolor='w', edgecolor='w')
        plt.suptitle('Chan ' + str(chan) + ' phase PSD')
        ax = figure.add_subplot(1,1,1)
        ax.set_xscale('log')
        ax.set_ylim((-150,-65))
        #plt.xlim((0.0001, self.accum_freq/2.))
        ax.set_ylabel('dBc/Hz', size = 20)
        ax.set_xlabel('log Hz', size = 20)
        plt.grid()
        Is, Qs, phases = self.get_stream(chan, time_interval)
        phase_mags = np.fft.rfft(phases)
        phase_vals = (np.abs(phase_mags)**2 * ((1./self.accum_freq)**2 / (time_interval)))
        #phase_vals = (np.abs(phase_mags)**2 )
        phase_vals = 10*np.log10(phase_vals)
        phase_vals -= phase_vals[0]
        #phase_vals = signal.convolve(phase_vals,np.hamming(20), mode = 'same')
        ax.plot(np.linspace(0, self.accum_freq/2., (Npackets/2) + 1), phase_vals, color = 'black', linewidth = 1)
        ax.axhline(10*np.log10(6.4e-8), linestyle = '--', c = 'g')
        plt.show()
        return

    def stream_and_save(self, time_interval, LO_freq, save_path, skip_packets=0, channels = None):
        Npackets = np.int(time_interval * self.accum_freq)
        I_buffer = np.empty((Npackets + skip_packets, len(channels)))
        Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
        #self.fpga.write_int('GbE_pps_start', 1)
        count = 0
        while count < Npackets + skip_packets:
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            odd_chan = channels[1::2]
            even_chan = channels[0::2]
            I_odd = data[1024 + ((odd_chan - 1) / 2)]    
            Q_odd = data[1536 + ((odd_chan - 1) /2)]    
            I_even = data[0 + (even_chan/2)]    
            Q_even = data[512 + (even_chan/2)]    
            even_phase = np.arctan2(Q_even,I_even)
            odd_phase = np.arctan2(Q_odd,I_odd)
            if len(channels) % 2 > 0:
                I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
                Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
                I = np.hstack((I, I_even[-1]))    
                Q = np.hstack((Q, Q_even[-1]))    
                I_buffer[count] = I
                Q_buffer[count] = Q
            else:
                I = np.hstack(zip(I_even, I_odd))
                Q = np.hstack(zip(Q_even, Q_odd))
                I_buffer[count] = I
                Q_buffer[count] = Q
            count += 1
        I_file = 'I' + str(LO_freq)
        Q_file = 'Q' + str(LO_freq)
        np.save(os.path.join(save_path,I_file), np.mean(I_buffer[skip_packets:], axis = 0)) 
        np.save(os.path.join(save_path,Q_file), np.mean(Q_buffer[skip_packets:], axis = 0)) 
        return 




    def make_format_complex(self):
        formatname = self.datadir+'/format_complex_extra'
        I_center = self.centers.real
        Q_center = self.centers.imag
        cosi = np.cos(-self.rotations)
        sini = np.sin(-self.rotations)	

        print "saving format_complex_extra in ", formatname
#	ftrunc = np.hstack(freqs.astype(int))
        format_file = open(formatname, 'w')
        for i in range(len(self.radii)):

                format_file.write( 'chC_'+str(i).zfill(3)+' LINCOM   ch_'+str(i).zfill(3)+' 1  '+str(-I_center[i])+';'+str(-Q_center[i])+' # centered \n')
                format_file.write('chCR_'+str(i).zfill(3)+' LINCOM  chC_'+str(i).zfill(3)+' '+str(cosi[i])+';'+str(sini[i])+'   0  # centered and rotated \n') 
                format_file.write('chCr_'+str(i).zfill(3)+' LINCOM chCR_'+str(i).zfill(3)+' '+str(1/self.radii[i])+'   0 #centered, rotated and scaled   \n')
                format_file.write( 'chi_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.r   0  # I centered \n')
                format_file.write( 'chq_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.i   0  # Q centered \n')
                format_file.write( 'chp_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.a   0  # Phase \n')
                format_file.write( 'cha_'+str(i).zfill(3)+' PHASE  '+'chCr_'+str(i).zfill(3)+'.m   0  # Magnitude \n \n')

        format_file.close()
        return


    def make_format(self, path_current = False):
        if path_current:
                formatname = self.datadir+'/format_extra'
                freqs = self.cold_array_bb/1.e6+self.center_freq/self.divconst
        else:
                file_resonances = raw_input('Absolute path to a list of resonances basebands (e.g. /home/data/olimpo/setup/kids/sweeps/target/current/bb_freqs.npy) ? ')

                freqs = np.load(file_resonances)/1.e6+self.center_freq/self.divconst
                folder_dirfile = raw_input('Dirfile folder (e.g. /home/data/olimpo/data_logger/log_kids/) ? ')
                formatname = os.path.join(folder_dirfile,'format_extra')

        print "saving freqs format in ", formatname
        ftrunc = np.hstack(freqs.astype(int))
        format_file = open(formatname, 'w')
        for i in range(len(freqs)):
                decimal = int(freqs[i]*1000 % ftrunc[i])
                format_file.write('/ALIAS  KID_'+str(ftrunc[i])+'_'+str(decimal).zfill(3)+' chQ_'+str(i).zfill(3)+'  \n'   )

        format_file.close()



    def double_freqs(self):
        self.do_double = 1 
        self.centers=np.repeat(self.centers,2)
        self.rotations=np.repeat(self.rotations,2)
        self.radii=np.repeat(self.radii,2)
        self.phases=np.repeat(self.phases,2)
        self.cold_array_rf=np.sort(np.concatenate((self.cold_array_rf, self.cold_array_rf+self.shift_freq)))
        self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/self.divconst))*1.0e6

        print "Setting frequencies to the located values "
        print len(self.cold_array_rf), len(self.radii), len(self.amps)

        time.sleep(0.7)
        if self.do_transf == True:
              self.writeQDR(self.cold_array_bb, transfunc = True)
        else:
              self.writeQDR(self.cold_array_bb) 


        self.make_format(path_current = False)



    def programLO(self, freq=200.0e6, sweep_freq=0):
        self.vi.simple_set_freq(0,freq)
        return

    def menu(self,prompt,options):
        print '\t' + prompt + '\n'
        for i in range(len(options)):
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
        opt = input()
        return opt


    def main_auto(self):
        if ("nofpga" not in sys.argv):
             sys.stdout.write("Uploading firmware to IP %s \n" % self.ip) 
             self.upload_fpg()
             sys.stdout.write("done \n")
             time.sleep(2)
             sys.stdout.write("Start RO inizialization...")
#             self.initialize()
        else:
             sys.stdout.write( "No FPGA firmware being uploaded!")


        # 0)
        time.sleep(.3)
        self.initialize()
        time.sleep(.3)
        # 8)
        if ('current' not in sys.argv):
                self.target_sweep(sweep = True, olimpo=True)

                print "Setting frequencies to the located values "
                time.sleep(0.7)
                if self.do_transf == True:
                        self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
                        self.writeQDR(self.cold_array_bb)
                time.sleep(1)
        else:		
                self.path_configuration = '/data/mistral/setup/kids/sweeps/target/current/' #olimpo/setup/kids/sweeps/target/current/'
                self.array_configuration()
                if self.do_transf == True:
                        self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
                        self.writeQDR(self.cold_array_bb)
                time.sleep(1)


        # 15)
        print "Using array configuration from" , self.path_configuration
        nchannel=len(self.radii)
        print "nchannles = ", nchannel

        try:
                self.dirfile_complex(nchannel)
        except KeyboardInterrupt:
                pass 





    def main_auto_old(self):
        if ("nofpga" not in sys.argv):
             sys.stdout.write("Uploading firmware to IP %s \n" % self.ip) 
             self.upload_fpg()
             sys.stdout.write("done \n")
             time.sleep(2)
             sys.stdout.write("Start RO inizialization...")
             self.initialize()
        else:
             sys.stdout.write( "No FPGA firmware being uploaded!")
        if("search" in sys.argv):
             sys.stdout.write("Searching for resonance...")
             #step 1
             #self.upconvert = np.sort(((self.test_comb + (self.center_freq/2)*1.0e6))/1.0e6)
             self.upconvert =((self.test_comb + (self.center_freq/self.divconst)*1.0e6))/1.0e6
             sys.stdout.write("RF tones = %s \n" %self.upconvert)
             #print "RF tones =", self.upconvert
             self.writeQDR(self.test_comb, transfunc = False)            
             #step 5
             self.vna_sweep(sweep = True,path_current=True, do_plot=False)
             #step 6
             path = '/home/data/olimpo/setup/kids/sweeps/vna/current'
             fk.main(path, savefile = True)
             #step 7
#	     self.global_attenuation = self.global_attenuation/2.0
             self.target_sweep(sweep = True,path_current=True, do_plot=True)
#	     path = '/home/data/olimpo/setup/kids/sweeps/target/current'
#	     print "locate resonances"
#	     fk.main(path, savefile = True)
             #print "second target sweep"
             #self.target_sweep(sweep = True,path_current=True, do_plot=True)
             #fk.main(path, savefile = True)
             try:
                self.cold_array_rf = np.load('/home/data/olimpo/setup/kids/sweeps/target/current/target_freqs_new.npy')
                print "			reads improved centers"
             except IOError:
                self.cold_array_rf = np.load('/home/data/olimpo/setup/kids/sweeps/target/current/target_freqs.npy')

             self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/self.divconst))*1.0e6
#	     self.cold_array_bb = np.roll(self.cold_array_bb, - np.argmin(np.abs(self.cold_array_bb)))
             self.centers = np.load('/home/data/olimpo/setup/kids/sweeps/target/current/centers.npy') 
             self.rotations = np.load('/home/data/olimpo/setup/kids/sweeps/target/current/rotations.npy')
             self.radii = np.load('/home/data/olimpo/setup/kids/sweeps/target/current/radii.npy')
        else:
             if("150" in sys.argv or "480" in sys.argv):
                self.path_configuration='/home/data/olimpo/setup/kids/sweeps/target/default_array150-480/'
             elif("200" in sys.argv or "350" in sys.argv):
                self.path_configuration='/home/data/olimpo/setup/kids/sweeps/target/default_array200-350/'

             elif("current" in sys.argv or "last" in sys.argv):
                self.path_configuration='/home/data/olimpo/setup/kids/sweeps/target/current/'
             self.array_configuration()	     		


        sys.stdout.write("RF tones = %s\n " %self.cold_array_rf)
        sys.stdout.write("BB tones = %s\n " %(self.cold_array_bb/1.0e6))
        if self.do_transf == True:
                self.writeQDR(self.cold_array_bb, transfunc = True)
        else:
                self.writeQDR(self.cold_array_bb)

        nchannel=len(self.cold_array_bb)
        sys.stdout.write("          nchannels = %s \n" %nchannel)

        self.make_format(path_current = True)
        try:
                self.v1.set_frequency(2,self.center_freq,0.01)  # move the LO by 15 kHz. RIMOSSO PER IL MOMENTO ELIA. 22/1/17  
                self.dirfile_all_chan_phase_centered(nchannel)
#		self.dirfile_all_chan(nchannel)
        except KeyboardInterrupt:
                pass						 





    def main_opt(self):
        while True:
            opt = self.menu(self.main_prompt,self.main_opts)
            if opt == 0:
                os.system('clear')
                self.initialize() 
            if opt == 1:
                #self.upconvert = np.sort(((self.test_comb + (self.center_freq/2)*1.0e6))/1.0e6)
                self.test_comb_flag = True

                self.upconvert = ((self.test_comb + (self.center_freq/self.divconst)*1.0e6))/1.0e6
                print "RF tones =", self.upconvert
#		print "test_comb = ", self.test_comb
                self.writeQDR(self.test_comb, transfunc = False)


                #prompt = raw_input('Apply inverse transfer function? (y/n)')
                #if prompt == 'n':
                #	self.writeQDR(self.test_comb, transfunc = False)
                #if prompt == 'y':
                #	self.writeQDR(self.test_comb, transfunc = False)
                #	time.sleep(15)
                #	self.writeQDR(self.test_comb, transfunc = True)
            if opt == 2:
                file_path = raw_input('Absolute path to .npy file (list of baseband frequencies in any order, e.g. /home/data/olimpo/setup/kids/sweeps/target/1601021_01/bb_freqs.npy): ' )
                self.cold_array_bb = np.load(file_path)
                self.cold_array_bb = self.cold_array_bb[self.cold_array_bb != 0]
                #freqs = np.roll(freqs, - np.argmin(np.abs(freqs)))
                #rf_tones = np.sort((self.cold_array_bb + ((self.center_freq/2)*1.0e6))/1.0e6)
                rf_tones = (self.cold_array_bb + ((self.center_freq/self.divconst)*1.0e6))/1.0e6
                print "BB tones =", self.cold_array_bb/1.e6
                print "RF tones (right order) =", rf_tones

            if opt == 3:

                path = raw_input('Absolute path to a folder with RF freqs ( e.g. /data/mistral/setup/kids/sweeps/target/current/ ): ' )
                self.path_configuration=path
                self.array_configuration()
#		try:
#			self.cold_array_rf = np.load(path+'/target_freqs_new.npy')
#		except IOError:
#			self.cold_array_rf = np.load(path+'/target_freqs.npy')
#
#		self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)/2.))*1.0e6
##                       self.cold_array_bb = np.roll(self.cold_array_bb, - np.argmin(np.abs(self.cold_array_bb)))
#                self.centers = np.load( path+ '/centers.npy')
#		self.rotations = np.load(path+'/rotations.npy')
#		self.radii = np.load( path+   '/radii.npy')

#		self.make_format(path_current=True)

                if self.do_transf == True:
                      self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
                      self.writeQDR(self.cold_array_bb)


            #if opt == 3:
        #	print "RF tones =", self.cold_array_rf
        #	print "BB tones =", self.cold_array_bb/1.0e6
        #    	self.writeQDR(self.cold_array_bb)
            if opt == 4:
                Npackets = input('\nNumber of UDP packets to stream? ' )
                chan = input('chan = ? ')
                self.stream_UDP(chan,Npackets)
            if opt == 5:
                prompt = raw_input('Do sweep? (y/n) ')
                if prompt == 'y':
                        self.vna_sweep(sweep = True)
                else:
                        self.vna_sweep()
            if opt == 6:
                path = raw_input("Absolute data path to a good VNA sweep dir (e.g. /home/data/olimpo/setup/kids/sweeps/vna/161021_01: ")
                fk.main(path, savefile=True)


            if opt == 7:
                prompt = raw_input('Do sweep? (y/n) ')
                if prompt == 'y':
                        self.target_sweep(sweep = True)
                else:
                        self.target_sweep()

                print "Setting frequencies to the located values "
                time.sleep(0.7)
                if self.do_transf == True:
                      self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
                      self.writeQDR(self.cold_array_bb)

            if opt == 8:
                self.target_sweep(sweep = True, olimpo=True)

                print "Setting frequencies to the located values "
                time.sleep(0.7)
                if self.do_transf == True:
                      self.writeQDR(self.cold_array_bb, transfunc = True)
                else:
                      self.writeQDR(self.cold_array_bb)







#            if opt == 8:
#	    	chan = input('Channel number = ? ')
#		time_interval = input('Time interval (s) ? ')
#		self.plotPSD(chan, time_interval)
            if opt == 9:

                nchannel=len(self.cold_array_bb)	
                print "nchannles = ", nchannel
                try:
                        self.dirfile_all_chan(nchannel)
                except KeyboardInterrupt:
                        pass 

            if opt == 10:

                if self.path_configuration=='':
                        print "Array configuration (freqs, centers, radii and rotations) undefined"
                        self.path_configuration = raw_input("Absolute path to a folder with freqs, centers, radii and rotations (e.g. /home/data/olimpo/setup/kids/sweeps/target/default_array150-480 )")
                        self.array_configuration()
                else: 
                        print "Using array configuration from" , self.path_configuration

                nchannel=len(self.radii)
                #nchannel=input('number of channels?')#aggiunto 09082017 AP
                print "nchannles = ", nchannel

                try:
                        self.dirfile_all_chan_phase_centered(nchannel)
        #		self.dirfile_phase_centered(time_interval)
                except KeyboardInterrupt:
                        pass 
            if opt == 11:
                time_interval = input('Time interval (s) ? ')
                try:
                        self.dirfile_avg_chan(time_interval)	
                except KeyboardInterrupt:
                        pass 
            if opt == 12:
                self.global_attenuation=input("Insert global attenuation (decimal, <1.0, e.g 0.01)")

            if opt == 13:
                self.path_configuration = raw_input("Absolute path to a folder with freqs, centers, radii and rotations (e.g. /home/data/olimpo/setup/kids/sweeps/target/default_array150-480 )")
                self.array_configuration()
            if opt == 14:
                self.shift_freq = input("insert freq shift in MHz")
                self.double_freqs()

            if opt == 15:
                if self.path_configuration=='':
                        print "Array configuration (freqs, centers, radii and rotations) undefined"
                        self.path_configuration = raw_input("Absolute path to a folder with freqs, centers, radii and rotations (e.g.  /data/mistral/setup/kids/sweeps/target/current)")
                        self.array_configuration()
                else: 
                        print "Using array configuration from" , self.path_configuration

                nchannel=len(self.radii)
                #nchannel=input('number of channels?')#aggiunto 09082017 AP
                print "nchannles = ", nchannel

                try:
                        self.dirfile_complex(nchannel)
        #		self.dirfile_phase_centered(time_interval)
                except KeyboardInterrupt:
                        pass 


            if opt == 16:
                sys.exit()
        return

    def main(self):
        #os.system('clear')

        sys.stdout.write("Restart dnsmasq \n")
        os.system("/home/olimpo/bin/restart_dnsmasq &")

        if("roach2" in sys.argv):
                self.roach2=True
        else:
                self.roach2=False

        self.path_configuration=''
        if("do_transf" in sys.argv):
                self.do_transf = True
        else:
                self.do_transf = False
        if("norun" not in sys.argv):  
                self.main_auto()	
        else:
                while True: 
                        self.main_opt()




if __name__=='__main__':
        #       sys.stderr=open("/home/olimpo/log/errors.txt")

        ri = roachInterface()
        ri.main()
