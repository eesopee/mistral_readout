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
import variable_attenuator_mistral as vatt

from mistral_pipeline_dev import pipeline
import tqdm


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

		self.divconst = 1. #not really necessary anymore since we only use MISTRAL and not olimpo, that needed double LO
		self.ip = conf.roach_ip
		self.center_freq = conf.LO
		self.global_attenuation= 1.0 #need to implement the class for Arduino variable attenuator!
		self.wavemax = 1.1543e-5 #what is this? Maybe we should increase it for 400 tones.

		self.path_configuration = conf.path_configuration
		'''
		VALON SETUP
		'''
		print("Setting up VALON")

		'''
		For some god forsaken reason, the VALON changes serial port. Here we cycle through the serial ports until it connects and sets the LO frequency correctly.
		'''

		i = 0
		while True:

			try:

				self.valon_port = conf.valon_port + str(i)
				sys.stdout.write("Attempting Valon connection on port "+self.valon_port+"\n")
				self.v1 = valon_synth.Synthesizer(self.valon_port)
				self.v1.set_frequency(2,self.center_freq,0.01)
				self.v1.set_frequency(1, 512.,0.01)
				sys.stdout.write("Success!\n")
				break

			except OSError:

				sys.stdout.write("Failed. Attempting next port\n")
				i +=1

				pass
		sys.stdout.write("Valon connected at port "+self.valon_port+"\n")

		'''
		Arduino attenuator setup
		'''
		i=0
		while True:
			try:
				self.arduino_port = conf.arduino_port + str(i)
				self.att = vatt.Attenuator(self.arduino_port)
				atts = self.att.get_att()
				print("Current attenuation = ", atts)
				sys.stdout.write("Success!\n")
				break
			except:

				sys.stdout.write("Failed. Attenpting next port\n")
				i +=1

				pass
		sys.stdout.write("Attenuator connected at port "+self.arduino_port+"\n")

		#self.nchan = len(self.raw_chan[0])

		sys.stdout.write("Activating ROACH interface\n")
		self.fpga = casperfpga.katcp_fpga.KatcpFpga(self.ip,timeout=120.)
		self.bitstream = "olimpo_firmware_current.fpg"   # October 2017 from Sam Gordon
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

	def lpf(self, zeros):
		zeros *=(2**31 - 1)
		for i in range(len(zeros)/2 + 1):
		    coeff = np.binary_repr(int(zeros[i]), 32)
		    coeff = int(coeff, 2)
		    #print 'h' + str(i), coeff
		    self.fpga.write_int('FIR_h'+str(i),coeff)
		return

	def qdrCal(self):

   		# Calibrates the QDRs. Run after writing to QDR.

		self.fpga.write_int('dac_reset',1)

		print 'DAC on'

		bFailHard = False
		calVerbosity = 0
		qdrMemName = 'qdr0_memory'
		qdrNames = ['qdr0_memory','qdr1_memory']

		print 'Fpga Clock Rate =',self.fpga.estimate_fpga_clock()

		self.fpga.get_system_information()
		results = {}

		for qdr in self.fpga.qdrs:
			#print qdr
			mqdr = myQdr.from_qdr(qdr)
			results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
		print 'qdr cal results:',results

		for qdrName in ['qdr0','qdr1']:
		    	if not results[qdr.name]:
		        	print 'Calibration Failed'
		        	break

	def initialize_GbE(self):
		# Configure GbE Block. Run immediately after calibrating QDR.
		self.fpga.write_int('GbE_tx_rst',0)
		self.fpga.write_int('GbE_tx_rst',1)
		self.fpga.write_int('GbE_tx_rst',0)
		return

    	def initialize(self):


		'''
		Parameters for the low pass filter
		'''

	    	self.zeros = signal.firwin(27, 1.5e6, window='hanning',nyq = 128.0e6)
		self.zeros = signal.firwin(29, 1.5e3, window='hanning',nyq = 128.0e6)
		self.zeros = self.zeros[1:-1]

		self.bin_fs = 256.0e6
	  	self.zeros = signal.firwin(23, 10.0e3, window='hanning',nyq = 0.5*self.bin_fs)   # for olimpo_firmaware_dds318.fpg

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
		self.qdrCal()
		self.initialize_GbE()

		print '\nQDR Calibrated'
		print 'Packet streaming activated\n'

	def make_format(self, path_current = False):

		if path_current == True:
			formatname = self.datadir+'/format_extra'
			freqs = self.cold_array_bb/1.e6+self.center_freq
		else:
			file_resonances = raw_input('Absolute path to a list of resonances basebands (e.g. /home/data/mistral/setup/kids/sweeps/target/current/bb_freqs.npy) ? ')
			freqs = np.load(file_resonances)/1.e6+self.center_freq/self.divconst
			folder_dirfile = raw_input('Dirfile folder (e.g. /home/data/mistral/data_logger/log_kids/) ? ')
			formatname = os.path.join(folder_dirfile,'format_extra')

		print "saving freqs format in ", formatname
		ftrunc = np.hstack(freqs.astype(int))
		format_file = open(formatname, 'w')

		'''
		Do we need these aliases?
		'''

		print("Making format file for", len(freqs), " resonances")
		for i in range(len(freqs)):
			decimal = int(freqs[i]*1000 % ftrunc[i])
			format_file.write('/ALIAS  KID_'+str(ftrunc[i])+'_'+str(decimal).zfill(3)+' chQ_'+str(i).zfill(3)+'  \n'   )

		format_file.close()

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


	def array_configuration(self, path):

		sys.stdout.write("setting freqs, centers, radii and rotations from %s \n " %path)

		'''
		If it exists, it opens target_freqs_new which has the "tuned" frequencies from the target sweep. Otherwise, it opens the default frequencies.
		'''

		try:

			self.cold_array_rf = np.loadtxt(path+'/target_freqs_new.dat')
			print("Loading target_freqs_new.dat")

		except IOError:

			print("Loading target_freqs.dat")
			self.cold_array_rf = np.loadtxt(path+'/target_freqs.dat')

		self.cold_array_bb = (((self.cold_array_rf) - (self.center_freq)))*1.0e6
		self.centers = np.load(path+'/centers.npy')
		self.rotations = np.load(path+'/rotations.npy')
		self.radii = np.load(path+'/radii.npy')

		sys.stdout.write( "reading freqs, centers, rotations and radii from %s\n" %path)

		#print 'radii', self.radii

		self.make_format(path_current = True)
		return

		'''
		Devi aggiungere la cartella CURRENT nel VNA
		'''

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

		'''
		Builds the DDS look-up-table from I and Q given by freq_comb.
		freq_comb is called with the sample rate equal to the sample rate for a single FFT bin.
		There are two bins returned for every fpga clock, so the bin sample rate is 256 MHz / half the fft length
		'''

		self.select_bins(freqs)
		I_dds, Q_dds = np.array([0.]*(self.LUTbuffer_len)), np.array([0.]*(self.LUTbuffer_len))

		for m in range(len(self.freq_residuals)):
			I, Q = self.freq_comb(np.array([self.freq_residuals[m]]), self.fpga_samp_freq/(self.fft_len/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
			I_dds[m::self.fft_len] = I
			Q_dds[m::self.fft_len] = Q

		return I_dds, Q_dds

	def fft_bin_index(self, freqs, fft_len, samp_freq):
	    	# returns the fft bin index for a given frequency, fft length, and sample frequency
		bin_index = np.round((freqs/samp_freq)*fft_len).astype('int')
		return bin_index

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
		
	def calc_transfunc(self, freqs):
		#print("Calculating attenuations for target freqs:", freqs)
		transfunc_parameters_file = conf.transfer_function_file

		poly_par = np.load(transfunc_parameters_file)

		baseline_attenuation = conf.baseline_attenuation

		attenuations = np.poly1d(poly_par)(freqs)
		#print("prima di 10**")
		#print(attenuations)
		attenuations = 10.**((baseline_attenuation-attenuations)/20.)


		#print(attenuations)
		bad_attenuations = np.argwhere(np.floor(attenuations))
		print("Checking for bad attenuations")
		print(bad_attenuations)
		if bad_attenuations.size > 0:
		    print("Warning: invalid attenuations found. Setting them to 1.0 by default.")
		    attenuations[bad_attenuations] = 1.0

		return attenuations

	def freq_comb(self, freqs, samp_freq, resolution, random_phase = True, DAC_LUT = True, apply_transfunc = False):
	    	# Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q
		freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res

		amp_full_scale = (2**15 - 1)

		if DAC_LUT == True:

			print 'freq comb uses DAC_LUT'
			fft_len = self.LUTbuffer_len
			bins = self.fft_bin_index(freqs, fft_len, samp_freq)
			phase = self.phases[0:len(bins)]

			if apply_transfunc:
				self.amps = self.calc_transfunc(self.center_freq+freqs*1e-6) # qui
			else:
				self.amps = np.ones(len(freqs))

			print self.amps

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

		else: #do we need this?

			fft_len = (self.LUTbuffer_len/self.fft_len)
			bins = self.fft_bin_index(freqs, fft_len, samp_freq)
			spec = np.zeros(fft_len,dtype='complex')
			amps = np.array([1.]*len(bins))
			phase = 0.
			spec[bins] = amps*np.exp(1j*(phase))
			wave = np.fft.ifft(spec)
			waveMax = np.max(np.abs(wave))
			I = (wave.real/waveMax)*(amp_full_scale)
			Q = (wave.imag/waveMax)*(amp_full_scale)

		return I, Q


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

		'''
		Writes packet LUTs to QDR
		'''

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


    	def store_UDP(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):

		channels = np.arange(channels)
		I_buffer = np.empty((Npackets + skip_packets, len(channels)))
		Q_buffer = np.empty((Npackets + skip_packets, len(channels)))

		count = 0

		while count < Npackets + skip_packets:

		    packet = self.s.recv(8234)# total number of bytes including 42 byte header

		    if(len(packet) == 8234):

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


	def target_sweep(self,  path_current = False, do_plot=True, olimpo=False):

		'''
		Function used for tuning. It spans a small range of frequencies around known resonances.
		'''

		target_path = self.setupdir+'sweeps/target/'
		center_freq = self.center_freq*1.e6

		if path_current:
			vna_path = self.setupdir+'sweeps/target/current' #era VNA. GLi facciamo prendere l'ultimo target sweep.
			sweep_dir = self.timestring
		else:

		    print "roach415: /home/mistral/src/parameters/roach415"
		    print "current: /data/mistral/setup/kids/sweeps/target/current"
		    print "roach_test: /home/mistral/src/parameters/new_client"
		    print "roach_test_destination: /data/mistral/setup/kids/sweeps/target"
		    vna_path = raw_input('Absolute path to VNA sweep dir ? ')

		    self.timestring = "%04d%02d%02d_%02d%02d%02d" % (time.localtime()[0],time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5])
		    #self.add_out_of_res(vna_path + "/target_freqs.dat", self.out_of_res_tones)


		    sweep_dir = raw_input('Insert new target sweep subdir to '+self.setupdir+ '/sweeps/target/ (eg. '+self.timestring+') Press enter for defualt:')
		    if(sweep_dir)=='': sweep_dir=self.timestring

		self.target_freqs = np.loadtxt(os.path.join(vna_path, 'target_freqs.dat'), unpack=True) #target_freqs.dat ha SOLO le frequenze, le amplificazioni vengono calcolate
		print("TARGET FREQS = ", self.target_freqs)
		self.calc_transfunc(self.target_freqs)

		'''
		Qui chiama una funzione che calcola le amps, da risistemare!!!
		'''

		save_path = os.path.join(target_path, sweep_dir)
		self.path_configuration = save_path

		try:
			os.mkdir(save_path)
		except OSError:
		        pass

		command_cleanlink = "rm -f " + target_path + "current"
		os.system(command_cleanlink)

		command_linkfile = "ln -f -s " + save_path +" "+ target_path+'current'
		os.system(command_linkfile)

		#np.save(save_path + '/target_freqs.npy', self.target_freqs)
		np.savetxt(save_path + '/target_freqs.dat', self.target_freqs)
		
		self.bb_target_freqs = ((self.target_freqs*1.0e6) - center_freq)
		
		upconvert = (self.bb_target_freqs + center_freq)/1.0e6
		print "RF tones =", upconvert
		self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO
		print '\nTarget baseband freqs (MHz) =', self.bb_target_freqs/1.0e6
		span =conf.sweep_span #200.0e3   #era 400.e3             # era 1000.e3 #era 400.e3 20170803
		start = center_freq - (span)  # era (span/2)
		stop = center_freq + (span)   # era (span/2)
		step = conf.sweep_step #1.25e3 * 2.                 # era senza
		sweep_freqs = np.arange(start, stop, step)
		sweep_freqs = np.round(sweep_freqs/step)*step
		print "LO freqs =", sweep_freqs

		np.save(save_path + '/bb_freqs.npy',self.bb_target_freqs)
		np.save(save_path + '/sweep_freqs.npy',sweep_freqs)

		self.writeQDR(self.bb_target_freqs, transfunc=True)

		for freq in tqdm.tqdm(sweep_freqs):
			if self.v1.set_frequency(2, freq/1.0e6, 0.01):
				self.store_UDP(100,freq,save_path,channels=len(self.bb_target_freqs))
				self.v1.set_frequency(2,center_freq / (1.0e6), 0.01) # LO

		return save_path

	def run_pipeline(self, path, do_plot=False):
		print(type(path))
		print "pipline.py su " + path
		self.path_configuration = path
		pipeline(path) #locate centers, rotations, and resonances
		
		if do_plot:
			self.plot_targ(path)

		self.array_configuration(path)#includes make_format
