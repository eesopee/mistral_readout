from pathlib import Path

LO =  435. #local oscillator frequency in MHz

sweep_step = 1.25e3 #step for the target sweep in Hz. Min=1.25 kHz. 
sweep_span = 250.e3 #half span of the target sweep i.e. it goes from target-span to target+span
sweep_offset = 0.0 #frequency offset for the target sweep.

roach_ip = '192.168.41.40' #ip of the Roach. Verify it with $arp0

data_socket = Path("enp1s0f1") #data socket. NOT THE PPC SOCKET

# default arduino attenuators values
att_RFOUT = 20.0 #dB
att_RFIN = 0.0 #dB

skip_tones_attenuation = 10 #It will not calculate the attenuations for the first 5 target freqs in the target_freqs.dat file. Set to 0 to 

#valon_port = Path("/dev/ttyUSB") #port for the valon. 
#arduino_port = Path("/dev/ttyACM") #Port for the Arduino variable attenuator

baseline_attenuation = -44.8  #dBm

datadir = Path("/data/mistral/data_logger/log_kids/") #directory where dirfiles are saved
setupdir = Path("/data/mistral/setup/kids/") #directory where array configurations are saved 
path_configuration = Path("/data/mistral/setup/kids/sweeps/target/current/") #default configuration file. It is the current folder, with the latest array configuration.
transfer_function_file = Path("/home/mistral/src/mistral_readout_dev/transfunc_polyfit_coefficients.npy") #transfer function file
folder_frequencies = Path("/home/mistral/src/parameters/OLIMPO_150_480_CURRENT/") #update to MISTRAL and not olimpo. Serve la cartella CURRENT!
log_dir = Path("/home/mistral/src/client_logs/")
