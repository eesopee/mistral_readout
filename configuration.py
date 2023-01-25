from pathlib import Path

LO = 435. #local oscillator frequency in MHz

sweep_step = 2.5e3 #step for the target sweep in Hz. Min=1.25 kHz. 
sweep_span = 100.e3 #half span of the target sweep i.e. it goes from target-span to target+span

roach_ip = '192.168.41.40' #ip of the Roach. Verify it with $arp0

data_socket = Path("enp1s0f1") #data socket. NOT THE PPC SOCKET

# default arduino attenuators values
att_RFOUT = 8. #dB
att_RFIN = 10. #dB


#valon_port = Path("/dev/ttyUSB") #port for the valon. 
#arduino_port = Path("/dev/ttyACM") #Port for the Arduino variable attenuator

baseline_attenuation = -44.8  #dBm

datadir = Path("/data/mistral/data_logger/log_kids/") #directory where dirfiles are saved
setupdir = Path("/data/mistral/setup/kids/") #directory where array configurations are saved 
path_configuration = Path("/data/mistral/setup/kids/sweeps/target/current/") #default configuration file. It is the current folder, with the latest array configuration.
transfer_function_file = Path("/home/mistral/src/mistral_readout_dev/transfunc_polyfit_coefficients.npy") #transfer function file
folder_frequencies = Path("/home/mistral/src/parameters/OLIMPO_150_480_CURRENT/") #update to MISTRAL and not olimpo. Serve la cartella CURRENT!
