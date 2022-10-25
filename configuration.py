LO = 450 #local oscillator frequency in MHz

sweep_step = 5e3 #step for the target sweep in Hz. Min=1.25 kHz. 
sweep_span = 50e3 #half span of the target sweep i.e. it goes from target-span to target+span

roach_ip = '192.168.41.40' #ip of the Roach. Verify it with $arp0

data_socket = 'enp1s0f1' #data socket. NOT THE PPC SOCKET
valon_port = "/dev/ttyUSB" #port for the valon. 
arduino_port = "/dev/ttyACM" #Port for the Arduino variable attenuator

baseline_attenuation = -44.8  #dBm

datadir = '/data/mistral/data_logger/log_kids/' #directory where dirfiles are saved
setupdir = '/data/mistral/setup/kids/' #directory where array configurations are saved 
path_configuration = '/data/mistral/setup/kids/sweeps/target/current/' #default configuration file. It is the current folder, with the latest array configuration.
transfer_function_file = "/home/mistral/src/mistral_readout_dev/transfunc_polyfit_coefficients.npy" #transfer function file
folder_frequencies='/home/mistral/src/parameters/OLIMPO_150_480_CURRENT/' #update to MISTRAL and not olimpo. Serve la cartella CURRENT!


