import serial
import time

class Attenuator:
	def __init__(self, port = "/dev/ttyACM0"): #default port
		self.conn = serial.Serial(None,
					9600, 
					serial.EIGHTBITS, 
					serial.PARITY_NONE, 
					serial.STOPBITS_ONE,
					timeout=0.5)
					
		self.conn.setPort(port)
		
		#print("Connection succesful at port",self.conn.name)
	
		self.conn.open()
		self.conn.write("2\n10\n")
		time.sleep(0.5)
		self.conn.close()
		
	def get_att(self):
		
		if self.conn.isOpen() == False:
			self.conn.open()
			time.sleep(0.5) #for some reason opening the port takes some time
		
		self.conn.write("0\n") #sending 1 to the arduino asks for the attenuation values
		time.sleep(0.1)
		att_values = self.conn.read(1000) #reading att values
		att_values = att_values.split(",")
		att1 = att_values[0]
		att2 = att_values[1]

		return float(att1),float(att2)
		
	def set_att(self,channel,attenuation):
		
		if self.conn.isOpen() == False:
			self.conn.open()
			time.sleep(0.5)
		
		if (channel == 0):
			print("WARNING: ch0 reads att values. Printing values")
			self.get_att()
	
		
		if (attenuation > 31.75):
			print("WARNING: attenuation can't be larger than 31.75 dB. Setting to max value.")
		if (attenuation < 0):
			print("WARNING: attenuation can't be negative. Setting to 0")
		
		data = str(channel)+"\n"+str(attenuation)+"\n"
		self.conn.write(data)
		time.sleep(0.5)
		result = self.conn.read(1000)
		
		self.conn.close()
		
		return str(result.split("\r")[0])
