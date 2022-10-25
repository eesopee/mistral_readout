class Console_interface(object):
	def __init__(self):
		self.state = 'Arizona'	
	def mk_menu(self,prompt,options):
		print '\t' + prompt + '\n'
		for i in range(len(options)):
			print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
		opt = input()
		return opt
