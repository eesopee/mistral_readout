import os,sys
import numpy as np
import scipy.stats as stats
from scipy.signal import argrelextrema
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal
from scipy import interpolate
import time

y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

class pipeline(object):
    def __init__(self, targ_path=""):
	t_init = time.time()
	self.targ_path = targ_path
	if(targ_path == ""):

	######## Target Sweeps ############
	       	self.targ_path = raw_input('Absolute path to known good target sweep dir (e.g. /data/mistral/setup/kids/sweeps/target/current): ' )

	data_files=[f for f in sorted(os.listdir(self.targ_path)) if f.endswith('.npy')]
        I = np.array([np.load(os.path.join(self.targ_path,f)) for f in data_files if f.startswith('I')])
        Q = np.array([np.load(os.path.join(self.targ_path,f)) for f in data_files if f.startswith('Q')])
	
        self.lo_freqs = np.loadtxt(self.targ_path + '/sweep_freqs.dat')
        #self.target_freqs = np.load(self.targ_path + '/target_freqs.npy')
	self.target_freqs = np.loadtxt(self.targ_path + '/target_freqs.dat')        
	self.raw_chan = I + 1j*Q
        
	#self.chan_ts /= (2**15 - 1)
	#self.chan_ts /= ((2**21 - 1) / 2048)
	
	self.nchan = len(self.raw_chan[0])
        self.cm = plt.cm.Spectral(np.linspace(0.05,0.95,self.nchan))
        self.raw_I = I
        self.raw_Q = Q
	
        self.mag = np.abs(self.raw_chan)
	# locate minima
	
	self.target_freqs_out = np.zeros(self.nchan) 

	print self.mag.shape
	nchannels = len(self.mag[0,:])
	n_sweep = len(self.mag[:,0])
	print 'nchannels ', nchannels,'self.nchan ', self.nchan

	self.indexmin = np.zeros(nchannels, dtype = int)
	self.minfreq = np.zeros(nchannels, dtype = float)
	self.minfreqs = np.zeros(nchannels, dtype= float)

	for ii in range(nchannels):
		indexmin = np.argmin(self.mag[:,ii])
		self.freqs = self.lo_freqs + self.target_freqs[ii]
		
		#self.lo_min[ii] = np.average(self.lo_freqs*self.mag[:,ii])/np.sum(self.mag[:,ii])

		target_freq = self.target_freqs[ii]
		
		self.freqs = target_freq + (self.lo_freqs - self.lo_freqs[n_sweep/2])/1e6

		self.minfreq = np.sum(self.mag[:,ii]*self.freqs)/np.sum(self.mag[:,ii])
		
		self.minfreqs[ii] = self.minfreq #tt + (self.minfreq-self.lo_freqs[n_sweep/2])/1e6		
		self.target_freqs_out[ii] = target_freq + (self.lo_freqs[indexmin] - self.lo_freqs[n_sweep/2])/1.e6 #/2
		print ii, indexmin, self.target_freqs[ii] , self.target_freqs_out[ii]
		self.indexmin[ii] = indexmin

        self.phase = np.angle(self.raw_chan)
        self.centers=self.loop_centers(self.raw_chan) # returns self.centers
	self.chan_centered = self.raw_chan - self.centers
	print self.raw_chan.shape
#       self.rotations = np.angle(self.chan_centered[self.chan_centered.shape[0]/2])
#	self.radii = np.absolute(self.chan_centered[self.chan_centered.shape[0]/2])
	self.rotations = np.zeros(nchannels, dtype = float)
	self.radii = np.zeros(nchannels, dtype = float)
	print self.indexmin.shape 
#	print " ii, self.indexmin[ii], self.target_freqs_out[ii],self.rotations[ii],self.radii[ii], self.centers[ii]  " 
	print 'ii, self.indexmin[ii], self.rotations[ii], np.angle(self.centers[ii]), delta angle'
	for ii in range(nchannels):
		self.rotations[ii] = np.angle(self.chan_centered[self.indexmin[ii],ii])
		self.radii[ii]  = np.absolute(self.chan_centered[self.indexmin[ii],ii])
#		print ii, self.indexmin[ii], self.target_freqs_out[ii],self.rotations[ii],self.radii[ii], self.centers[ii]
		print ii, self.indexmin[ii], np.rad2deg((self.rotations[ii])), np.rad2deg(np.angle(self.centers[ii])), np.rad2deg(self.rotations[ii]-np.angle(self.centers[ii]))

	self.chan_rotated = self.chan_centered * np.exp(-1j*self.rotations)#/self.radii
        self.phase_rotated = np.angle(self.chan_rotated)
        self.bb_freqs = np.loadtxt(self.targ_path + '/bb_freqs.dat')
        
        #self.delta_lo = 2.5e3 #mai utilizzato nel codice
#        prompt = raw_input('Save phase centers and rotations in ' + self.targ_path + ' (**** MAY OVERWRITE ****) (y/n)? ')
#	if prompt == 'y':
	np.save(self.targ_path + '/centers.npy', self.centers)
       	np.save(self.targ_path + '/rotations.npy', self.rotations)
	np.save(self.targ_path + '/radii.npy',self.radii)
	sys.stdout.write("     Saving fine tuned resonances on file\n")	
	#np.save(self.targ_path + '/target_freqs_new.npy', self.target_freqs_out)
	np.savetxt(self.targ_path+"/target_freqs_new.dat", self.target_freqs_out)

	np.save(self.targ_path + '/index_freqs_new.npy', self.indexmin)
        ######## Time streams #################3
	"""
	self.ts_on = np.load(os.path.join(self.datapath,)) + 1j*np.load(os.path.join(self.path + '/timestreams/'))
        self.ts_on_centered = self.ts_on - self.centers
        self.ts_on_rotated = self.ts_on_centered *np.exp(-1j*self.rotations)
        self.i_off, self.q_off = self.ts_off.real, self.ts_off.imag
        self.i_on, self.q_on = self.ts_on_rotated.imag, self.ts_on_rotated.imag
        self.phase_on = np.angle(self.ts_on_rotated)    
    	"""
	print("Execution time=", time.time()-t_init)

    def open_stored(self, save_path = None):
        files = sorted(os.listdir(save_path))
        sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
        I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
        Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
        Is = np.array([np.load(filename) for filename in I_list])
        Qs = np.array([np.load(filename) for filename in Q_list])
        return sweep_freqs, Is, Qs


    def loop_centers(self,timestream):
        def least_sq_circle_fit(chan):
            """
            Least squares fitting of circles to a 2d data set. 
            Calcultes jacobian matrix to speed up scipy.optimize.least_sq. 
            Complements to scipy.org
            Returns the center and radius of the circle ((xc,yc), r)
            """
            #x=self.i[:,chan]
            #y=self.q[:,chan]
            x=timestream[:,chan].real
            y=timestream[:,chan].imag
            xc_guess = x.mean()
            yc_guess = y.mean()
                        
            def calc_radius(xc, yc):
                """ calculate the distance of each data points from the center (xc, yc) """
                return np.sqrt((x-xc)**2 + (y-yc)**2)
    
            def f(c):
                """ calculate f, the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
                Ri = calc_radius(*c)
                return Ri - Ri.mean()
    
            def Df(c):
                """ Jacobian of f.The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
                xc, yc = c
                dfdc = np.empty((len(c), x.size))
                Ri = calc_radius(xc, yc)
                dfdc[0] = (xc - x)/Ri            # dR/dxc
                dfdc[1] = (yc - y)/Ri            # dR/dyc
                dfdc = dfdc - dfdc.mean(axis=1)[:, np.newaxis]
                return dfdc
        
            (xc,yc), success = optimize.leastsq(f, (xc_guess, yc_guess), Dfun=Df, col_deriv=True)
        
            Ri = calc_radius(xc,yc)
            R = Ri.mean()
            residual = sum((Ri - R)**2) #error in fit if needed
            #print xc_guess,yc_guess,xc,yc
            return (xc,yc),R

        centers=[]
        for chan in range(self.nchan):
                (xc,yc),r = least_sq_circle_fit(chan)
                centers.append(xc+1j*yc)
        #self.centers = np.array(centers)
        return np.array(centers)

    def plot_loop_raw(self,chan):
        print self.raw_chan.real[self.indexmin[chan],chan], self.raw_chan.imag[self.indexmin[chan],chan]
        plt.plot(self.raw_chan.real[:,chan],self.raw_chan.imag[:,chan],'x',linestyle='-',color=self.cm[chan])
	#plt.plot(self.centers[0,chan],self.centers[1,chan], 'o', color = 'green')
        plt.axvline(0.0, linestyle='dashed', color='gray', alpha=0.4)
        plt.axhline(0.0, linestyle='dashed', color='gray', alpha=0.4)
        plt.gca().set_aspect('equal')
	plt.plot(self.raw_chan.real[self.indexmin[chan],chan], self.raw_chan.imag[self.indexmin[chan],chan], 'o', color='red')
        #plt.xlim(np.std(self.raw_chan.real[:,chan])*-3.,np.std(self.raw_chan.real[:,chan]*3))
        #plt.ylim(np.std(self.raw_chan.imag[:,chan])*-3.,np.std(self.raw_chan.imag[:,chan]*3))
        plt.tight_layout()
	plt.show()
        return

    def plot_loop_rotated(self,chan):
        plt.figure(figsize = (20,20))
        plt.title('IQ loop Channel = ' + str(chan) + ', centered and rotated')
        plt.plot(self.chan_rotated.real[:,chan],self.chan_rotated.imag[:,chan],'x',color='red',mew=2, ms=6)
	plt.plot(self.chan_rotated.real[self.indexmin[chan],chan], self.chan_rotated.imag[self.indexmin[chan],chan], 'o', color='green')
        plt.gca().set_aspect('equal')
        plt.xlim(np.std(self.chan_rotated.real[:,chan])*-3,np.std(self.chan_rotated.real[:,chan])*3)
        plt.ylim(np.std(self.chan_rotated.imag[:,chan])*-3,np.std(self.chan_rotated.imag[:,chan])*3)
        plt.xlabel('I', size = 20)
        plt.ylabel('Q', size = 20)
        plt.tight_layout()
	plt.show()
        return

    def multiplot(self, chan):
        #for chan in range(self.nchan):
        #rf_freqs = np.load(os.path.join(self.datapath,'light_kids.npy'))
        #rf_freq= rf_freqs[chan] - (np.max(rf_freqs) + np.min(rf_freqs))/2. + self.lo_freqs
        #print np.shape(rf_freq)
        rf_freqs = (self.bb_freqs[chan] + (self.lo_freqs))/1.0e6 #lo/2
	self.mag /= (2**15 - 1)
	self.mag /= ((2**21 - 1) / 2048)
        fig,axs = plt.subplots(1,3)
        plt.suptitle('Chan ' + str(chan))

	axs[0].plot(rf_freqs, 20*np.log10(self.mag[:,chan]),'b', linewidth = 3)

        axs[1].plot(rf_freqs, self.phase_rotated[:,chan],'b',linewidth = 3)

        axs[2].plot(self.chan_rotated[:,chan].real,self.chan_rotated[:,chan].imag,'b',marker='x', linewidth = 2)
        axs[2].axis('equal')                    
        axs[2].legend(loc = 'lower left', fontsize = 15)
        plt.grid()            
        plt.tight_layout()
	#plt.savefig(os.path.join(self.datapath,'multiplot%04drotated.png'%chan), bbox = 'tight')
        plt.show()         
        #plt.clf()
        #print ' plotting ',chan,;sys.stdout.flush()
        return

    def plot_targ(self, path):
    	plt.ion()
	plt.figure(6)
	plt.clf()
	lo_freqs, Is, Qs = self.open_stored(path)
	lo_freqs = np.loadtxt(path + '/sweep_freqs.dat')
	bb_freqs = np.loadtxt(path + '/bb_freqs.dat')
	channels = len(bb_freqs)
	mags = np.zeros((channels,len(lo_freqs))) 
	chan_freqs = np.zeros((channels,len(lo_freqs)))
        new_targs = np.zeros((channels))
	print('channels =',channels)
	for chan in range(channels):
		print(chan)#,Is[:,chan],Qs[:,chan])

		print(Is)
		print(Qs)
        	mags[chan] = np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2)
		mags[chan] /= 2**15 - 1
		mags[chan] /= ((2**21 - 1) / 512.)
		mags[chan] = 20*np.log10(mags[chan])
		chan_freqs[chan] = (lo_freqs + bb_freqs[chan])/1.0e6 #lo/2

		plt.vlines(self.target_freqs_out[chan], np.min(mags[chan]), np.max(mags[chan]), color="red")
		plt.vlines(self.minfreqs[chan], np.min(mags[chan]), np.max(mags[chan]), color="green")

	#????? era scommentato mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
	#bb_freqs = np.concatenate(bb_freqs[len(b_freqs)/2:],bb_freqs[:len(bb_freqs)/2]))
	#????? era scom chan_freq = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
	#new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]
	#print new_targs

	for chan in range(channels):
		plt.plot(chan_freqs[chan],mags[chan])
	plt.title('Target sweep')
	plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
	return

