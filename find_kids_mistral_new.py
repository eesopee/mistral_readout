import numpy as np
import sys, os
import matplotlib.pyplot as plt
import scipy.ndimage
#import astropy.stats.sigma_clipping
import scipy.signal
import time
import os



def openStored(path):
    files = sorted(os.listdir(path))
    I_list = [os.path.join(path, filename) for filename in files if filename.startswith('I')]
    Q_list = [os.path.join(path, filename) for filename in files if filename.startswith('Q')]
    chan_I = np.array([np.load(filename) for filename in I_list])
    chan_Q = np.array([np.load(filename) for filename in Q_list])
    return chan_I, chan_Q

def normalize_and_stack(path, bb_freqs, lo_freqs):


    chan_I, chan_Q = openStored(path)
    channels = np.arange(np.shape(chan_I)[1])
    print("VNA with ", len(channels), " channels")
    mag = np.zeros((len(channels),len(lo_freqs)))
    chan_freq = np.zeros((len(channels),len(lo_freqs)))

    for chan in channels:        
        mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2)) 
        chan_freq[chan] = (lo_freqs + bb_freqs[chan])/1.0e6 #era lo_freqs/2
    
    '''
    Normalization and conversion in dB
    '''
        
    for chan in channels:
        mag[chan] /= (2**31-1)
        mag[chan] /= ((2**21-1)/512)
        mag[chan] = 20*np.log10(mag[chan])
    
    for chan in channels:
        delta = mag[chan-1][-1]-mag[chan][0]
        mag[chan] += delta
    
    mags = np.hstack(mag)        
    chan_freqs = np.hstack(chan_freq)

    return chan_freqs, mags

def lowpass_cosine( y, tau, f_3db, width, padd_data=True):
    
    import numpy as nm
        # padd_data = True means we are going to symmetric copies of the data to the start and stop
    # to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
    #
    # False means we're going to have transients at the start and stop of the data

    # kill the last data point if y has an odd length
    if nm.mod(len(y),2):
        y = y[0:-1]

    # add the weird padd
    # so, make a backwards copy of the data, then the data, then another backwards copy of the data
    if padd_data:
        y = nm.append( nm.append(nm.flipud(y),y) , nm.flipud(y) )

    # take the FFT
        import scipy
        import scipy.fftpack
    ffty=scipy.fftpack.fft(y)
    ffty=scipy.fftpack.fftshift(ffty)

    # make the companion frequency array
    delta = 1.0/(len(y)*tau)
    nyquist = 1.0/(2.0*tau)
    freq = nm.arange(-nyquist,nyquist,delta)
    # turn this into a positive frequency array
    pos_freq = freq[(len(ffty)//2):]

    # make the transfer function for the first half of the data
    i_f_3db = min( nm.where(pos_freq >= f_3db)[0] )
    f_min = f_3db - (width/2.0)
    i_f_min = min( nm.where(pos_freq >= f_min)[0] )
    f_max = f_3db + (width/2);
    i_f_max = min( nm.where(pos_freq >= f_max)[0] )

    transfer_function = nm.zeros(int(len(y)//2))
    transfer_function[0:i_f_min] = 1
    transfer_function[i_f_min:i_f_max] = (1 + nm.sin(-nm.pi * ((freq[i_f_min:i_f_max] - freq[i_f_3db])/width)))/2.0
    transfer_function[i_f_max:(len(freq)//2)] = 0

    # symmetrize this to be [0 0 0 ... .8 .9 1 1 1 1 1 1 1 1 .9 .8 ... 0 0 0] to match the FFT
    transfer_function = nm.append(nm.flipud(transfer_function),transfer_function)

    # apply the filter, undo the fft shift, and invert the fft
    filtered=nm.real(scipy.fftpack.ifft(scipy.fftpack.ifftshift(ffty*transfer_function)))

    # remove the padd, if we applied it
    if padd_data:
        filtered = filtered[(len(y)//3):(2*(len(y)//3))]

    # return the filtered data
        return filtered

'''
            adaptive_iteratively_reweighted_penalized_least_squares_smoothing function
'''
def adaptive_iteratively_reweighted_penalized_least_squares_smoothing(data, lam=1.0e6, N_iter=5):
    '''
    lam: adjusting parameter
    N_iter: number of iteration
    '''
    from scipy.sparse import spdiags, linalg, diags
    from scipy.linalg import norm
    L = len(data)
    D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(N_iter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*data)
        d_mod = norm((z-data)[z>data])
        if d_mod < 0.001 * norm(data):
            return z
        p = np.exp(i*(data-z)/d_mod)
        w = 0.0*(data < z) + p*(data >= z)
    return z


def main(path, savefile=False):
    print("Searching for KIDs")
    
    bb_freqs = np.load(path + "/bb_freqs.npy")
    lo_freqs = np.load(path + "/sweep_freqs.npy")
 
    chan_freqs, mags = normalize_and_stack(path, bb_freqs, lo_freqs)
   
    sweep_step = 1.25 # kHz
    smoothing_scale = 2500.0 # kHz
    
    #filtered = lowpass_cosine( mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    filtered = adaptive_iteratively_reweighted_penalized_least_squares_smoothing(mags)
    from scipy.signal import find_peaks

    # parametri buoni per MISTRAL 415 v4
    peak_width=(1.0, 150.0)
    peak_height=0.3
    peak_prominence=(0.2, 30.0)

    peaks, roba = find_peaks(-(mags-filtered),
                                            width=peak_width,
                                            prominence=peak_prominence,
                                            height=peak_height)
                                       
    
    

    target_freqs = chan_freqs[peaks]
    print(target_freqs)
    np.save(path + '/target_freqs.npy', target_freqs)
    np.savetxt(path + '/target_freqs.dat', target_freqs)
        

    plt.plot(chan_freqs, mags-filtered, label="VNA sweep, low-passed")

    print("Found ", len(peaks), "KIDs")
    
    plt.title("VNA sweep and automatic KIDs search")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB]")
    plt.plot(chan_freqs[peaks], mags[peaks]-filtered[peaks],"x", label="KIDs")
    plt.show()
    
    '''
    kids_distance = []
    
    for i in range(0, len(peaks)):
        kids_distance.append(chan_freqs[peaks[i]]-chan_freqs[peaks[i-1]])
        
    print(kids_distance)
    min_distance = np.argmin(kids_distance[1:])
    print("Min sep between kids=", kids_distance[min_distance],"MHz at",chan_freqs[peaks[min_distance]])
    
    plt.plot( (chan_freqs[peaks[min_distance]], chan_freqs[peaks[min_distance+1]]), 
             (mags[peaks[min_distance]]-filtered[peaks[min_distance]],mags[peaks[min_distance+1]]-filtered[peaks[min_distance]+1]), "o", color="red", label="Closest KIDs: "+str(round(kids_distance[min_distance]*100, 2))+"kHz")
    '''
    
    plt.legend()

    return target_freqs
