import numpy as np
import sys

LPF_par = np.load("transfunc_lowpass_coefficients.npy")
Poly_par = np.load("transfunc_polyfit_coefficients.npy")

baseline_attenuation = -44.8 #db

def transferFunctionLowpass(x, k, nu0, nuC, y0):
    return 20*np.log10(k/np.sqrt(1.+((x-nuC)/nu0)**2.) + y0)


# working directory
wd = sys.argv[1]

# read target_freqs.dat
target_freqs_file = open("target_freqs.dat", "r+")

freqs = []

flagIsTrue = True
while(flagIsTrue):
    line = target_freqs_file.readline()
    if line == '':
        flagIsTrue = False
    else:
        freqs.append(float(line))

target_freqs_file.close()

# compute attenuations
target_freqs_output = open("target.dat", "w+")


attenuationCheckFlag = True
attenuations = []
for f in freqs:
    # polynomial attenuation profile
    attenuation = np.poly1d(Poly_par)(f)
    # lowpass filter attenuation profile
    #attenuation = transferFunctionLowpass(f, LPF_par[0], LPF_par[1], LPF_par[2], LPF_par[3])
    
    attenuation = 10**((baseline_attenuation-attenuation)/20)
    attenuations.append(attenuation)
    
    if attenuationCheckFlag and attenuation > 1.0:
        attenuationCheckFlag = False
    
    target_freqs_output.write("{:.2f}\n{:.3f}\n".format(f, attenuation))

print("target_freqs.dat generated with baseline_attenuation = {:.2f} db".format(baseline_attenuation))
    
if not attenuationCheckFlag:
    print("Attenuation warning: a value >1 encountered! Check the baseline_attenuation value!")

target_freqs_output.close()

'''
import matplotlib.pyplot as plt
plt.plot(freqs, attenuations)
plt.show()
'''
