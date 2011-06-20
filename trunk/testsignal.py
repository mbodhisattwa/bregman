#testsignal.py - generate some multi-dimensional test signals
#
#Author: Michael A. Casey
#Copyright (C) 2010 Bregman Music and Audio Research Studio
#Dartmouth College, All Rights Reserved
#
# A collection of test signal generators
#

# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "New BSD License"
__email__ = 'mcasey@dartmouth.edu'

import pylab
import scipy.signal

TWO_PI = 2.0 * pylab.pi

# Exception Handling class
class TestSignalError(Exception):
    """
    Test signal exception class.
    """
    def __init__(self):
        print "An error occured inside a TestSignal function call"

# Return parameter dict used by all of the test signal generators
def default_signal_params():
    """    
    ::
    
      Return a new parameter dict consisting of:
        'sr': 44100.0      # audio sample rate 
        'num_harmonics': 2 # number of harmonics to render
    """
    p = {'sr':44100.0,
         'num_harmonics':2
         }
    return p

# A single sinusoid
def sinusoid(params=None, f0=441.0, num_points=44100, phase_offset=0):
    """
    ::

        Generate a sinusoidal audio signal
         params - a parameter dict containing sr, and num_harmonics elements
             f0 - fundamental frequency in Hertz [441.0]
             num_points - how many samples to generate [44100]
             phase_offset - initial phase of the sinusoid
    """
    if params==None:
        params = default_signal_params()
    sr = float(params['sr'])
    t = pylab.arange(num_points)
    x = pylab.sin( TWO_PI*f0/sr * t + phase_offset)
    return x

# Harmonic sinusoids
def harmonics(params=None, f0=441.0, afun=lambda x: pylab.exp(-0.5*x), num_points=44100, phase_offset=0):
    """
    ::

        Generate a harmonic series using a harmonic weighting function
         params - parameter dict containing sr, and num_harmonics elements
         afun   - a lambda function of one parameter (harmonic index) returning a weight
         num_points - how many samples to generate [44100]
         phase_offset - initial phase of the harmonic series
    """
    if params==None:
        params = default_signal_params()
    f0 = float(f0)
    sr = float(params['sr'])
    num_harmonics  = params['num_harmonics']
    x = pylab.zeros(num_points)
    for i in pylab.arange(1, num_harmonics+1):    
        x +=  afun(i) * sinusoid(params, f0=i*f0, num_points=num_points, phase_offset=i*phase_offset)
    x /= pylab.rms_flat(x)
    return x

# Shepard tones from harmonic sinusoids
def shepard(params=None, f0=55, num_octaves=7, num_points=44100, phase_offset=0, center_freq=440, band_width=150):
    """
    ::

        Generate shepard tones
             params - parameter dict containing sr, and num_harmonics elements
             f0 - base frequency in Hertz of shepard tone [55]
             num_octaves - number of sinusoidal octave bands to generate [7]
             num_points - how many samples to generate [44100]
             phase_offset - initial phase offset for shepard tone
             center_freq - where the peak of the spectrum will be
             band_width - how wide a spectral band to use for shepard tones
    """
    if params==None:
        params = default_signal_params()
    x = pylab.zeros(num_points)
    shepard_weight = gauss_pdf(20000, center_freq, band_width)
    for i in pylab.arange(0, num_octaves):
        #afun=lambda x: ]
        a = shepard_weight[int(round(f0*2**i))]
        x += a * harmonics(params, f0=f0*2**i, num_points=num_points, phase_offset=phase_offset)
    x /= pylab.rms_flat(x)
    return x

# 1d Gaussian kernel
def gauss_pdf(n,mu=0.0,sigma=1.0):
    """
    ::

        Generate a gaussian kernel
         n - number of points to generate
         mu - mean
         sigma - standard deviation
    """
    var = sigma**2
    return 1.0 / pylab.sqrt(2 * pylab.pi * var) * pylab.exp( -(pylab.r_[0:n] - mu )**2 / ( 2.0 * var ) )

# Chromatic sequence of shepard tones
def devils_staircase(params, f0=441, num_octaves=7, num_steps=12, step_size=1, hop=4096, 
                     overlap=True, center_freq=440, band_width=150):
    """
    ::

        Generate an auditory illusion of an infinitely ascending/descending sequence of shepard tones
            params - parameter dict containing sr, and num_harmonics elements
            f0 - base frequency in Hertz of shepard tone [55]
            num_octaves - number of sinusoidal octave bands to generate [7]
            num_steps - how many steps to take in the staircase
            step_size - semitone change per step, can be fractional [1.]
            hop - how many points to generate per step
            overlap - whether the end-points should be cross-faded for overlap-add
            center_freq - where the peak of the spectrum will be
            band_width - how wide a spectral band to use for shepard tones
    """
    sr = params['sr']
    norm_freq = 2*pylab.pi/sr
    wlen = min([hop/2, 2048])
    print wlen
    x = pylab.zeros(num_steps*hop+wlen)
    h = scipy.signal.hanning(wlen*2)
    # overlap add    
    phase_offset=0
    for i in pylab.arange(num_steps):
        freq = f0*2**(((i*step_size)%12)/12.0)        
        s = shepard(params, f0=freq, num_octaves=num_octaves, num_points=hop+wlen, 
                    phase_offset=0, center_freq=center_freq, band_width=band_width)
        s[0:wlen] *= h[0:wlen]
        s[hop:hop+wlen] *= h[wlen:wlen*2]
        x[i*hop:(i+1)*hop+wlen] += s
        phase_offset += hop*freq*norm_freq
    if not overlap:
        x = pylab.resize(x, num_steps*hop)
    x /= pylab.rms_flat(x)
    return x

# Overlap-add two signals
def overlap_add(x, y, wlen):
    """
    ::

        Overlap-add two sequences x and y by wlen samples
    """
    z = pylab.zeros(x.size + y.size - wlen)
    z[0:x.size] = x;
    z[x.size-wlen:x.size+y.size-wlen]+=y
    return z

# Parameter dict for noise test signals
def default_noise_params():
    """
    ::

        Returns a new parameter dict for noise generators consisting of:
             'noise_dB':24.0  - relative amplitude of noise to harmonic signal content
             'num_harmonics':1 - how many harmonics (bands) to generate
             'cf':441.0 - center frequency in Hertz
             'bw':50.0 - bandwidth in Hertz
             'sr':44100.0 - sample rate in Hertz
    """
    p = {'noise_dB':24.0,
         'num_harmonics':1,
         'cf':441.0,
         'bw':50.0,
         'sr':44100.0
         }
    return p

# Combine harmonic sinusoids and noise signals
def noise(params=None, num_points=44100, filtered=True, modulated=True, noise_fun=pylab.rand):
    """
    ::

        Generate noise according to params dict
            params - parameter dict containing sr, and num_harmonics elements [None=default params]
            num_points - how many samples to generate [44100]
            filtered - set to True for filtered noise sequence [True]
            modulated - set to True for modulated noise sequence [True]
            noise_fun - the noise generating function [pylab.rand]
    """
    if params==None:
        params = default_noise_params()
    noise_dB = params['noise_dB']
    num_harmonics = params['num_harmonics']
    cf = params['cf']
    bw = params['bw']
    sr = params['sr']
    g = 10**(noise_dB/20.0)*noise_fun(num_points)
    if filtered or modulated:
        [b,a] = scipy.signal.filter_design.butter(4, bw*2*pylab.pi/sr, btype='low', analog=0, output='ba')
        g = scipy.signal.lfilter(b, a, g)

    if not modulated:
        # Additive noise
        s = harmonics(params, f0=cf, num_points=num_points)
        x = s + g
    else:
        # Phase modulation with *filtered* noise (side-band modulation should be narrow-band at bw)
        x = pylab.zeros(num_points)
        for i in pylab.arange(1,num_harmonics+1):
            x += pylab.exp(-0.5*i) * pylab.sin( (2.0*pylab.pi*cf*i / sr) * pylab.arange(num_points) + g)
    x /= pylab.rms_flat(x)
    return x

def modulate(sig, env, nsamps):
    """
    ::

        Signal modulation by an envelope
        sig - the full-rate signal
        env - the reduced-rate envelope
        nsamps - audio samples per envelope frame
    """
    if( sig.size != len(env)*nsamps ):
        print "Source signal size must equal len(env) * nsamps"
        return False
    y = pylab.zeros(sig.size)
    start = 0
    for a in env:
        end = start + nsamps
        y[start:end] = a * sig[start:end]
        start = end
    return y

def default_rhythm_params():
    """
    ::

        Return signal_params and pattern_params dicts, and a patterns tuple for 
        a default rhythm signal such that:
                'sr' : 48000,        # sample rate
                'bw' : [80., 2500., 1000.], # band-widths
                'cf' : [110., 5000., 16000.], # center-frequencies
                'dur': [0.5, 0.5, 0.5] # relative duration of timbre
                'normalize' : 'rms' # balance timbre channels 'rms', 'maxabs', 'norm', 'none'
        Example:
         signal_params, rhythm_params, patterns = default_rhythm_params()
         sig = rhythm(signal_params, rhythm_params, patterns)
    """
    sp = {
        'sr' : 48000,
        'tc' : 2.0,
        'cf' : [110., 5000., 16000.],
        'bw' : [80., 2500., 1000.],
        'dur' : [1.0, 0.5, 0.25],
        'normalize' : 'none'
        }
    rp = {
        'tempo' : 120.,
        'subdiv' : 16
        }
    pats = (0b1010001010100000, 0b0000100101001001, 0b1010101010101010)
    return (sp, rp, pats)

def _check_rhythm_params(signal_params, patterns):
    num_timbres = len(signal_params['cf'])
    if not ( num_timbres == len(signal_params['bw']) == len(signal_params['dur']) == len(patterns) ):
        return 0
    return num_timbres

def balance_signal(sig, balance_type):
    """
    ::
    
        Perform signal balancing using:
          rms - root mean square
          maxabs - maximum absolte value
          norm - Euclidean norm
          none - do nothing [default]
    """
    balance_types = ['rms', 'maxabs', 'norm', 'none']
    if balance_type==balance_types[0]:
        return sig
    if balance_type==balance_types[1]:
        return sig    
    if balance_type==balance_types[2]:
        return sig    
    if balance_type==balance_types[3]:
        return sig    
    print "signal balancing type not supported: ", balance_type
    raise TestSignalError()

def rhythm(signal_params=None, rhythm_params=None, patterns=None):
    """
    ::

        Generate a multi-timbral rhythm sequence using noise-band timbres 
        with center-frequency, bandwidth, and decay time controls

        Timbre signal synthesis parameters are specified in 
        the signal_params dict:
            ['cf'] - list of center-frequencies for each timbre
            ['bw'] - list of band-widths for each timbre
            ['dur'] - list of timbre durations relative to a quarter note
            ['sr'] - sample rate of generated audio
            ['tc'] - constant of decay envelope relative to subdivisions:
             The following expression yields a time-constant for decay to -60dB 
             in a given number of beats at the given tempo:
               t = beats * tempo / 60.
               e^( -tc * t ) = 10^( -60dB / 20 )
               tc = -log( 0.001 ) / t           

        The rhythm sequence is generated with musical parameters specified in
        the rhythm_params dict: 
            ['tempo']  - how fast
            ['subdiv'] - how many pulses to divide a 4/4 bar into

        Rhythm sequences are specified in the patterns tuple (p1,p2,...,pn)
           patterns - n-tuple of integers with subdiv-bits onset patterns, 
            one integer element for each timbre

           Parameter constraints:
             Fail if not:
               len(bw) == len(cf) == len(dur) == len(patterns)
    """
    # Short names
    p = default_rhythm_params()
    if signal_params==None: signal_params = p[0]
    if rhythm_params==None: rhythm_params = p[1]
    if patterns==None: patterns = p[2]
    sp = signal_params
    rp = rhythm_params
    num_timbres = _check_rhythm_params(signal_params, patterns)
    if not num_timbres: 
        print "rhythm: signal_params lists and pattern n-tuple lengths don't match"
        raise TestSignalError()
    # Duration parameters
    qtr_dur = 60.0 / rp['tempo'] * sp['sr'] # duration of 1/4 note
    eth_dur = 60.0 / (2.0 * rp['tempo']) * sp['sr'] # duration of 1/8 note
    sxt_dur = 60.0 / (4.0 * rp['tempo']) * sp['sr'] # duration of 1/16 note
    meter = 4.0
    bar_dur = meter * qtr_dur # duration of 1 bar

    # Audio signal wavetables from parameters
    ns_sig=[]
    ns_env=[]
    for cf, bw, dur in zip(sp['cf'], sp['bw'], sp['dur']):
        ns_par = default_noise_params()
        ns_par['sr'] = sp['sr']
        ns_par['cf'] = cf
        ns_par['bw'] = bw
        ns_sig.append( balance_signal(noise( ns_par, num_points = 2 * bar_dur ), sp['normalize']))
        ns_env.append( pow( 10, -sp['tc'] * pylab.r_[ 0 : 2 * bar_dur ] / (qtr_dur * dur) ) )

    # Music wavetable sequencer
    snd = [[] for i in range(num_timbres)] 
    snd_ptr = [qtr_dur  for i in range(num_timbres)]
    num_beats = rp['subdiv']
    test_bit = 1 << ( num_beats - 1 )
    dt = 16.0 / num_beats
    for beat in range(num_beats):
        for p, pat in enumerate(patterns):
            if (pat & (test_bit >> beat) ): snd_ptr[p] = 0

        for t in range(num_timbres):
            idx = pylab.array(pylab.r_[snd_ptr[t]:snd_ptr[t]+sxt_dur*dt], dtype='int')
            snd[t].append( ns_sig[t][idx] * ns_env[t][idx] )
            snd_ptr[t] += sxt_dur * dt

    all_sig = pylab.concatenate( snd[0] )
    for t in pylab.arange(1, num_timbres):
        sig = pylab.concatenate( snd[t] )
        all_sig += sig
    return all_sig / ( 3 * num_timbres) # protect from overflow

