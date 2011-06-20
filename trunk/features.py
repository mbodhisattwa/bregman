
# features.py - feature extraction and plotting
# Bregman - music information retrieval toolkit

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'


import pylab
import error
import glob
from sound import *
from audiodb import *

# Features Class
class Features:
    """        
    ::

        F = Features(arg, feature_params)
        type(arg) is str: load audio from file
        type(arg) is ndarray: set audio from array

        feature_params['feature'] = 
         'stft'   - short-time fourier transform
         'power'  - power
         'cqft'   - constant-q fourier transform
         'mfcc'   - Mel-frequency cepstral coefficients
         'lcqft'  - low-quefrency cepstral coefficients
         'hcqft'  - high-quefrency cepstral coefficients
         'chroma' - chroma (pitch-class) power coefficients

         Features are extracted in the following hierarchy:
          stft->cqft->mfcc->[lcqft,hcqft]->chroma,
          if a later feature was extracted, then an earlier feature is also available

         Plotting. Features are available to plot in the following order:
          F.feature_plot(feature='power', dbscale=True)
          F.feature_plot(feature='stft', dbscale=True, normalize=True)
          F.feature_plot(feature='cqft', dbscale=True, normalize=True)
          F.feature_plot(feature='mfcc', normalize=True) # already log scaled
          F.feature_plot(feature='chroma', dbscale=True, normalize=True)
          F.feature_plot(feature='lcqft', dbscale=True, normalize=True)
          F.feature_plot(feature='hcqft', dbscale=True, normalize=True)

          dbscale and normalize are optional, but recommended for inspecting magnitude and power values

         Access to feature arrays. Use the following members which are numpy ndarrays:
          F.STFT
          F.POWER
          F.CQFT
          F.MFCC
          F.LCQFT
          F.HCQFT
          F.CHROMA
    """
    def __init__(self, arg=None, feature_params=None):
        self.reset()
        self.feature_params = feature_params
        if feature_params==None:
            self.feature_params = self.default_feature_params()

        if type(arg)==pylab.ndarray:
            self.set_audio(arg)
            self.extract()
        elif type(arg)==type(''):
            filename = arg
            if filename:
                self.load_audio(filename) # open file as MONO signal
                self.extract()

    @staticmethod
    def default_feature_params():
        """
        ::

            Return a new feature parameter dict. 
            Feature opcodes are listed in the Features documentation.
            default_feature_params = {
                'sample_rate': 44100,
                'feature':'cqft', 
                'nbpo' : 12,
                'ncoef' : 10,
                'lcoef' : 0,
                'lo': 63.5444, 
                'hi': 16000,
                'nfft': 16384,
                'wfft': 8192,
                'nhop': 4410,
                'log10': False,
                'magnitude': True,
                'power_ext': ".power",
                'intensify' : False
                'verbosity' : 1
                'nsamples' : None} - use nsamples to sample from the STFT for subsequent features 
        """

        feature_params = {
            'sample_rate': 44100,
            'feature':'cqft', 
            'nbpo': 12,
            'ncoef' : 10,
            'lcoef' : 1,
            'lo': 63.5444, 
            'hi': 16000,
            'nfft': 16384,
            'wfft': 8192,
            'nhop': 4410,
            'log10': False,
            'magnitude': True,
            'power_ext': ".power",
            'intensify' : False,
            'verbosity' : 1,
            'nsamples' : None
            }
        return feature_params

    def reset(self):
        """
        ::

            Reset the feature extractor state. No signal. No features.
        """
        self._have_x=False
        self.x=None # the audio signal
        self._have_stft=False
        self.STFT=None
        self._have_cqft=False
        self.POWER=None
        self._have_power=False
        self._is_intensified=False
        self.CQFT=None
        self._have_mfcc=False
        self.MFCC=None
        self._have_lcqft=False
        self.LCQFT=None
        self._have_hcqft=False
        self.HCQFT=None
        self._have_chroma=False
        self.CHROMA=None

    def load_audio(self,filename):
        """
        ::

            Open a WAV/AIFC/AU file as a MONO signal [L], sets audio buffer
        """
        wav=WavOpen(filename)
        self.set_audio(wav.sig, wav.sample_rate)
        
    def set_audio(self, x, sr=44100.):
        """
        ::

            Set audio buffer to extract as an array
        """
        self.reset()
        if len(x.shape) > 1:
            x = x.sum(1) / x.shape[1] # handle stereo files
        self.x = x
        self._have_x=True
        self.sample_rate = sr

    def _check_feature_params(self,feature_params=None):
        if feature_params:
            self.feature_params = feature_params
        if self.feature_params==None:
            print "You must specify feature_params for extraction"
            raise error.BregmanError()
        return self.feature_params

    def extract(self, feature_params=None):
        """
        ::
        
            Extract audio features according to feature_params specification:
        """
        f = self._check_feature_params(feature_params)['feature']
        # processing chain        
        if f == 'power':
            self._power()
        if f == 'chroma':
            self._chroma()
        if f == 'hcqft':
            return self._hcqft()
        if f == 'lcqft':
            return self._lcqft()
        if f == 'mfcc':
            return self._mfcc()
        if f == 'cqft':
            return self._cqft()
        if f == 'stft':
            return self._stft()

    def feature_plot(self,feature=None,normalize=False,dbscale=False, norm=False, interp='bicubic', labels=False):
        """
        ::

          Plot the given feature, default is self.feature_params['feature'], 
           returns an error if feature not extracted

          Inputs:
           feature   - the feature to plot self.feature_params['feature']
                        features are extracted in the following hierarchy:
                           stft->cqft->mfcc->[lcqft,hcqft]->chroma,
                        if a later feature was extracted, then an earlier feature can be plotted
           normalize - column-wise normalization ['False']
           dbscale   - transform linear power to decibels: 20*log10(X) ['False']
           norm      - make columns unit norm ['False']
           interp    - how to interpolate values in the plot ['bicubic']
        """
        if feature == None:
            feature = self._check_feature_params()['feature']
        # check plots        
        if feature =='stft':
            if not self._have_stft:
                print "Error: must extract STFT first"
            else:
                adb.feature_plot(pylab.absolute(self.STFT), normalize, dbscale, norm, title_string="STFT", interp=interp)
                if labels:
                    self._feature_plot_xticks(pylab.linspace(0, self.STFT.shape[1],10)*(self.feature_params['nhop']/self.sample_rate))
                    self._feature_plot_yticks(pylab.linspace(0, self.STFT.shape[0], 20)*(self.sample_rate/(self.feature_params['nfft'])))
                pylab.xlabel('Time (secs)')
                pylab.ylabel('Frequency (Hz)')
        elif feature == 'power':
            if not self._have_power:
                print "Error: must extract POWER first"
            else:
                pylab.figure()
                pylab.plot(adb.feature_scale(self.POWER, normalize, dbscale)/20.0)
                pylab.title("Power")
                pylab.xlabel("Sample Index")
                pylab.ylabel("Power (dB)")
        elif feature == 'cqft':
            if not self._have_cqft:
                print "Error: must extract CQFT first"
            else:
                adb.feature_plot(self.CQFT, normalize, dbscale, norm, title_string="CQFT",interp=interp)
                if labels:
                    self._feature_plot_xticks(pylab.linspace(0, self.STFT.shape[1],10)*(self.feature_params['nhop']/self.sample_rate))
                    self._feature_plot_yticks(110*2**(pylab.linspace(0, self.CQFT.shape[1],10)/12.))
                pylab.xlabel('Time (secs)')
                pylab.ylabel('Frequency (Hz)')
        elif feature == 'mfcc':
            if not self._have_mfcc:
                print "Error: must extract MFCC first"
            else:
                fp = self._check_feature_params()
                X = self.MFCC[fp['lcoef']:fp['lcoef']+fp['ncoef'],:]
                adb.feature_plot(X, normalize, dbscale, norm, title_string="MFCC",interp=interp)
        elif feature == 'lcqft':
            if not self._have_lcqft:
                print "Error: must extract LCQFT first"
            else:
                adb.feature_plot(self.LCQFT, normalize, dbscale, norm, title_string="LCQFT",interp=interp)
        elif feature == 'hcqft':
            if not self._have_hcqft:
                print "Error: must extract HCQFT first"
            else:
                adb.feature_plot(self.HCQFT, normalize, dbscale, norm, title_string="HCQFT",interp=interp)
        elif feature == 'chroma':
            if not self._have_chroma:
                print "Error: must extract CHROMA first"
            else:
                adb.feature_plot(self.CHROMA, normalize, dbscale, norm, title_string="CHROMA",interp=interp)
        else:
            print "Unrecognized feature, skipping plot: ", feature

    def _feature_plot_xticks(self, xticks):
        x = pylab.plt.xticks()[0]
        locs = pylab.linspace(0,pylab.plt.xlim()[1],len(xticks))
        pylab.plt.xticks(locs, [round(x*100)/100 for x in xticks])
        pylab.axis('tight')

    def _feature_plot_yticks(self, yticks):
        y = pylab.plt.yticks()[0]
        locs = pylab.linspace(0,pylab.plt.ylim()[1],len(yticks))
        pylab.plt.yticks(locs, [round(y*100)/100 for y in yticks])
        pylab.axis('tight')
        
    def _stft_specgram(self):
        if not self._have_x:
            print "Error: You need to load a sound file first: use self.load_audio('filename.wav')\n"
            return False
        else:
            fp = self._check_feature_params()
            self.STFT=pylab.mlab.specgram(self.x, NFFT=fp['nfft'], noverlap=fp['nfft']-fp['nhop'])[0]
            self.STFT/=pylab.sqrt(fp['nfft'])
            self._have_stft=True
        if fp['verbosity']:
            print "Extracted STFT: nfft=%d, hop=%d" %(fp['nfft'], fp['nhop'])
        return True

    def _win_mtx(self, x, w, h, nsamples=None, win=None):
        num_frames = int( pylab.ceil( x.size / float( h ) ) )
        X = pylab.zeros((w, num_frames))
        if win is None:
            win = pylab.hamming(w)
        if nsamples is None:
            frames = range(num_frames)
        else:
            frames = sort(permutation(num_frames)[0: nsamples])
        for k in frames:
            start_pos = k*h
            end_pos = start_pos + w
            if x.size < end_pos:
                X[:,k] = win * pylab.concatenate((x[start_pos:-1], pylab.zeros(w - (x.size - start_pos - 1))), axis=1)
            else:
                X[:,k] = win * x[start_pos:end_pos]
        return X.T


    def _make_log_freq_map(self):
        """
        ::

            For the given ncoef (bands-per-octave) and nfft, calculate the center frequencies
            and bandwidths of linear and log-scaled frequency axes for a constant-Q transform.
        """
        fp = self.feature_params
        bpo = float(fp['nbpo']) # Bands per octave
        self._fftN = float(fp['nfft']/2+1)
        hi_edge = float( fp['hi'] )
        lo_edge = float( fp['lo'] )
        f_ratio = 2.0**( 1.0 / bpo ) # Constant-Q bandwidth
        self._cqtN = float( pylab.floor(pylab.log(hi_edge/lo_edge)/pylab.log(f_ratio)) )
        self._dctN = self._cqtN
        self._outN = float( self._fftN )
        if self._cqtN<1: print "warning: cqtN not positive definite"
        mxnorm = pylab.empty(self._cqtN) # Normalization coefficients        
        fftfrqs=pylab.array([i * self.sample_rate / float(self._fftN) for i in pylab.arange(self._outN)])
        logfrqs=pylab.array([lo_edge * pylab.exp(pylab.log(2.0)*i/bpo) for i in pylab.arange(self._cqtN)])
        logfbws=pylab.array([max(logfrqs[i] * (f_ratio - 1.0), self.sample_rate / float(self._fftN)) 
                         for i in pylab.arange(self._cqtN)])
        self._fftfrqs = fftfrqs
        self._logfrqs = logfrqs
        self._logfbws = logfbws
        self._make_cqt()

    def _make_cqt(self):
        """
        ::    

            Build a constant-Q transform (CQT) from lists of 
            linear center frequencies, logarithmic center frequencies, and
            constant-Q bandwidths.
        """
        fftfrqs = self._fftfrqs
        logfrqs = self._logfrqs
        logfbws = self._logfbws
        fp = self.feature_params
        ovfctr = 0.5475 # Norm constant so CQT'*CQT close to 1.0
        tmp2 = 1.0 / ( ovfctr * logfbws )
        tmp = ( logfrqs.reshape(1,-1) - fftfrqs.reshape(-1,1) ) * tmp2
        self.Q = pylab.exp( -0.5 * tmp * tmp )
        self.Q *= 1.0 / ( 2.0 * pylab.sqrt( (self.Q * self.Q).sum(0) ) )
        self.Q = self.Q.T

    def _make_dct(self):
        """
        ::

            Construct the discrete cosine transform coefficients for the 
            current size of constant-Q transform
        """
        DCT_OFFSET = self.feature_params['lcoef']
        nm = 1 / pylab.sqrt( self._cqtN / 2.0 )
        self.DCT = pylab.empty((self._dctN, self._cqtN))
        for i in pylab.arange(self._dctN):
          for j in pylab.arange(self._cqtN):
            self.DCT[ i, j ] = nm * pylab.cos( i * (2 * j + 1) * (pylab.pi / 2.0) / float(self._cqtN)  )
        for j in pylab.arange(self._cqtN):
            self.DCT[ 0, j ] *= pylab.sqrt(2.0) / 2.0

    def _stft(self):
        if not self._have_x:
            print "Error: You need to load a sound file first: use self.load_audio('filename.wav')"
            return False
        fp = self._check_feature_params()
        WX = self._win_mtx(self.x, fp['wfft'], fp['nhop'], fp['nsamples'])
        self.STFT=pylab.rfft(WX, fp['nfft']).T
        self.STFT/=fp['nfft']
        self._have_stft=True
        if fp['verbosity']:
            print "Extracted STFT: nfft=%d, hop=%d" %(fp['nfft'], fp['nhop'])
        return True

    def _istftm(self, X_hat, Phi_hat=None):
        """
        ::

            Inverse short-time Fourier transform magnitude. Make a signal from a |STFT| transform.
            Uses phases from self.STFT if Phi_hat is None.
        """
        if not self._have_stft:
                return False        
        if Phi_hat is None:
            Phi_hat = pylab.exp( 1j * pylab.angle(self.STFT))
        fp = self._check_feature_params()
        X_hat = X_hat *  Phi_hat
        self.x_hat = self._overlap_add( pylab.real(fp['nfft'] * pylab.irfft(X_hat.T)) )
        if fp['verbosity']:
            print "Extracted iSTFTM->self.x_hat"
        return True


    def _power(self):
        if not self._have_stft:
            if not self._stft():
                return False
        fp = self._check_feature_params()
        self.POWER=(pylab.absolute(self.STFT)**2).sum(0)
        self._have_power=True
        if fp['verbosity']:
            print "Extracted POWER"
        return True

    def _cqft(self):
        """
        ::

            Constant-Q Fourier transform.
        """
        if not self._have_power:
            if not self._power():
                return False
        fp = self._check_feature_params()
        if fp['intensify']:
            self._cqft_intensified()
        else:
            self._make_log_freq_map()
            self.CQFT=pylab.array(pylab.sqrt(pylab.mat(self.Q)*pylab.mat(pylab.absolute(self.STFT)**2)))
            self._is_intensified=False
        self._have_cqft=True
        if fp['verbosity']:
            print "Extracted CQFT: intensified=%d" %self._is_intensified
        return True

    def _icqft(self, V_hat):
        """
        ::

            Inverse constant-Q Fourier transform. Make a signal from a constant-Q transform.
        """
        if not self._have_cqft:
                return False        
        fp = self._check_feature_params()
        X_hat = pylab.array( pylab.dot(self.Q.T, V_hat) ) * pylab.exp( 1j * pylab.angle(self.STFT) )
        self.x_hat = self._overlap_add( pylab.real(fp['nfft'] * pylab.irfft(X_hat.T)) )
        if fp['verbosity']:
            print "Extracted iCQFT->x_hat"
        return True

    def _overlap_add(self, X):
        wfft = self.feature_params['wfft']
        nhop = self.feature_params['nhop']
        x = pylab.zeros((X.shape[0] - 1)*nhop + wfft)
        for k in range(X.shape[0]):
            x[ k * nhop : k * nhop + wfft ] += X[ k, 0 : wfft ]
        return x
    
    def _cqft_intensified(self):
        """
        ::

            Constant-Q Fourier transform using only max abs(STFT) value in each band
        """
        if not self._have_stft:
            if not self._stft():
                return False
        self._make_log_freq_map()
        r,b=self.Q.shape
        b,c=self.STFT.shape
        self.CQFT=pylab.zeros((r,c))
        for i in pylab.arange(r):
            for j in pylab.arange(c):
                self.CQFT[i,j] = (self.Q[i,:]*pylab.absolute(self.STFT[:,j])).max()
        self._have_cqft=True
        self._is_intensified=True
        return True

    def _mfcc(self): 
        """
        ::

            DCT of the Log magnitude CQFT 
        """
        fp = self._check_feature_params()
        if not self._cqft():
            return False
        self._make_dct()
        AA = pylab.log10(self.CQFT)
        self.MFCC = pylab.dot(self.DCT, AA)
        self._have_mfcc=True
        if fp['verbosity']:
            print "Extracted MFCC: lcoef=%d, ncoef=%d, intensified=%d" %(fp['lcoef'], fp['ncoef'], fp['intensify'])
        return True

    def _lcqft(self):
        """
        ::

            Apply low-lifter to MFCC and invert to CQFT domain
        """
        fp = self._check_feature_params()
        if not self._mfcc():
            return False
        a,b = self.CQFT.shape
        a = (a-1)*2
        n=fp['ncoef']
        l=fp['lcoef']
        AA = self.MFCC[l:l+n,:] # apply Lifter
        self.LCQFT = 10**pylab.dot( self.DCT[l:l+n,:].T, AA )
        self._have_lcqft=True
        if fp['verbosity']:
            print "Extracted LCQFT: lcoef=%d, ncoef=%d, intensified=%d" %(fp['lcoef'], fp['ncoef'], fp['intensify'])
        if not self._have_hcqft:
            self._hcqft() # compute complement
        return True

    def _hcqft(self):
        """
        ::

            Apply high lifter to MFCC and invert to CQFT domain
        """
        fp = self._check_feature_params()
        if not self._mfcc():
            return False
        a,b = self.CQFT.shape
        n=fp['ncoef']
        l=fp['lcoef']
        AA = self.MFCC[n+l:a,:] # apply Lifter
        self.HCQFT=10**pylab.dot( self.DCT[n+l:a,:].T, AA)
        self._have_hcqft=True
        if fp['verbosity']:
            print "Extracted HCQFT: lcoef=%d, ncoef=%d, intensified=%d" %(fp['lcoef'], fp['ncoef'], fp['intensify'])
        if not self._have_lcqft:
            self._lcqft() # compute complement
        return True

    def _chroma(self):
        """
        ::
    
            Chromagram, like 12-BPO CQFT modulo one octave. Energy is folded onto first octave.
        """
        fp = self._check_feature_params()
        if not self._cqft():
            return False
        a,b = self.CQFT.shape
        complete_octaves = a/12 # integer division, number of complete octaves
        #complete_octave_bands = complete_octaves * 12
        # column-major ordering, like a spectrogram, is in FORTRAN order
        self.CHROMA=pylab.zeros((12,b))
        for k in pylab.arange(complete_octaves):
            self.CHROMA +=  self.CQFT[k*12:(k+1)*12,:]
        self.CHROMA = (self.CHROMA / complete_octaves)**0.5
        self._have_chroma=True
        if fp['verbosity']:
            print "Extracted CHROMA: intensified=%d" %fp['intensify']
        return True

    def _chroma_hcqft(self):
        """
        ::

            Chromagram formed by high-pass liftering in cepstral domain, then usual 12-BPO folding.
        """
        fp = self._check_feature_params()
        if not self._hcqft():
            return False
        a,b = self.HCQFT.shape
        complete_octaves = a/12 # integer division, number of complete octaves
        #complete_octave_bands = complete_octaves * 12
        # column-major ordering, like a spectrogram, is in FORTRAN order
        self.CHROMA=pylab.zeros((12,b))
        for k in pylab.arange(complete_octaves):
            self.CHROMA +=  self.HCQFT[k*12:(k+1)*12,:]
        self.CHROMA = (self.CHROMA / complete_octaves)**0.5
        self._have_chroma=True
        if fp['verbosity']:
            print "Extracted HCQFT CHROMA: lcoef=%d, ncoef=%d, intensified=%d" %(fp['lcoef'], fp['ncoef'], fp['intensify'])
        return True

    def valid_features(self):
        """
        ::

            Valid feature extractors:
            stft - short-time Fourier transform
            power- per-frame power
            cqft - constant-Q Fourier transform
            mfcc - Mel-frequency cepstral coefficients
            lcqft - low-cepstra constant-Q Fourier transform
            hcqft - high-cepstra constant-Q Fourier transform
            chroma - 12-chroma-band pitch-class profile
        """

        print """Valid feature extractors:
        stft - short-time Fourier transform
        cqft - constant-Q Fourier transform
        mfcc - Mel-frequency cepstral coefficients
        lcqft - low-cepstra constant-Q Fourier transform
        hcqft - high-cepstra constant-Q Fourier transform
        chroma - 12-chroma-band pitch-class profile
        """

