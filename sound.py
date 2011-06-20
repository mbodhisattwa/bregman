# sound.py - audio file I/O and play functionality
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

import os
import numpy
import subprocess
import error

# audio file handling
try:
    import scikits.audiolab
    HAVE_AUDIOLAB=True
except ImportError:
    HAVE_AUDIOLAB=False
import wave 

# WaveOpen class
class WavOpen:
    """
    ::

        WavOpen: sound-file handling class
        wav = WavOpen(filename)
    """
    def __init__(self,arg):
        self.initialize_from_file(arg)

    def initialize_from_file(self,filename):
        self.filename = filename
        sound = wave.open(self.filename, "r")
        self.sample_rate = sound.getframerate()
        print 'sample_rate=%i' %self.sample_rate
        self.num_channels = sound.getnchannels()
        print 'num_channels=%i' %self.num_channels
        self.sample_width = sound.getsampwidth()
        print 'sample_width=%i' %self.sample_width
        self.num_frames = sound.getnframes()
        print 'num_frames=%i, num_secs=%f' %(self.num_frames, self.num_frames/self.sample_rate)
        self.buffer_size = 2048
        self.bytes_per_frame = self.sample_width * self.num_channels
        print 'bytes_per_frame=%i' %self.bytes_per_frame
        self.bytes_per_second = self.sample_rate * self.bytes_per_frame
        print 'bytes_per_second=%i' %self.bytes_per_second
        self.bytes_per_buffer = self.buffer_size * self.bytes_per_frame
        print 'bytes_per_buffer=%i' %self.bytes_per_buffer
        rawdata = sound.readframes(self.num_frames)
        if rawdata:
            signal = wave.struct.unpack('%ih' %self.num_frames*self.num_channels, rawdata) # transform to signal
            del rawdata
            self.sig = numpy.zeros((self.num_frames,), dtype='float32')
            for index in range(self.num_frames):
                self.sig[index] = signal[index*self.num_channels]                
            self.sig = self.sig / (numpy.max(self.sig)+.0)

def wav_write(signal, wav_name, sample_rate):
    """
    ::

        Utility routine for writing wav files, use scikits.audiolab if available
        otherwise uses wave module
    """
    return _wav_write(signal, wav_name, sample_rate)

def _wav_write(signal, wav_name, sample_rate):
    """
    ::

        Utility routine for writing wav files, use scikits.audiolab if available
    """
    if HAVE_AUDIOLAB:
        scikits.audiolab.wavwrite(signal, wav_name, sample_rate)
    else:
        w = wave.Wave_write(wav_name)
        if not w:
            print "Error opening file named: ", wav_name
            raise error.BregmanError()
        w.setparams((1,2,sample_rate,signal.size,'NONE','NONE'))
        b_signal = '' # C-style binary string
        for i in range(len(signal)):
            b_signal += wave.struct.pack('h',32767*signal[i]) # transform to C-style binary string
        w.writeframes(b_signal)
        w.close()
    return True

def wav_read(wav_name):
    """
    ::

        Utility routine for reading wav files, use scikits.audiolab if available
        otherwise uses wave module.
    """
    return _wav_read(wav_name)

def _wav_read(wav_name):
    """
    ::

        Utility routine for reading wav files, use scikits.audiolab if available
        otherwise uses wave module.
    """
    if HAVE_AUDIOLAB:
        signal, sample_rate, pcm = scikits.audiolab.wavread(wav_name)
        return (signal, sample_rate)
    else:
        wav=WavOpen(wav_name)
        return (wav.sig, wav.sample_rate)

# Define sound player helper functions
AUDIO_TMP_FILE = ".tmp.wav"
sound_options = {"soundplayer": "open"}

# The play() function from audiolab
if HAVE_AUDIOLAB:
    from scikits.audiolab import *

# Bregman's own play_snd(...) function
try: # OSX / Linux   
    dummy_path = os.environ['HOME']
    def play_snd(data, sample_rate=44100):
        """
        ::

            Bregman Linux/OSX/Windows sound player function.
            data - a numpy array
            sample_rate - default 44100
        """
        m = abs(data).max() + 0.001
        if  m > 1.0: data /= m
        _wav_write(data, AUDIO_TMP_FILE, sample_rate)
        command = [sound_options['soundplayer'], AUDIO_TMP_FILE]
        res = subprocess.call(command)
        if res:
            print "Error in ", command
            raise error.BregmanError()
        return res            
except: # Windows     
    import winsound
    dummy_path = os.environ['HOMEPATH']
    def play_snd(data, sample_rate=44100):            
        """
        ::

            Bregman Linux/OSX/Windows sound player function.
            data - a numpy array
            sample_rate - default 44100
        """
        m = abs(data).max() + 0.001
        if  m > 1.0: data /= m
        _wav_write(data, AUDIO_TMP_FILE, sample_rate)
        winsound.PlaySound(AUDIO_TMP_FILE, winsound.SND_FILENAME|winsound.SND_ASYNC)
