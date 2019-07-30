# Susannah Meyer
# Below resource for explanation on FFT and Hann window
# http://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf

import numpy as np
import pyaudio

######################################################################
# range of notes for violin that I play: G3 - C7 
# notes to MIDI numbers: https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
# FRAME_SIZE and FRAMES_PER_FFT are powers of two

LOWEST_NOTE = 55    # G3
HIGHEST_NOTE = 96   # C7

SAMP_FREQ = 22050   # in Hz
FRAME_SIZE = 2048   # num samples per frame
FRAMES_PER_FFT = 16 # num frames for FFT to average across
SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT # increasing this will decrease FREQ_STEP and increase resolution but also increase time needed
FREQ_STEP = float(SAMP_FREQ) / SAMPLES_PER_FFT

NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

