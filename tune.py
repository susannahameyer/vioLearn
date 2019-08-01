# Susannah Meyer
# Below resource for explanation on FFT and Hann window and formulas used to convert bt MIDI and frequency
# http://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
# https://newt.phys.unsw.edu.au/jw/notes.html

import numpy as np
import pyaudio

######################################################################
# range of notes for violin that I play: G3 - C7 
# notes to MIDI numbers: https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
# FRAME_SIZE and FRAMES_PER_FFT are powers of two

LOWEST_NOTE = 55    # G3
HIGHEST_NOTE = 96   # C7

SAMPLING_RATE = 22050   # in Hz
FRAME_SIZE = 2048   # num samples per frame
FRAMES_PER_FFT = 32 # num frames for FFT to average across
SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT # increasing this will decrease FREQ_STEP_SIZE and increase resolution but also increase time needed
FREQ_STEP_SIZE = float(SAMPLING_RATE) / SAMPLES_PER_FFT

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_midi(f): 
    # add 69, the MIDI number of A4, which is the standard reference note
    # the formula appended to 69 gives the number of semitones from A4
    return 69 + 12 * np.log2(f / 440.0)

def midi_to_freq(n): 
    # subtract 69, the MIDI number of A4
    # formula without 69 is frequency of a note that lies n semitones higher than A4, so we want the base number added
    return 440 * 2.0 ** ((n - 69) / 12.0)

# min/max index within FFT of notes we care about
def note_to_fftbin(n): 
    return midi_to_freq(n)/FREQ_STEP_SIZE
imin = max(0, int(np.floor(note_to_fftbin(LOWEST_NOTE-1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(HIGHEST_NOTE+1))))

# Allocate space to run an FFT. 
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

# Initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=SAMPLING_RATE,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)

stream.start_stream()

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print('sampling at', SAMPLING_RATE, 'Hz with max resolution of', FREQ_STEP_SIZE, 'Hz')
print()

# As long as we are getting data:
while stream.is_active():

    # Shift the buffer down and new data in
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
    buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16)

    # Run the FFT on the windowed buffer
    fft = np.fft.rfft(buf * window)

    # Get frequency of maximum response in range
    freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP_SIZE

    # Get note number and nearest note
    n = freq_to_midi(freq)
    n0 = int(round(n))

    # Console output once we have a full buffer
    num_frames += 1

    if num_frames >= FRAMES_PER_FFT:
        print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
            freq, NOTE_NAMES[n0 % 12], n-n0))
