from music21 import converter, instrument, note, chord
import numpy
import glob
from keras.utils import np_utils

notes = []
faulty = ['BachCelloSuite.mid']

for file in glob.glob("midi_examples/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = []

    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instrument parts
        # only looking at files for violin (single instrument)
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print('data processed')
print(len(notes))

sequence_length = 50  # dealing with relatively short pieces here
pitch_names = sorted(set(item for item in notes))
pitch_to_int = {pitch: index for index, pitch in enumerate(pitch_names)}

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    # grab sequence of sequence_length consecutive notes
    sequence_in = notes[i:i + sequence_length]
    # output note after grabbed sequence
    sequence_out = notes[i + sequence_length]
    # input grabbed sequence converted to ints
    network_input.append([pitch_to_int[char] for char in sequence_in])
    # output note converted to int
    network_output.append(pitch_to_int[sequence_out])

num_patterns = len(network_input)
num_vocab = len(set(notes))


# reshape the input into a format compatible with LSTM layers
# resize to (num samples, input shape (sequence_length, 1 (num features)))
network_input = numpy.reshape(
    network_input, (num_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(num_vocab)
network_output = np_utils.to_categorical(network_output)
