from music21 import converter, instrument, note, chord, stream, duration
import numpy as np
import glob
from keras.utils import np_utils
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
import sys
from fractions import Fraction

# TODO: need to refactor to save values to disk or pass around
########################################################################
########################################################################
########################################################################
notes = []
average_len = 0
file_count = 0

for file in glob.glob("midi_examples/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = []
    len_before = len(notes)

    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instrument parts
        # only looking at files for violin (single instrument)
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch) + ',' +
                         str(element.duration.quarterLength))
        elif isinstance(element, chord.Chord):
            chord_id = '.'.join(str(n) for n in element.normalOrder)
            notes.append(chord_id + ',' + str(element.duration.quarterLength))
    len_after = len(notes)
    file_count += 1
    if file_count == 1:
        average_len = len_after - len_before
    else:
        average_len += (len_after - len_before) / 2.0

print('data processed')

sequence_length = 40  # dealing with relatively short pieces here
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
normalized_input = np.reshape(
    network_input, (num_patterns, sequence_length, 1))

# normalize input
normalized_input = normalized_input / float(num_vocab)
# one hot output
network_output = np_utils.to_categorical(network_output)

print('inputs and outputs formatted')
########################################################################
########################################################################
########################################################################


model = Sequential()
model.add(LSTM(
    256,
    input_shape=(normalized_input.shape[1], normalized_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(num_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.load_weights('weights.hdf5')

start_note = np.random.randint(0, len(network_input) - 1)
int_to_pitch = {index: pitch for (index, pitch) in enumerate(pitch_names)}
sequence = network_input[start_note]
prediction_output = []
for note_index in range(400):
    prediction_input = np.reshape(sequence, (1, len(sequence), 1))
    # normalize as in training
    prediction_input = prediction_input / float(num_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_pitch[index]
    prediction_output.append(result)
    # append the prediction note and shift sequence over by one
    sequence.append(index)
    sequence = sequence[1:len(sequence)]

print('prediction sequence complete')

output_notes = []
output_notes.append(instrument.Violin())
offset = 0
# create note and chord objects based on the values generated by the model
for pattern in prediction_output:
    pitch_and_duration = pattern.split(',')
    pitch = pitch_and_duration[0]
    pitch_duration = float(Fraction((pitch_and_duration[1])))
    # pitch is a chord
    if ('.' in pitch) or pitch.isdigit():
        notes_in_chord = pitch.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.duration = duration.Duration(pitch_duration)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pitch is a note
    else:
        new_note = note.Note(pitch)
        new_note.duration = duration.Duration(pitch_duration)
        new_note.offset = offset
        output_notes.append(new_note)
    # increase offset each iteration so that notes do not stack
    offset += pitch_duration

print('note decoding complete')
print('writing to midi')
# print(output_notes)
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output3.mid')
