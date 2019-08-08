from music21 import converter, instrument, note, chord
import numpy as np
import glob
from keras.utils import np_utils
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

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
network_input = np.reshape(
    network_input, (num_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(num_vocab)
# one hot output
network_output = np_utils.to_categorical(network_output)

print('inputs and outputs formatted')

model = Sequential()
model.add(LSTM(
    256,
    input_shape=(network_input.shape[1], network_input.shape[2]),
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

print('model created')

filepath = "weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=200,
          batch_size=64, callbacks=callbacks_list)

print('model trained')
