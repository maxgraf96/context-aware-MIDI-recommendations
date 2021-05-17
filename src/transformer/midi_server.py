import os
import tempfile
from multiprocessing import Process, Value
from pathlib import Path
from threading import Thread

import mido
from audiolazy import lazy_midi  # For converting midi strings to midi numbers
from mido import MidiFile, MidiTrack, Message, MetaMessage

from main_model import generate_from_scratch, init_transformer_model, generate_from_midi_file

ppqn = 24
ppbar = ppqn * 4  # in 4/4 metre

shared_bool = False

def write_midi(notes):
    filename = 'mido_created.mid'

    mid = MidiFile()
    mid.ticks_per_beat = 24
    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to acoustic grand piano
    # See https://www.noterepeat.com/articles/how-to/213-midi-basics-common-terms-explained
    # for all default MIDI instruments
    instrument = 0
    track.append(Message('program_change', program=instrument, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    for note_msg in notes:
        track.append(note_msg)

    mid.save(filename)
    return filename


def sanity_check(notes):
    # 1) All note ons have note offs
    ons = sum(1 for note in notes if note.type == 'note_on')
    offs = sum(1 for note in notes if note.type == 'note_off')
    clear = ons == offs
    if not clear:
        print("Found some notes without note offs, appending with quarter length...")
        # Create note offs a quarter note later
        # Get rogue last few note ons (without note off events)
        rogue_notes = []
        i = -1
        last_note = notes[i]
        while last_note is not None and last_note.type == 'note_on':
            rogue_notes.append(last_note)
            i -= 1
            # Take previous note
            last_note = notes[i] if notes[i] is not None else None
        # Create note off events for rogue notes
        for i in range(len(rogue_notes)):
            note = rogue_notes[i]
            if i == 0:
                new_note = Message('note_off', note=note.note, velocity=note.velocity,
                                   time=ppqn)
            else:
                new_note = Message('note_off', note=note.note, velocity=note.velocity,
                                   time=0)
            notes.append(new_note)

    return notes


def midi_loop(data, name=''):
    print(mido.get_input_names())
    port = mido.open_input('IAC Driver Bus 1')

    notes = []
    bar_counter = 0
    generate_every_n_bars = 2
    # Current tick
    tick = 1
    last_event_tick = 0

    while True:
        for msg in port.iter_pending():
            # Get message type
            if tick > 1 and msg.type == 'note_on':
                # Get time difference in ticks to last event
                last_event_delta = tick - last_event_tick
                msg.velocity = 80
                # Create note_on for note
                note_on = Message('note_on', note=msg.note, velocity=msg.velocity, time=last_event_delta)
                # Append note on
                notes.append(note_on)
                last_event_tick = tick
            if tick > 1 and msg.type == 'note_off':
                # Get time difference in ticks to last event
                last_event_delta = tick - last_event_tick
                msg.velocity = 80
                note_off = Message('note_off', note=msg.note, velocity=msg.velocity, time=last_event_delta)
                notes.append(note_off)
                last_event_tick = tick

            # MIDI clock runs at 24ppqn (pulses per quarter note)
            # => 96 pulses =^= 1 bar
            if msg.type == 'clock':
                tick += 1
                if tick % ppbar == 0:
                    bar_counter += 1
                    print(bar_counter)
                if bar_counter > 0 and tick % (generate_every_n_bars * ppbar) == 0:
                    if len(notes) == 0:
                        # Need to have at least one note on
                        continue
                    print("Writing MIDI for ", generate_every_n_bars, " bars.")
                    # Sanity checks
                    notes = sanity_check(notes)

                    # Create MIDI file
                    write_midi(notes)

                    # Clear the note list and reset tick
                    notes.clear()
                    tick = 1
                    bar_counter = 0
                    data.value = True



if __name__ == '__main__':
    # Initialise transformer
    model = init_transformer_model()

    shared_bool = Value('b', False)

    # Generate bars
    process = Process(target=midi_loop, args=(shared_bool, 'shared_bool'))
    process.start()
    while(True):
        if shared_bool.value == 1:
            generate_from_midi_file('mido_created.mid')
            shared_bool.value = False
            print("Done writing MIDI")


    process.join()




