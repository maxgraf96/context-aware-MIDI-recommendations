import os
import pickle
from datetime import datetime

import numpy as np

import utilities as utils
from constants import HIST_DEPTH
from data_loading import init_data, extract_events
from probas_containers import transitions, b_tree, create_histories

# Load event2word (vocabulary)
prefix = '' if 'transformer' in os.getcwd() else 'transformer/'
dictionary_path = prefix + '{}/dictionary.pkl'.format('REMI-my-checkpoint')
event2word, word2event = pickle.load(open(dictionary_path, 'rb'))

probas = None
tree = None


def load_data_structures():
    # Get data
    num_segments = 979
    data = init_data(checkpoint_path='REMI-my-checkpoint', num_segments_limit=num_segments)
    # Extract training sequence lists
    sequences = data[:, :, 0, :].reshape((-1, 512))
    # Convert to transition probability matrix
    probas = transitions(sequences)

    create = False
    if create:
        print("Creating histories, ", datetime.now().strftime("%H:%M:%S"))
        all_hists_list = create_histories(sequences, HIST_DEPTH)
        dall = {}
        print("Making mama histories dict, ", datetime.now().strftime("%H:%M:%S"))
        for d in all_hists_list:
            dall.update(d)

        # Magic b-tree experiment
        print("Building b-tree, ", datetime.now().strftime("%H:%M:%S"))
        tree = b_tree(dall)
        print("B-tree complete, ", datetime.now().strftime("%H:%M:%S"))
    else:
        # Load b_tree from disk
        tree = b_tree(None)

    return probas, tree

def is_tempo_token(word):
    return "Tempo" in word2event[word]

def generate(prompt=None):
    global probas, tree
    if probas is None or tree is None:
        probas, tree = load_data_structures()

    if not prompt:
        # Generate a random prompt
        words = []
        # Init new bar
        ws = [event2word['Bar_None']]
        # Set tempo
        tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
        tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
        # Load some random notes / chords
        chords = [v for k, v in event2word.items() if 'Chord' in k]
        ws.append(event2word['Position_1/16'])
        ws.append(np.random.choice(chords))
        ws.append(event2word['Position_1/16'])
        ws.append(np.random.choice(tempo_classes))
        ws.append(np.random.choice(tempo_values))
        words.append(ws)
    else:
        # Load prompt and start new bar
        events = extract_events('REMI-my-checkpoint', prompt)
        words = [[event2word['{}_{}'.format(e.name, e.value)] for e in events]]
        words[0].append(event2word['Bar_None'])


    # Sampling here
    n_target_bars = 16
    resort_counter = 0
    try_counter = 0
    for bar in range(n_target_bars):
        for i in range(100):
            try_counter += 1
            last_words = words[0][-HIST_DEPTH:]
            try:
                ret = tree.find(last_words)
                tokens = ret[0]
                probabilities = ret[1]
                new_word = np.random.choice(tokens, 1, p=probabilities)[0]
            # History sequence does not exist in database
            except:
                resort_counter += 1
                last_word = last_words[-1]
                new_word = np.random.choice(probas[last_word], 1)[0]

            words[0].append(new_word)

            # If bar event (only works for batch_size=1, so don't generate with batch_size > 1 currently)
            if new_word == event2word['Bar_None']:
                break

    print("Had to resort to transition probabilities ", 100 * resort_counter / try_counter, " of times.")

    # Filter duplicate bar tokens
    print("Deleting duplicate bar markers")
    to_delete_indices = []
    for index in range(len(words[0])):
        if index in to_delete_indices:
            continue

        if words[0][index] == event2word['Bar_None']:
            count_idx = index + 1
            try:
                while words[0][count_idx] == event2word['Bar_None']:
                    to_delete_indices.append(count_idx)
                    count_idx += 1
            except IndexError:
                break
        elif index > 10 and "Tempo" in word2event[words[0][index]]:
            to_delete_indices.append(index)

    cleaned_up = [i for j, i in enumerate(words[0]) if j not in to_delete_indices]

    print("Removing extra tempo markers")
    print("Writing MIDI...")

    utils.write_midi(
        words=cleaned_up,
        word2event=word2event,
        output_path='from_scratch_proba.mid',
        prompt_path=None,
        counter=0)


if __name__ == '__main__':
    prompt = None
    # prompt = 'data/evaluation/000.midi'
    generate(prompt)


