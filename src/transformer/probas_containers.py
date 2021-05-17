import os
import pickle

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

from collections import Counter
from multiprocessing import Process, Manager

import numpy as np
from tqdm import tqdm
from numba import jit

from constants import HIST_DEPTH, N_TOKENS, N_TAKES
from custom_b_tree import CustomBTree


def b_tree(hist):
    path = 'data_btree.pkl'
    exists = return_if_exists(path)
    if hist is None and exists is not False:
        print("Found existing b-tree.")
        return exists

    print()
    print("Creating b-tree")
    tree = CustomBTree()
    for key, value in tqdm(hist.items(), position=0, leave=True):
        if key:
            tree.add(key, value)

    # with open(path, "wb") as output_file:
    #     pickle.dump(tree, output_file)
    with open(path, 'wb') as f:
        pickle.dump(tree, f)

    return tree


all_hists = []


def create_histories(sequences, max_length):
    """
    Create max_length-1 histories of 2-max_length depth
    :param max_length: The maximum lookback length
    :return: List of all history dicts
    """
    print()
    # with Manager() as manager:
    #     manager_list = manager.list()  # <-- can be shared between processes.
    #
    #     # Only do 10 processes at a time!!
    #
    #     counter = 0
    #     step = 5
    #     while (counter != max_length):
    #         processes = []
    #         for i in range(counter, counter + step):
    #             # Start processes
    #             p = Process(target=histories, args=(sequences, i, manager_list))
    #             processes.append(p)
    #             p.start()
    #         for p in processes:
    #             # Start processes
    #             p.join()
    #
    #         counter += step
    #         if counter > max_length:
    #             break
    #
    #     # Merging all histories to one big list
    #     print("Merging histories...")
    #     all_hists = list(manager_list)


    for current_depth in tqdm(range(2, max_length), position=0, leave=True):
        cur_hist = histories(sequences, current_depth)
        all_hists.append(cur_hist)

    print("Finished creating histories")
    return all_hists


def histories(sequences, hist_depth=None, all_hists=None):
    path = 'database.pkl'
    # exists = return_if_exists(path)
    # if hist_depth is None and not exists:
    #     print("Found existing histories.")
    #     return exists

    # Doesn't exist -> create
    if hist_depth is None:
        hist_depth = HIST_DEPTH
    print("Creating history with depth ", hist_depth)
    database = {}
    for s in tqdm(sequences, position=0, leave=True):
        for index in s:
            if index <= hist_depth:
                continue
            prevs = s[index - hist_depth:index]
            if tuple(prevs) in database:
                database[tuple(prevs)].append(s[index])
            else:
                database[tuple(prevs)] = [s[index]]

    # Create pdfs
    print()
    print("Generating probabilities:")
    for key, item in tqdm(database.items(), position=0, leave=True):
        # Count occurrences in each item
        counts = Counter(item).most_common()
        tokens = [elem[0] for elem in counts]
        occurances = [elem[1] for elem in counts]
        # Normalise probabilities to [0...1]
        total = sum(occurances)
        probabilities = [x / total for x in occurances]
        database[key] = (tokens, probabilities)

    # with open(path, "wb") as output_file:
    #     pickle.dump(database, output_file)

    if all_hists is not None:
        all_hists.append(database)
    return database


def transitions(sequences):
    """
    Create transitions from list of sequences
    :param sequences:
    :return: Transition probability table
    """
    path = 'data_transition_table.pkl'
    exists = return_if_exists(path)
    if exists is not False:
        print("Found existing transition probability table.")
        return exists

    # Doesn't exist -> create
    # Size of the transition array
    n = N_TOKENS
    # Transition array, initially empty
    arr = np.zeros((n, n), dtype=int)
    for s in sequences:
        ind = (s[1:], s[:-1])  # Indices of elements for existing transitions
        arr[ind] += 1  # Add existing transitions

    # Normalize by columns and return as a DataFrame
    data = np.nan_to_num(arr / arr.sum(axis=0))

    transition_probabilities_matrix = {}
    for token in range(N_TOKENS):
        largest_indices = np.argpartition(data[:, token], -N_TAKES)[-N_TAKES:]
        largest_indices = largest_indices[np.argsort(data[:, token][largest_indices])]
        transition_probabilities_matrix[token] = largest_indices

    # with open(path, "wb") as output_file:
    #     pickle.dump(transition_probabilities_matrix, output_file)
    with open(path, 'wb') as f:
        pickle.dump(transition_probabilities_matrix, f)

    return transition_probabilities_matrix


def return_if_exists(path):
    """
    Helper function to load something from a file if it exists
    :param path: The path to the file
    :return: The content of the file (dict, list, b-tree, etc.) or False if it doesn't exist
    """
    exists = os.path.exists(path)
    if exists:
        with open(path, "rb") as input_file:
            file = pickle.load(input_file)

            return file
    else:
        return False
