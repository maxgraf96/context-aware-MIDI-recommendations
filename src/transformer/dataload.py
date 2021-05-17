import pickle

import numpy as np
import pandas as pd


def extract_segments_from_one_dataframe(df):
    # Convert to numpy array
    array = df.to_numpy()
    # Get array height as number
    array_height = array.shape[0]

    # Define segment height
    seg_height = 50

    # Define list for that specific dataframe
    # This will contain all the segments and labels in tuples (see below)
    segments_and_labels_list = []

    # Iterate over numpy array
    for i in range(0, array_height, seg_height):
        # Don't include last few notes if they don't nicely align to 50
        # Only happens once at the end of the loop
        if i + seg_height > array_height:
            break

        # Current segment: From i to i + segment height
        # Will be 50x4
        current_segment = array[i:i + seg_height, :]

        # Magic function to see whether the current segment contains trills
        # has_trill = magic_trill_detection_function()
        # Set to True now for demonstration purposes
        has_trill = True

        # Convert the current segment and has_trill into a tuple
        # This will explicitly store the segment and label as a pair and make data access later super easy
        # See https://www.w3schools.com/python/python_tuples.asp
        my_tuple = (current_segment, has_trill)

        # Append tuple of current segment and label to list
        segments_and_labels_list.append(my_tuple)

        # Just a note: Later you can access the tuple elements by
        seg = my_tuple[0]
        label = my_tuple[1]

    return segments_and_labels_list




if __name__ == '__main__':
    # Define your list that you're gonna pickle at the end
    mama_list = []

    # Create data frame with random numbers as bogus data
    num_rows = 2320
    num_cols = 4
    df = pd.DataFrame(np.random.randint(0, 100, size=(num_rows, num_cols)), columns=list('ABCD'))

    # This will later be inside a loop where you iterate over all the dataframes for all the different MIDI files
    # Here it's done just for one
    segments_and_labels = extract_segments_from_one_dataframe(df)

    # Append all the tuples you just generated to the mama list
    # Extend vs. append bc append will add the "segments_and_labels" as a whole list to "mama_list"
    # So you'd get a list of lists which is ugly
    mama_list.extend(segments_and_labels)

    # Mama list now contains all the (segment, label) tuples
    # Dump into pickle
    with open('data.pkl', "wb") as output_file:
        pickle.dump(mama_list, output_file)

    # ......
    # ......
    # ......

    # Load later with
    with open('data.pkl', "rb") as input_file:
        mama_list = pickle.load(input_file)

    # Love you bb <3