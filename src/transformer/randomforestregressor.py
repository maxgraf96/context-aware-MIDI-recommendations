import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from data_loading import generate_all_remi, init_data

if __name__ == '__main__':
    # Generate all REMI representations if they do not exist
    # This will generate remi segments and save them to a folder if they don't exist,
    # Otherwise just load pickled segments generated before
    generate_all_remi()

    # Prepare training data
    data = init_data(checkpoint_path='REMI-my-checkpoint', num_segments_limit=200)

    input = data[:, 0, 0, :]
    labels = data[:, 0, 1, -1]

    regr = RandomForestRegressor(max_depth=100, random_state=0)
    regr.fit(input, labels)

    # print(regr.predict([[0, 0, 0, 0]]))
    prediction = regr.predict([data[0, 0, 0, :]])
    prediction = int(prediction)
    print("Correct: ", data[0, 0, 1, -1])
    print("Predicted: ", prediction)