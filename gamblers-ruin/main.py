"""
Solution to the gambler's ruin problem. Setup:
- Player A has a tokens at the start of the game.
- Player B has b tokens at the start of the game.
- In each round player A can win 1 token with probability p and lose 1 token with probability 1-p.

Solution: Probability distribution of the tokens for player A after n rounds.
"""

import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Starting number of tokens for player A.")
    parser.add_argument("-b", help="Starting number of tokens for player B.")
    parser.add_argument("-n", help="Number of rounds.")
    parser.add_argument("-p", help="Probability of winning.")
    args = parser.parse_args()

    a = int(args.a)
    b = int(args.b)
    n = int(args.n)
    p = float(args.p)

    num_states = a + b + 1 # Player A can have 0, 1, ..., a + b tokens

    transition_matrix = np.zeros(shape=(num_states, num_states))
    transition_matrix[0, 0] = transition_matrix[-1, -1] = 1

    first_row = np.zeros(num_states)
    first_row[0] = p
    first_row[2] = 1 - p

    for row in range(1, num_states-1):
        transition_matrix[row] = np.roll(first_row.copy(), row - 1)

    initial_state_distribution = np.zeros(num_states)
    initial_state_distribution[a] = 1

    solution = initial_state_distribution.T @ np.linalg.matrix_power(transition_matrix, n)

    print(pd.DataFrame({"p": solution.round(2)}))