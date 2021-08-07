"""Games, or Adversarial Search."""

from collections import namedtuple
import random
import operator
import numpy as np
import time
from copy import deepcopy
import pandas as pd
from datetime import datetime

infinity = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, board, moves')
TOTAL_NODES = 20000
STARTING_MAPS = 100
# EPSILON = 0  # 0 2 4 8 16
PARANOID_EPSILON = 2
SHALLOW_EPSILON = 36
promised = 0
max_sum = 64
test_player_index = 3
# ______________________________________________________________________________


def maxN_search(state, game, maxsum):
    """Search game to determine best action """

    def place_in_arr(move):
        if move == 'R':
            return 0
        elif move == "G":
            return 1
        elif move == "Y":
            return 2
        elif move == "B":
            return 3

    def leaf_value(board):
        """Difference in the number of coins."""
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        return np.array([coin_red, coin_green, coin_yellow, coin_blue])

    def maxN(state):
        global max_sum, n_players, max_depth, all_vectors
        n_players = 4
        if len(state.moves) == 0:
            return leaf_value(state.board), 1
        possible_moves = state.moves
        possibilities = [(move, (game.result(state, move)).utility[place_in_arr(state.to_move)]) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1), reverse=True)
        new_moves = []
        for x, y in possibilities:
            new_moves.append(tuple(x))
        state = state._replace(moves=new_moves)
        first_optional_move = state.moves[0]
        state_new = game.result(state, first_optional_move)
        best, nodes = maxN(state_new)
        nodes += 1
        for child in range(1, len(state.moves)):
            current, nodes = maxN(
                game.result(state, state.moves[child]))
            nodes += 1
            if current[place_in_arr(state.to_move)] > best[place_in_arr(state.to_move)]:
                best = current
            if nodes > TOTAL_NODES:
                break
        #print(nodes)
        return best, nodes

    best_score = -infinity
    best_action = None
    possible_moves = game.actions(state)
    possibilities = [(move, (game.result(state, move)).utility[place_in_arr(state.to_move)]) for move in possible_moves]
    possibilities.sort(key=operator.itemgetter(1), reverse=True)
    moves = []
    for x, y in possibilities:
        moves.append(tuple(x))
    state = state._replace(moves=moves)
    for a in game.actions(state):
        state = game.result(state, a)
        v, count = maxN(state)
        if v[0] > best_score:
            best_score = v[0]
            best_action = a
    return best_action


def optimal_cutoff_search(state, game, maxsum, max_depth):
    """Search game to determine best action """

    def place_in_arr(move):
        if move == 'R':
            return 0
        elif move == "G":
            return 1
        elif move == "Y":
            return 2
        elif move == "B":
            return 3

    def leaf_value(board):
        """Difference in the number of coins."""
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        return np.array([coin_red, coin_green, coin_yellow, coin_blue])

    def Call_shallow(state, max_depth):
        v, selected_state, nodes = Shallow2(state, 64, 0, 0, max_depth)
        past_board = state.board
        future_board = selected_state.board
        best_action = {k: future_board[k] for k in set(future_board)-set(past_board)}
        best_action = list(best_action.keys())[0]
        return best_action, nodes

    def Shallow2(state, bound, nodes, depth, max_depth):
        global n_players, all_vectors
        n_players = 4
        if len(state.moves) == 0:
            return leaf_value(state.board), state, nodes + 1
        if depth > max_depth:
            return state.utility, state, nodes + 1
        possible_moves = state.moves
        possibilities = [(move, (game.result(state, move)).utility[place_in_arr(state.to_move)]) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1), reverse=True)
        new_moves2 = []
        for x, y in possibilities:
            new_moves2.append(tuple(x))
        state = state._replace(moves=new_moves2)
        first_optional_move = state.moves[0]
        nodes += 1
        best, recursive_state, nodes = Shallow2(game.result(state, first_optional_move), maxsum, nodes, depth + 1, max_depth)
        for child in range(1, len(state.moves)):
            if best[place_in_arr(state.to_move)] >= bound:
                if depth == 0:
                    return best, recursive_state, nodes
                else:
                    return best, state, nodes
            current, tmp_state, nodes = Shallow2(game.result(state, state.moves[child]), maxsum - best[place_in_arr(state.to_move)],
                                      nodes, depth + 1, max_depth)
            if current[place_in_arr(state.to_move)] > best[place_in_arr(state.to_move)]:
                best = current
                recursive_state = tmp_state
        if depth == 0:
            return best, recursive_state, nodes
        else:
            return best, state, nodes


    return Call_shallow(state, max_depth)


def suboptimal_cutoff_search(state, game, maxsum, epsilon, max_depth):
    """Search game to determine best action """

    player = game.to_move(state)

    def place_in_arr(move):
        if move == 'R':
            return 0
        elif move == "G":
            return 1
        elif move == "Y":
            return 2
        elif move == "B":
            return 3

    def leaf_value(board):
        """Difference in the number of coins."""
        if (len(board) < 64):
            print("less than 64")
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        return np.array([coin_red, coin_green, coin_yellow, coin_blue])

    def  Call_Bounded_Pruning(state, max_depth):
        global SHALLOW_EPSILON, promised
        promised = 0
        v, selected_state, nodes = Bounded_Pruning2(state, 64, SHALLOW_EPSILON, 0, 0, max_depth)
        past_board=state.board
        future_board=selected_state.board
        best_action={k : future_board[k] for k in set(future_board)-set(past_board)}
        best_action=list(best_action.keys())[0]
        return best_action, nodes

    def Bounded_Pruning2(state, bound, epsilon, nodes, depth, max_depth):  # this function get action and return value (vector values)
        global promised
        # global max_sum, n_players, all_vectors
        n_players = 4
        max_sum = 64
        if len(state.moves) == 0:
            return leaf_value(state.board), state, nodes + 1
        if depth > max_depth:
            return state.utility, state, nodes + 1
        possible_moves = state.moves
        possibilities = [(move, (game.result(state, move)).utility[place_in_arr(state.to_move)]) for move in
                         possible_moves]
        possibilities.sort(key=operator.itemgetter(1), reverse=True)
        new_moves3 = []
        for x, y in possibilities:
            new_moves3.append(tuple(x))
        state = state._replace(moves=new_moves3)
        first_optional_move = state.moves[0]
        nodes += 1
        b_best, recursive_state, nodes = Bounded_Pruning2(game.result(state, first_optional_move), max_sum, epsilon, nodes, depth + 1,max_depth)
        for child in range(1, len(state.moves)):
            if b_best[place_in_arr(state.to_move)] >= bound:  # not sure which pruning
                if depth == 0:
                    return b_best, recursive_state, nodes
                else:
                    return b_best, state, nodes

            if state.to_move == 'B' and depth == 0:
                promised = max(promised, b_best[place_in_arr(state.to_move)])
                if promised + epsilon >= max_sum:
                    return b_best, recursive_state, nodes

            if state.to_move == 'R' and depth == 1 and promised + epsilon >= max_sum - b_best[place_in_arr(state.to_move)] and not promised == 0:  # change promised to bound
                return np.array([0] * (n_players - 1) + [-1]), state, nodes

            current, tmp_state, nodes = Bounded_Pruning2(game.result(state, state.moves[child]), max_sum - b_best[place_in_arr(state.to_move)], epsilon, nodes, depth + 1,
                                              max_depth)
            if current[place_in_arr(state.to_move)] > b_best[place_in_arr(state.to_move)]:
                b_best = current
                recursive_state = tmp_state
        if depth == 0:
            return b_best, recursive_state, nodes
        else:
            return b_best, state, nodes

    return Call_Bounded_Pruning(state, max_depth)
    #best_score = -infinity
    #best_action = None
    #nodes_arr = []
    # promised=0
    # v, best_action , nodes = Bounded_Pruning2(game.result(state, a), maxsum, epsilon, nodes, 1, max_depth)
    # for a in game.actions(state):
    #     nodes = 0
    #     v, nodes = Bounded_Pruning2(game.result(state, a), maxsum, epsilon, nodes, 1, max_depth)
    #     nodes_arr.append(nodes)
    #     if v[0] > best_score:
    #         best_score = v[0]
    #         best_action = a


def paranoid_cutoff_search(state, game, maxsum, max_depth):
    player_paranoid = game.to_move(state)

    def get_player_before_paranoid(move):
        if move == 'R':
            return "B"
        elif move == "G":
            return 'R'
        elif move == "Y":
            return "G"
        elif move == "B":
            return "Y"

    def place_in_arr(move):
        if move == 'R':
            return 0
        elif move == "G":
            return 1
        elif move == "Y":
            return 2
        elif move == "B":
            return 3

    def leaf_value(board):
        """Difference in the number of coins."""
        if (len(board) < 64):
            print("less than 64")
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        arr = np.array([coin_red, coin_green, coin_yellow, coin_blue])
        return arr[place_in_arr(player_paranoid)]

    def calculate_utility(vector):  # need to change
        return 2 * vector[place_in_arr(player_paranoid)] - (sum(vector))

        #if player == "max":
        #   return 2 * vector[place_in_arr(player_paranoid)] - (sum(vector))
        #else:
        #    return -2 * (vector[place_in_arr(player_paranoid)] - (sum(vector)))

    def max_value(state, alpha, beta, depth, max_depth, nodes):
        if len(state.moves) == 0:
            return leaf_value(state.board), nodes + 1
        if depth > max_depth:
            return calculate_utility(state.utility), nodes + 1
        v = -infinity
        possible_moves = game.actions(state)
        possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1), reverse=True)
        new_moves5 = []
        for x, y in possibilities:
            new_moves5.append(tuple(x))
        state = state._replace(moves=new_moves5)
        for a in game.actions(state):
            new_state = game.result(state, a)
            value_from_min, nodes = min_value(new_state, alpha, beta, depth + 1, max_depth, nodes)
            v = max(v, value_from_min)
            if v >= beta:
                return v, nodes
            alpha = max(alpha, v)
        return v, nodes

    def min_value(state, alpha, beta, depth, max_depth, nodes):
        if len(state.moves) == 0:
            return leaf_value(state.board), nodes + 1
        if depth > max_depth:
            return calculate_utility(state.utility), nodes + 1
        v = infinity
        possible_moves = game.actions(state)
        possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1))
        new_moves6 = []
        for x, y in possibilities:
            new_moves6.append(tuple(x))
        state = state._replace(moves=new_moves6)
        for a in game.actions(state):
            new_state = game.result(state, a)
            if state.to_move == get_player_before_paranoid(paranoid_player_color):
                value_from_max, nodes = max_value(new_state, alpha, beta, depth + 1, max_depth, nodes)
                v = min(v, value_from_max)
            else:
                value_from_min, nodes = min_value(new_state, alpha, beta, depth + 1, max_depth, nodes)
                v = min(v, value_from_min)
            if v <= alpha:
                return v, nodes
            beta = min(beta, v)
        return v, nodes

    paranoid_player_color = state.to_move
    best_score = -infinity
    beta = infinity
    best_action = None
    nodes_arr = []
    possible_moves = game.actions(state)
    possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
    possibilities.sort(key=operator.itemgetter(1), reverse=True)
    new_moves4 = []
    for x, y in possibilities:
        new_moves4.append(tuple(x))
    state = state._replace(moves=new_moves4)
    for a in game.actions(state):
        v, nodes = max_value(game.result(state, a), best_score, beta, 1, max_depth, 0)
        nodes_arr.append(nodes)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action, sum(nodes_arr)


def bounded_paranoid_cutoff_search(state, game, maxsum, epsilon, max_depth):
    player_paranoid = game.to_move(state)
    #bestU = infinity
    #bestL = -infinity

    def get_player_before_paranoid(move):
        if move == 'R':
            return "B"
        elif move == "G":
            return 'R'
        elif move == "Y":
            return "G"
        elif move == "B":
            return "Y"

    def place_in_arr(move):
        if move == 'R':
            return 0
        elif move == "G":
            return 1
        elif move == "Y":
            return 2
        elif move == "B":
            return 3

    def leaf_value(board):
        """Difference in the number of coins."""
        if (len(board) < 64):
            print("less than 64")
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        arr = np.array([coin_red, coin_green, coin_yellow, coin_blue])
        val = arr[place_in_arr(player_paranoid)]
        return [val, val]

    def calculate_utility(vector):  # need to change
        val = 2 * vector[place_in_arr(player_paranoid)] - (sum(vector))
        return [val, val]
        #if player == "max":
        #    return 2 * vector[place_in_arr(player_paranoid)] - (sum(vector))
        #else:
        #    return -2 * (vector[place_in_arr(player_paranoid)] - (sum(vector)))

    def max_value(state, alpha, beta, depth, max_depth, epsilon, nodes):
        #nonlocal bestU, bestL
        v = [-infinity, infinity]
        if len(state.moves) == 0:
            return leaf_value(state.board), nodes + 1
        if depth > max_depth:
            return calculate_utility(state.utility), nodes + 1
        bestU = -infinity
        possible_moves = game.actions(state)
        possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1), reverse=True)
        new_moves7 = []
        for x, y in possibilities:
            new_moves7.append(tuple(x))
        state = state._replace(moves=new_moves7)
        for a in game.actions(state):
            new_state = game.result(state, a)
            value_from_min, nodes = min_value(new_state, alpha, beta, depth + 1, max_depth, epsilon, nodes)
            bestU = max(bestU, value_from_min[1])
            v[0] = max(v[0], value_from_min[0])
            alpha = max(alpha, v[0])
            if beta <= alpha + epsilon:
                return v, nodes

        v[1] = bestU
        return v, nodes

    def min_value(state, alpha, beta, depth, max_depth, epsilon, nodes):
        #nonlocal bestU, bestL
        v = [-infinity, infinity]
        if len(state.moves) == 0:
            return leaf_value(state.board), nodes + 1
        if depth > max_depth:
            return calculate_utility(state.utility), nodes + 1
        bestL = infinity
        possible_moves = game.actions(state)
        possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
        possibilities.sort(key=operator.itemgetter(1))
        new_moves8 = []
        for x, y in possibilities:
            new_moves8.append(tuple(x))
        state = state._replace(moves=new_moves8)
        for a in game.actions(state):
            new_state = game.result(state, a)
            if state.to_move == get_player_before_paranoid(paranoid_player_color):
                value_from_max, nodes = max_value(new_state, alpha, beta, depth + 1, max_depth, epsilon, nodes)
                bestL = min(bestL, value_from_max[0])
                v[1] = min(v[1], value_from_max[1])
            else:
                value_from_min, nodes = min_value(new_state, alpha, beta, depth + 1, max_depth, epsilon, nodes)
                bestL = min(bestL, value_from_min[0])
                v[1] = min(v[1], value_from_min[1])
            beta = min(beta, v[1])
            if beta <= alpha + epsilon:
                return v, nodes

        v[0] = bestL
        return v, nodes

    paranoid_player_color = state.to_move
    best_score = -infinity
    beta = infinity
    best_action = None
    nodes_arr = []
    possible_moves = game.actions(state)
    possibilities = [(move, calculate_utility((game.result(state, move)).utility)) for move in possible_moves]
    possibilities.sort(key=operator.itemgetter(1), reverse=True)
    new_moves9 = []
    for x, y in possibilities:
        new_moves9.append(tuple(x))
    state = state._replace(moves=new_moves9)
    for a in game.actions(state):
        val, nodes = max_value(game.result(state, a), best_score, beta, 1, max_depth, epsilon, 0)
        nodes_arr.append(nodes)
        if val[0] > best_score:
            best_score = val[0]
            best_action = a
    return best_action, sum(nodes_arr)

# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move_string = input('Your move? ')
    try:
        move = eval(move_string)
    except NameError:
        move = move_string
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    game.display(state)
    print()
    print('random_player turn')
    print('board have {} tiles'.format(len(state.board)))
    return random.choice(game.actions(state)), 1, 0


def maxN_player(game, state):
    maxsum = 64
    game.display(state)
    print()
    print('maxN_player turn')
    print('board have {} tiles'.format(len(state.board)))
    return maxN_search(state, game, maxsum)


def optimal_player(game, state):
    maxsum = 64
    game.display(state)
    print()
    nodes_count = 0
    max_depth = 1
    action_to_return = None
    action = None
    print('optimal_player turn')
    print('board have {} tiles'.format(len(state.board)))
    while (nodes_count <= TOTAL_NODES and maxsum-len(state.board)>=max_depth):
        max_depth += 1
        action_to_return = action
        action, nodes = optimal_cutoff_search(state, game, maxsum, max_depth)
        if nodes_count + nodes <= TOTAL_NODES:
            nodes_count += nodes
        else:
            break

    if action_to_return is None and max_depth==2:
        if len(state.board)==63:
            max_depth -= 1
        print("nodes_count" + str(nodes))
        print("max_depth" + str(max_depth))
        return action, nodes, max_depth
    max_depth -= 1
    print("nodes_count" + str(nodes_count))
    print("max_depth" + str(max_depth))
    return action_to_return, nodes_count, max_depth


def paranoid_player(game, state):
    maxsum = 64
    game.display(state)
    print()
    nodes_count = 0
    max_depth = 1
    action_to_return = None
    action = None
    print('paranoid_player turn')
    print('board have {} tiles'.format(len(state.board)))
    while (nodes_count <= TOTAL_NODES and maxsum-len(state.board)>=max_depth):

        max_depth += 1
        action_to_return = action
        action, nodes = paranoid_cutoff_search(state, game, maxsum, max_depth)
        if nodes_count + nodes <= TOTAL_NODES:
            nodes_count += nodes
        else:
            break

    if action_to_return is None and max_depth==2:
        if len(state.board)==63:
            max_depth -= 1
        print("nodes" + str(nodes))
        print("max_depth" + str(max_depth))
        return action, nodes, max_depth
    max_depth -= 1
    print("nodes_count" + str(nodes_count))
    print("max_depth" + str(max_depth))
    return action_to_return, nodes_count, max_depth


def bounded_paranoid_player(game, state):
    maxsum = 64
    game.display(state)
    epsilon = PARANOID_EPSILON
    print()
    nodes_count = 0
    max_depth = 1
    action_to_return = None
    action = None
    print('bounded_paranoid_player turn')
    print('board have {} tiles'.format(len(state.board)))
    while (nodes_count <= TOTAL_NODES and maxsum - len(state.board) >= max_depth):

        max_depth += 1
        action_to_return = action
        action, nodes = bounded_paranoid_cutoff_search(state, game, maxsum, epsilon,  max_depth)
        if nodes_count + nodes <= TOTAL_NODES:
            nodes_count += nodes
        else:
            break

    if action_to_return is None and max_depth == 2:
        if len(state.board) == 63:
            max_depth -= 1
        print("nodes" + str(nodes))
        print("max_depth" + str(max_depth))
        return action, nodes, max_depth
    max_depth -= 1
    print("nodes_count" + str(nodes_count))
    print("max_depth" + str(max_depth))
    return action_to_return, nodes_count, max_depth


def suboptimal_player(game, state):
    global promised, max_sum
    promised = 0
    game.display(state)
    print()
    epsilon = SHALLOW_EPSILON
    nodes_count = 0
    max_depth = 1
    action_to_return = None
    action = None
    print('suboptimal_player turn')
    print('board have {} tiles'.format(len(state.board)))
    while nodes_count <= TOTAL_NODES and max_sum - len(state.board) >= max_depth:
        max_depth += 1
        action_to_return = action
        action, nodes = suboptimal_cutoff_search(state, game, max_sum, epsilon, max_depth)
        if nodes_count + nodes <= TOTAL_NODES:
            nodes_count += nodes
        else:
            break
    if action_to_return is None and max_depth == 2:
        if len(state.board) == 63:
            max_depth -= 1
        print("nodes" + str(nodes))
        print("max_depth" + str(max_depth))
        return action, nodes, max_depth
    max_depth -= 1
    print("nodes_count" + str(nodes_count))
    print("max_depth" + str(max_depth))
    return action_to_return, nodes_count, max_depth


def weak_player(game, state):
    global max_sum
    game.display(state)
    print()
    print('weak_player turn')
    print('board have {} tiles'.format(len(state.board)))
    weak_max_depth = 0
    action, nodes = optimal_cutoff_search(state, game, max_sum, weak_max_depth)
    print("nodes_count" + str(nodes))
    print("max_depth" + str(weak_max_depth))
    return action, nodes, weak_max_depth
# ______________________________________________________________________________
# Some Sample Games


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)


class Reversi(Game):
    """Reversi game."""

    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        init_red_pos = [(4, 4)]
        init_green_pos = [(5, 5)]
        init_yellow_pos = [(4, 5)]
        init_blue_pos = [(5, 4)]
        init_red_board = dict.fromkeys(init_red_pos, 'R')
        init_green_board = dict.fromkeys(init_green_pos, 'G')
        init_yellow_board = dict.fromkeys(init_yellow_pos, 'Y')
        init_blue_board = dict.fromkeys(init_blue_pos, 'B')
        board = {**init_red_board, **init_green_board, **init_yellow_board, **init_blue_board}
        moves = self.get_valid_moves(board, 'R')
        self.initial = GameState(
            to_move='R', utility=[0, 0, 0, 0], board=board, moves=moves)

    def capture_enemy_in_dir(self, board, move, player, delta_x_y):
        """Returns true if any enemy is captured in the specified direction."""
        enemy_list = ['B', 'G', 'Y', 'R']
        enemy_list.remove(player)

        (delta_x, delta_y) = delta_x_y
        x, y = move
        x, y = x + delta_x, y + delta_y
        enemy_list_0 = []
        # if board.get((x, y)) in enemy_list:
        # enemy = board.get((x, y))
        while board.get((x, y)) in enemy_list:
            enemy_list_0.append((x, y))
            x, y = x + delta_x, y + delta_y
        if board.get((x, y)) != player:
            del enemy_list_0[:]

        x, y = move
        x, y = x - delta_x, y - delta_y
        enemy_list_1 = []
        # if board.get((x, y)) in enemy_list:
        # enemy = board.get((x, y))
        while board.get((x, y)) in enemy_list:
            enemy_list_1.append((x, y))
            x, y = x - delta_x, y - delta_y
        if board.get((x, y)) != player:
            del enemy_list_1[:]

        return enemy_list_0 + enemy_list_1

    def enemy_captured_by_move(self, board, move, player):
        return self.capture_enemy_in_dir(board, move, player, (0, 1)) \
               + self.capture_enemy_in_dir(board, move, player, (1, 0)) \
               + self.capture_enemy_in_dir(board, move, player, (1, -1)) \
               + self.capture_enemy_in_dir(board, move, player, (1, 1))

    def actions(self, state):
        """Legal moves."""
        return state.moves

    def player_not_exist_in_board(self, board, player):
        for state in board.keys():
            if board[state] == player:
                return False
        return True

    def link_to_exist_coin(self, board, move):
        x, y = move
        x_new, y_new = x + 0, y + 1
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x + 1, y + 0
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x + 0, y - 1
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x - 1, y + 0
        if (x_new, y_new) in board.keys():
            return True
        '''x_new, y_new = x + 1, y + 1
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x - 1, y - 1
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x - 1, y + 1
        if (x_new, y_new) in board.keys():
            return True
        x_new, y_new = x + 1, y - 1
        if (x_new, y_new) in board.keys():
            return True'''
        return False

    def get_valid_moves(self, board, player):
        """Returns a list of valid moves for the player judging from the board."""
        if self.player_not_exist_in_board(board, player):  # all his disk was eaten
            return self.new_place_possibilities(board)
        else:
            valid_moves = [(x, y) for x in range(1, self.width + 1)
                           for y in range(1, self.height + 1)
                           if (x, y) not in board.keys() and
                           self.enemy_captured_by_move(board, (x, y), player)]
            if len(valid_moves) == 0:
                return self.new_place_possibilities(board)
            else:
                return valid_moves

    def new_place_possibilities(self, board):
        return [(x, y) for x in range(1, self.width + 1)
                for y in range(1, self.height + 1)
                if (x, y) not in board.keys() and
                self.link_to_exist_coin(board, (x, y))]

    def result(self, state, move):
        # Invalid move
        if move not in state.moves:
            return state

        opponent_player = self.return_next_player(state.to_move)
        board = state.board.copy()
        board[move] = state.to_move  # Show the move on the board
        # Flip enemy
        for enemy in self.enemy_captured_by_move(board, move, state.to_move):
            board[enemy] = state.to_move
        # Regenerate valid moves
        moves = self.get_valid_moves(board, opponent_player)
        return GameState(to_move=opponent_player, utility=self.compute_utility(board), board=board,
                         moves=moves)

    def return_next_player(self, player):
        if player == "R":
            return 'G'
        elif player == "G":
            return 'Y'
        elif player == "Y":
            return 'B'
        elif player == "B":
            return 'R'

    def utility(self, state):
        return state.utility  # if player == 'B' else -state.utility

    def terminal_test(self, state):
        return len(state.moves) == 0

    def display(self, state):
        board = state.board
        print('coin_diff = ' + str(self.coin_diff(board)))
        print('choice_diff = ' + str(self.choice_diff(board)))
        print('corner_diff = ' + str(self.corner_diff(board)))
        for y in range(0, self.height + 1):
            for x in range(0, self.width + 1):
                if x > 0 and y > 0:
                    if (x, y) in state.moves:
                        print(board.get((x, y), '_', ), end=' ')
                    else:
                        print(board.get((x, y), '.', ), end=' ')
                if x == 0:
                    if y > 0:
                        print(y, end=' ')
                if y == 0:
                    print(x, end=' ')
            print()

    def compute_utility(self, board):
        coin = self.coin_diff(board)
        choice = self.choice_diff(board)
        corner = self.corner_diff(board)
        if corner != 0:
            utility = [coin[0] * 0.4 + choice[0] * 0.3 + corner[0] * 0.3,
                       coin[1] * 0.4 + choice[1] * 0.3 + corner[1] * 0.3,
                       coin[2] * 0.4 + choice[2] * 0.3 + corner[2] * 0.3,
                       coin[3] * 0.4 + choice[3] * 0.3 + corner[3] * 0.3]
        else:
            utility = [coin[0] * 0.4 + choice[0] * 0.3 + 0 * 0.3,
                       coin[1] * 0.4 + choice[1] * 0.3 + 0 * 0.3,
                       coin[2] * 0.4 + choice[2] * 0.3 + 0 * 0.3,
                       coin[3] * 0.4 + choice[3] * 0.3 + 0 * 0.3]

        arr = np.array(utility)
        sum_arr = sum(arr)
        new_arr = (arr / sum_arr) * 64
        return np.array(new_arr)

    def coin_diff(self, board):
        """Difference in the number of coins."""
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        return [coin_red, coin_green, coin_yellow, coin_blue]

    def choice_diff(self, board):
        """Difference in the number of choices available."""
        red_moves_num = len(self.get_valid_moves(board, 'R'))
        green_moves_num = len(self.get_valid_moves(board, 'G'))
        yellow_moves_num = len(self.get_valid_moves(board, 'Y'))
        blue_moves_num = len(self.get_valid_moves(board, 'B'))
        sum_move = red_moves_num + green_moves_num + yellow_moves_num + blue_moves_num
        if sum_move != 0:
            return [100 * red_moves_num / sum_move, 100 * green_moves_num / sum_move,
                    100 * yellow_moves_num / sum_move, 100 * blue_moves_num / sum_move]
        else:
            return [0, 0, 0, 0]

    def corner_diff(self, board):
        """Difference in the number of corners captured."""
        corner = [board.get((1, 1)), board.get((1, self.height)), board.get((self.width, 1)),
                  board.get((self.width, self.height))]
        red_corner = corner.count('R')
        green_corner = corner.count('G')
        yellow_corner = corner.count('Y')
        blue_corner = corner.count('B')
        sum_corner = red_corner + green_corner + yellow_corner + blue_corner
        if sum_corner != 0:
            return [100 * red_corner / sum_corner, 100 * green_corner / sum_corner,
                    100 * yellow_corner / sum_corner, 100 * blue_corner / sum_corner]
        else:
            return 0

    def create_start_maps(self):
        start_states = []
        number_of_maps = STARTING_MAPS
        counter = 0
        while counter < number_of_maps:
            random.seed(counter)
            random_count = 0
            state_to_add = self.initial
            while random_count < 12:
                move = random.choice(state_to_add.moves)
                state_to_add = self.result(state_to_add, move)
                random_count += 1
            start_states.append(state_to_add)
            counter += 1
        return start_states

    def play_game_from_map_list(self, games_players_vectors, state_list):
        global test_player_index
        total_nodes = {}
        game_utilities = {}
        final_hands = {}
        avg_depth = {}
        counter = 1
        for players_list in range(len(games_players_vectors)):
            total_nodes[players_list] = []
            game_utilities[players_list] = []
            final_hands[players_list] = []
            avg_depth[players_list] = []
        initial_time = time.perf_counter()
        for players_list in range(len(games_players_vectors)):
            for starting_state in state_list:
                print(counter)
                counter += 1
                curr_state = starting_state
                total_nodes_played = 0
                game_done = False
                avg_depth_list = []
                while not game_done:
                    for p in range(len(games_players_vectors[players_list])):
                        player = games_players_vectors[players_list][p]
                        move, nodes, max_depth_per_game = player(self, curr_state)
                        if p == test_player_index:
                            total_nodes_played += nodes
                            avg_depth_list.append(max_depth_per_game)
                        curr_state = self.result(curr_state, move)
                        if self.terminal_test(curr_state):
                            self.display(curr_state)
                            game_utilities[players_list].append(self.utility(curr_state))
                            game_done = True
                total_nodes[players_list].append(total_nodes_played)
                final_hands[players_list].append(game.calculate_winning(curr_state.board))
                avg_depth[players_list].append(np.mean(avg_depth_list))

        return game_utilities, total_nodes, final_hands, time.perf_counter() - initial_time, avg_depth

    def calculate_winning(self, board):
        coin_blue = sum(x == 'B' for x in board.values())
        coin_green = sum(x == 'G' for x in board.values())
        coin_yellow = sum(x == 'Y' for x in board.values())
        coin_red = sum(x == 'R' for x in board.values())
        win_arr = [coin_red, coin_green, coin_yellow, coin_blue]
        print(win_arr)
        return win_arr


    def players_lists(self, num_games = 100, players_in_game = 4, different_players_types  = 6, seed = 42):
        random.seed(seed)
        player_games = int((num_games * players_in_game) / different_players_types)
        players_groups = []
        players_bank = [[optimal_player] * player_games, [suboptimal_player] * player_games, [paranoid_player] * player_games, [bounded_paranoid_player] * player_games, [weak_player] * player_games, [random_player] * player_games]
        for game_index in range(num_games - 1):
            players_types_left = len([player_list for player_list in players_bank if len(player_list)>0])
            if players_types_left < players_in_game:
                break
            playing_indexes = set()
            game_players = []
            while len(game_players) < players_in_game:
                random_index = random.choice(range(different_players_types))
                if random_index not in playing_indexes and len(players_bank[random_index]) > 0:
                    playing_indexes.add(random_index)
                    player = players_bank[random_index].pop()
                    game_players.append(player)
            players_groups.append(game_players)
        return players_groups


    def play_tournament(self, games_players_vectors, state_list): # len(games_players_vectors) = len(state_list) = NUMBER OF GAMES
        global test_player_index
        total_nodes = []
        game_utilities = []
        final_hands = []
        avg_depth = []
        players_names = []
        counter = 1
        initial_time = time.perf_counter()
        for players_list in range(len(games_players_vectors)):  # number of games
            game_players_names = [player.__name__ for player in games_players_vectors[players_list]]
            n_players = len(games_players_vectors[players_list])
            print(counter)
            counter += 1
            curr_state = state_list[players_list]  #starting_state
            total_nodes_played = [0] * n_players
            game_done = False
            avg_depth_list = [[], [], [], []]  # Four Players hard coded :( will be changed
            while not game_done:
                for p in range(n_players):
                    player = games_players_vectors[players_list][p]
                    move, nodes, max_depth_per_game = player(self, curr_state)
                    total_nodes_played[p] += nodes
                    avg_depth_list[p].append(max_depth_per_game)
                    curr_state = self.result(curr_state, move)
                    if self.terminal_test(curr_state):
                        self.display(curr_state)
                        avg_depth_list = [np.mean(depth_list) for depth_list in avg_depth_list]
                        game_utilities.append(self.utility(curr_state))
                        game_done = True
            total_nodes.append(total_nodes_played)
            final_hands.append(game.calculate_winning(curr_state.board))
            avg_depth.append(avg_depth_list)
            players_names.append(game_players_names)
        return game_utilities, total_nodes, final_hands, time.perf_counter() - initial_time, avg_depth, players_names


game = Reversi()
players_lists = game.players_lists()
df_g1 = pd.DataFrame(columns=['game_utilities', 'total_nodes', 'final_hands', 'avg_depth', 'players_names'])
start_state_list = game.create_start_maps()
step = 8
for j in range(0, len(players_lists), step):
    game_utilities_end, total_nodes_end, final_hands_end, total_time_end, avg_depth_end, players_names_end = game.play_tournament(players_lists[j:j+step], start_state_list[j:j+step])

    for i in range(len(game_utilities_end)):

        entry1 = {'game_utilities': game_utilities_end[i], 'total_nodes': total_nodes_end[i],
                  'final_hands': final_hands_end[i], 'players_names': players_names_end[i],
                  'avg_depth': avg_depth_end[i]}
        df_g1 = df_g1.append(entry1, ignore_index=True)
    df_g1.to_csv(f'game{j + step}_new_tournament.csv', mode='w+')

