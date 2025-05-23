from typing import Iterable, Optional
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State
import random
from itertools import permutations


# Nim, NimState, and NimMove implementations
class Nim(Game):
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, heaps: Iterable[int] = (7, 7, 7), first_player: Player = None, second_player: Player = None):
        self.first_player = first_player or Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        self.second_player = second_player or Player(self.SECOND_PLAYER_DEFAULT_CHAR)

        state = NimState(self.first_player, self.second_player, heaps)
        super().__init__(state)


class NimMove(Move):
    def __init__(self, heap: int, n: int):
        self.heap = heap
        self.n = n

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, NimMove):
            return False
        return self.heap == value.heap and self.n == value.n


class NimState(State):
    def __init__(self, current_player: Player, other_player: Player, heaps: Iterable[int]):
        self.heaps = tuple(heaps)
        assert self.heaps and all(heap >= 0 for heap in self.heaps)

        super().__init__(current_player, other_player)

    def get_moves(self) -> Iterable[NimMove]:
        return [NimMove(i, n) for i, heap in enumerate(self.heaps) for n in range(1, heap + 1)]

    def make_move(self, move: NimMove) -> 'NimState':
        assert move.n > 0 and self.heaps[move.heap] >= move.n
        heaps = tuple(heap - move.n if i == move.heap else heap for i, heap in enumerate(self.heaps))
        return NimState(self._other_player, self._current_player, heaps)

    def is_finished(self) -> bool:
        return all(heap == 0 for heap in self.heaps)

    def get_winner(self) -> Optional[Player]:
        if not self.is_finished():
            return None
        else:
            return self._current_player

    def __str__(self) -> str:
        text = [('Winner: ' if self.is_finished() else 'Current player: ') + self._current_player.char]

        for i, heap in enumerate(self.heaps):
            text.append(str(i + 1) + ': ' + '|' * heap)

        return '\n'.join(text)


def evaluate_nim_state(state: NimState) -> int:
    """
    Returns:
        - Positive value for advantageous positions.
        - Negative value for disadvantageous positions.
        - Zero for neutral positions.
    """
    heaps = state.heaps
    nim_sum = 0
    for heap in heaps:
        nim_sum ^= heap

    non_empty_heaps = [heap for heap in heaps if heap > 0]
    num_non_empty = len(non_empty_heaps)

    # Base evaluation: Nim-Sum
    if nim_sum == 0:
        base_score = -10  # Losing position if Nim-sum is zero
    else:
        base_score = 10  # Winning position if Nim-sum is non-zero

    # Endgame considerations (1 or 2 heaps left)
    if num_non_empty == 1:
        last_heap = non_empty_heaps[0]
        if last_heap == 1:  # If the last heap is size 1
            return -100  # Losing
        else:
            return 100  # Winning
    elif num_non_empty == 2:
        # Two heaps: Evaluate based on their sizes
        heap1, heap2 = non_empty_heaps
        if heap1 == heap2 == 1:  # Both heaps are size 1
            return -100  # Losing
        elif heap1 == 1 or heap2 == 1:  # One heap is size 1
            return 50  # Advantageous
        else:
            return 20  # Neutral position

    # Midgame considerations (more than 2 heaps)
    # Favor positions with uneven heap sizes to maximize opponent mistakes
    uneven_heaps = sum(1 for heap in non_empty_heaps if heap % 2 != 0)
    heap_sum = sum(non_empty_heaps)

    # Combine factors into a score
    score = (
        base_score + 
        5 * num_non_empty +  # More heaps generally means more options
        3 * uneven_heaps -  # Penalize even heap distributions
        2 * heap_sum  # Discourage leaving large sums of heaps
    )

    return score


def minimax(state: NimState, depth: int, alpha: float, beta: float, maximizing_player: bool):
    if depth == 0 or state.is_finished():
        # Evaluate the state
        if state.is_finished():
            score = 1 if state.get_winner() == state._current_player else -1
        else:
            score = evaluate_nim_state(state)  
        return score, None

    best_value = float('-inf') if maximizing_player else float('inf')
    best_moves = []

    for move in state.get_moves():
        child_state = state.make_move(move)
        value, _ = minimax(child_state, depth - 1, alpha, beta, not maximizing_player)

        if maximizing_player:
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
            alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
            beta = min(beta, best_value)

        if alpha >= beta:
            break

    # Select a random move if there are multiple best moves
    best_move = random.choice(best_moves) if best_moves else None
    return best_value, best_move


# Simulate the Game
def play_nim_with_minimax():
    game = Nim([7,7,7])  # Heaps definition
    state = game.state
    depth_player_1 = 5
    depth_player_2 = 1

    print(state)
    while not state.is_finished():
        if state._current_player == game.first_player:
            print("\nPlayer 1's turn:")
            _, move = minimax(state, depth_player_1, float('-inf'), float('inf'), True)
        else:
            print("\nPlayer 2's turn:")
            _, move = minimax(state, depth_player_2, float('-inf'), float('inf'), True)

        print(f"Move: Remove {move.n} from heap {move.heap + 1}")
        state = state.make_move(move)
        print(state)

    winner = state.get_winner()
    print(f"\nGame Over! Winner: {winner.char}")

def simulate_games(depth_player_1: int, depth_player_2: int, num_games: int = 100):
    """
    Simulate games with specific minimax depths for each player.

    Parameters:
        depth_player_1: Depth for Player 1's minimax search.
        depth_player_2: Depth for Player 2's minimax search.
        num_games: Number of games to simulate.

    Returns:
        A dictionary with the win counts for each player.
    """
    wins = {1: 0, 2: 0}  # Count of wins for Player 1 and Player 2

    for _ in range(num_games):
        game = Nim([1,2,3])
        state = game.state

        while not state.is_finished():
            if state._current_player == game.first_player:
                _, move = minimax(state, depth_player_1, float('-inf'), float('inf'), True)
            else:
                _, move = minimax(state, depth_player_2, float('-inf'), float('inf'), True)

            state = state.make_move(move)

        winner = state.get_winner()
        if winner == game.first_player:
            wins[1] += 1
        else:
            wins[2] += 1

    return wins

def test_depth_impact():
    # Define depth values to test
    depths = [1,4,8,10]
    num_games = 100  # Number of games per configuration
    
    print("Testing depth impact on win rates...\n")
    print(f"{'Player 1 Depth':<15}{'Player 2 Depth':<15}{'Player 1 Wins':<15}{'Player 2 Wins':<15}{'Higher depth wins':<15}")
    print("-" * 60)
    res = []
    # Iterate over unique depth combinations (e.g., (1, 4) or (4, 4))
    for depth_1, depth_2 in permutations(depths, 2):
        results = simulate_games(depth_1, depth_2, num_games)

        is_good_result = "True" if (depth_1 > depth_2 and results[1] > results[2]) or  (depth_1 < depth_2 and results[1] < results[2]) else "False"
        res.append(is_good_result)
        print(f"{depth_1:<15}{depth_2:<15}{results[1]:<15}{results[2]:<15}{is_good_result:<15}")
    ones = [r for r in res if r == "True"]
    print(len(ones)/len(res))

if __name__ == "__main__":
    # play_nim_with_minimax()
    test_depth_impact()
