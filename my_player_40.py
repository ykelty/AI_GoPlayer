import math
import random
from copy import deepcopy

# Define constants for the game
BLACK = 1
WHITE = 2
EMPTY = 0
BOARD_SIZE = 5
KOMI = 2.5
MAX_MOVES = 24


class LearningMinimaxAgent:

    def __init__(self, side=None, previous_board=None, alpha=0.5, gamma=0.8):
        self.side = side
        self.consecutive_passes = 0
        self.move_count = 0
        self.ko_state = None
        self.previous_board = previous_board

        self.alpha = alpha
        self.gamma = gamma

        self.default_weights = {
            'stability': 1.0,
            'capture': 1.0,
            'territory': 1.0,
            'stone_weight': 1.0,
            'liberty': 1.0
        }

        self.weights = self.default_weights.copy()

        self.opponent_capture_attempts = 0
        self.opponent_aggressive_moves = 0
        self.opponent_random_moves = 0

    def set_side(self, side):
        self.side = side

    def move(self, board, opponent_move=None):
        if self.is_game_over(board):
            return None

        if opponent_move:
            #print("OBSERVING", opponent_move)
            self.observe_opponent_move(board, opponent_move)

        self.adjust_weights_opponent()

        if self.previous_board is not None and self.last_move_capture(board):

            self.ko_state = self.tuple_board(self.previous_board)
            print("KO state updated due to opponent capture:", self.ko_state)

        the_evaluation, best_move = self.minimax_with_learned_weights(board, depth=4, alpha=-math.inf, beta=math.inf,
                                                                      maximizing_player=True)

        if best_move:
            # Simulate the move to check captures and apply the KO rule
            simulated_board = deepcopy(board)
            simulated_board[best_move[0]][best_move[1]] = self.side

            captured = self.remove_captured_stones(simulated_board, opponent=self.get_opponent())

            #print("Simulated move on board:")
            #self.print_board(simulated_board)
            #print("KO state:", self.ko_state)

            # Update KO state only if a capture is made
            # if captured > 0:
            #    self.ko_state = self.tuple_board(board)  # Store the previous state for KO comparison

            # Apply the move on the actual board and update tracking variables
            board[best_move[0]][best_move[1]] = self.side
            self.consecutive_passes = 0
            self.move_count += 1
        else:
            self.consecutive_passes += 1  # Pass if no valid moves available

        #print("Updated previous_board after move:")
        self.previous_board = deepcopy(board)
        return best_move

    def observe_opponent_move(self, board, move):

        x, y = move
        if board[x][y] == self.get_opponent():
            if self.does_move_capture(board, x, y, self.side):
                self.opponent_capture_attempts += 1
            elif self.is_adjacent_to_my_stones(board, x, y):
                #if self.reduces_liberties(board, x, y):
                self.opponent_aggressive_moves += 1
            else:
                self.opponent_random_moves += 1

    def does_move_capture(self, board, x, y, opponent_side):
        temp_board = deepcopy(board)
        temp_board[x][y] = opponent_side
        return self.remove_captured_stones(temp_board, opponent_side) > 0

    def is_adjacent_to_my_stones(self, board, x, y):

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == self.side:
                # print("RETURN TRUE?")
                return True
        return False

    def reduces_liberties(self, board, x, y):
        temp_board = deepcopy(board)
        temp_board[x][y] = self.get_opponent()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == self.side:
                if (self.count_liberties(temp_board, self.get_group(temp_board, nx, ny)) <
                        self.count_liberties(board,self.get_group(board, nx, ny))):
                    return True
        return False

    def adjust_weights_opponent(self):
        # print("ADJUSTING")
        style = self.classify_opponent_style()
        if style == "Greedy":
            print("greedy opponent")
            self.weights['stability'] = 1.5
            self.weights['liberty'] = 1.5
            self.weights['capture'] = 1.5
            self.weights['territory'] = 1.0
            self.weights['stone_weight'] = 1.0
        elif style == "Aggressive":
            print("aggressive opponent")
            self.weights['stability'] = 1.7
            self.weights['liberty'] = 1.7
            self.weights['capture'] = 1.2
            self.weights['territory'] = 1.0
            self.weights['stone_weight'] = 1.5
        else:
            print("random opponent")
            self.weights['stability'] = 1.2
            self.weights['liberty'] = 1.2
            self.weights['capture'] = 1.0
            self.weights['territory'] = 1.3
            self.weights['stone_weight'] = 1.0

    def classify_opponent_style(self):
        # print("HELLO")
        total_moves = self.opponent_capture_attempts + self.opponent_aggressive_moves + self.opponent_random_moves
        # print("PASSING")
        # if total_moves < 5:
        #    return "Unknown"
        print("capture attempts", self.opponent_capture_attempts)
        print("aggressive moves", self.opponent_aggressive_moves)
        print("random moves", self.opponent_random_moves)
        #capture_ratio = self.opponent_capture_attempts / total_moves
        #aggressive_ratio = self.opponent_aggressive_moves / total_moves
        #random_ratio = self.opponent_random_moves / total_moves

        if self.opponent_capture_attempts >= 1:
            return "Greedy"
        elif self.opponent_aggressive_moves >= 1:
            return "Aggressive"
        else:
            return "Random"

    def detect_opponent_move(self, previous_board, current_board):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                #print(x,y)
                #print(current_board)
                if previous_board[x][y] == EMPTY and current_board[x][y] == self.get_opponent():
                    # print("DETECTED:", x, y)
                    return (x, y)
        return None


    def last_move_capture(self, board):
        if self.previous_board is None:
            return False

        stone_count_previous = sum(row.count(self.side) for row in self.previous_board)
        stone_count_current = sum(row.count(self.side) for row in board)

        capture_detected = stone_count_current < stone_count_previous
        if capture_detected:
            print("Opponent capture detected.")
        return capture_detected

    def get_opponent(self):
        return BLACK if self.side == WHITE else WHITE

    def remove_captured_stones(self, board, opponent):
        captured_positions = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == opponent and not self.liberties_check(board, x, y):
                    captured_positions.extend(self.get_group(board, x, y))

        for (x, y) in captured_positions:
            board[x][y] = EMPTY
        return len(captured_positions)

    def liberties_check(self, board, x, y):
        visited = set()
        return self.liberties_recursive(board, x, y, visited)

    def liberties_recursive(self, board, x, y, visited):
        if (x, y) in visited:
            return False

        visited.add((x, y))
        if board[x][y] == EMPTY:
            return True

        color = board[x][y]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if board[nx][ny] == EMPTY:
                    return True
                if board[nx][ny] == color and self.liberties_recursive(board, nx, ny, visited):
                    return True

        return False

    def get_group(self, board, x, y):
        color = board[x][y]
        group = []
        visited = set()
        self.group_recursive(board, x, y, color, group, visited)
        return group

    def group_recursive(self, board, x, y, color, group, visited):
        if (x, y) in visited:
            return

        visited.add((x, y))
        group.append((x, y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == color:
                self.group_recursive(board, nx, ny, color, group, visited)

    def calculate_capture_potential(self, board):
        # Count potential captures
        capture_score = 0
        opponent = self.get_opponent()
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == opponent and not self.liberties_check(board, x, y):
                    capture_score += 1
        return capture_score

    def minimax_with_learned_weights(self, board, depth, alpha, beta, maximizing_player):
        if self.is_game_over(board) or depth == 0:
            return self.evaluate_board(board), None

        if maximizing_player:
            max_eval = -math.inf
            best_move = None

            for move in self.get_ordered_moves(board, self.side):
                if self.is_valid_move(board, move[0], move[1]):
                    new_board = deepcopy(board)
                    new_board[move[0]][move[1]] = self.side

                    # Check for captures
                    self.remove_captured_stones(new_board, opponent=self.get_opponent())

                    eval, _ = self.minimax_with_learned_weights(new_board, depth - 1, alpha, beta, False)

                    if eval > max_eval:
                        max_eval = eval
                        best_move = move

                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break

            return max_eval, best_move

        else:
            min_eval = math.inf
            best_move = None
            opponent = self.get_opponent()

            for move in self.get_ordered_moves(board, opponent):
                if self.is_valid_move(board, move[0], move[1]):
                    new_board = deepcopy(board)
                    new_board[move[0]][move[1]] = opponent

                    # Check for captures
                    self.remove_captured_stones(new_board, opponent=self.get_opponent())

                    eval, _ = self.minimax_with_learned_weights(new_board, depth - 1, alpha, beta, True)

                    if eval < min_eval:
                        min_eval = eval
                        best_move = move

                    beta = min(beta, eval)
                    if beta <= alpha:
                        break

            return min_eval, best_move

    def evaluate_board(self, board):
        features = self.calculate_features(board)

        score = (
                self.weights['territory'] * features['territory'] +
                self.weights['stone_weight'] * features['stone_weight'] +
                self.weights['stability'] * features['stability'] +
                self.weights['capture'] * features['capture'] +
                self.weights['liberty'] * features['liberty']
        )
        return score

    def calculate_features(self, board):
        # Calculate the individual feature values for the current board state
        return {
            'territory': self.calculate_territory_control(board),
            'stone_weight': self.calculate_stone_weight(board),
            'stability': self.calculate_stone_stability(board),
            'capture': self.calculate_capture_potential(board),
            'liberty': self.calculate_liberty_count(board)
        }

    def calculate_territory_control(self, board):
        score = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == EMPTY:
                    # Check if the empty cell is surrounded more by agent’s stones
                    neighbors = self.get_neighbors(x, y)
                    agent_count = sum(1 for a, b in neighbors if board[a][b] == self.side)
                    opponent_count = sum(1 for a, b in neighbors if board[a][b] == self.get_opponent())

                    if agent_count > opponent_count:
                        score += 1
        return score

    def get_neighbors(self, x, y):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy in directions:
            a, b = x + dx, y + dy
            if 0 <= a < BOARD_SIZE and 0 <= b < BOARD_SIZE:
                neighbors.append((a, b))
        return neighbors

    def calculate_stone_weight(self, board):
        # Prioritize central positions over edges and corners
        weight = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.side:
                    if (x, y) in [(0, 0), (0, 4), (4, 0), (4, 4)]:
                        weight += 1
                    elif x == 0 or y == 0 or x == 4 or y == 4:
                        weight += 2
                    elif (x, y) in [(2, 1), (1, 2), (2, 3), (3, 2)]:
                        weight += 6
                    else:
                        weight += 5
        return weight

    def calculate_stone_stability(self, board):
        stability = 0
        visited = set()
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.side and (x, y) not in visited:
                    group = self.get_group(board, x, y)
                    visited.update(group)
                    stability += len(group)
        return stability

    def calculate_liberty_count(self, board):
        liberty_count = 0
        visited = set()
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.side and (x, y) not in visited:
                    group = self.get_group(board, x, y)
                    visited.update(group)
                    liberty_count += self.count_liberties(board, group)
        return liberty_count

    def count_liberties(self, board, group):
        liberties = set()
        for (x, y) in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == EMPTY:
                    liberties.add((nx, ny))
        return len(liberties)

    def tuple_board(self, board):
        return tuple(tuple(row) for row in board)

    def get_ordered_moves(self, board, side):
        move_scores = []
        vulnerable_groups = self.detect_vulnerable_groups(board)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == EMPTY:
                    new_board = deepcopy(board)
                    new_board[x][y] = side

                    features = self.calculate_features(new_board)
                    score = (self.weights['capture'] * features['capture']
                             + self.weights['stability'] * features['stability']
                             + self.weights['liberty'] * features['liberty']
                             + self.weights['territory'] * features['territory']
                             + self.weights['stone_weight'] * features['stone_weight'])

                    if side == self.side:
                        threat = self.detect_threats(new_board)
                        score += threat

                        for group, liberties in vulnerable_groups:
                            if any((nx, ny) in group for nx, ny in self.get_neighbors(x, y)):
                                score += 3 if liberties == 1 else 2

                        overextended_score = self.detect_overextended_opponent_stones(new_board)
                        score += overextended_score

                        connection_score = self.detect_connection_opportunities(new_board)
                        score += connection_score

                    move_scores.append(((x, y), score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        return [move[0] for move in move_scores]

    def detect_overextended_opponent_stones(self, board):
        score = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.get_opponent():
                    # Check the liberties of each opponent stone
                    liberties = self.count_liberties(board, self.get_group(board, x, y))
                    if liberties == 1:
                        score += 3  # Score boost for targeting stones with 1 liberty
                    elif liberties == 0:
                        score += 5  # Higher bonus for moves that capture opponent stones
        return score

    def detect_connection_opportunities(self, board):
        connection_score = 0
        vulnerable_groups = self.detect_vulnerable_groups(board)

        for group, liberties in vulnerable_groups:
            if liberties <= 2:
                for x, y in group:
                    for a, b in self.get_neighbors(x, y):
                        if board[a][b] == self.side:
                            connection_score += 3

        return connection_score

    def detect_vulnerable_groups(self, board):
        vulnerable_groups = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.side:
                    group = self.get_group(board, x, y)
                    liberties = self.count_liberties(board, group)
                    if liberties <= 2:
                        vulnerable_groups.append((group, liberties))
        return vulnerable_groups

    def detect_threats(self, board):
        score = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == self.side:
                    liberties = self.count_liberties(board, self.get_group(board, x, y))
                    adjacency = self.adjacent_to_opponent(board, x, y)
                    if liberties == 1 and adjacency:
                        score -= 10  # Higher penalty for groups with 1 liberty adjacent to opponent
                    elif liberties == 2 and adjacency:
                        score -= 5  # Moderate penalty for groups with 2 liberties adjacent to opponent
                    elif liberties == 1:
                        score -= 5
                    elif liberties == 2:
                        score -= 3
        return score

    def adjacent_to_opponent(self, board, x, y):
        for a, b in self.get_neighbors(x, y):
            if board[a][b] == self.get_opponent():
                return True
        return False

    def is_valid_move(self, board, x, y):
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == EMPTY:
            simulated_board = deepcopy(board)
            simulated_board[x][y] = self.side

            captured = self.remove_captured_stones(simulated_board, opponent=self.get_opponent())

            if self.tuple_board(simulated_board) == self.ko_state:
                print("KO rule violation detected:", self.tuple_board(simulated_board))
                return False

            if self.tuple_board(simulated_board) == self.previous_board:
                print("KO rule violation detected:", self.tuple_board(simulated_board))
                return False
            #print(x,y,"simulated board:", simulated_board[x][y])

            if self.previous_board is not None:
                #print("previous board:", self.previous_board[x][y])
                if simulated_board[x][y] == self.previous_board[x][y]:
                    return False

            # No-suicide check: ensure move does not result in a group with no liberties
            if not captured and not self.liberties_check(simulated_board, x, y):
                return False

            return True
        return False

    def print_board(self, board):
        for row in board:
            print(" ".join(str(cell) for cell in row))
        print()

    def is_game_over(self, board):
        if self.move_count >= MAX_MOVES:
            return True
        if self.consecutive_passes >= 2:
            return True
        return not any(self.is_valid_move(board, x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE))

    def determine_game_result(self, board):
        black_count = sum(row.count(BLACK) for row in board)
        white_count = sum(row.count(WHITE) for row in board) + KOMI

        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return 0

    def update_weights(self, features, reward):

        for feature in self.weights:
            self.weights[feature] += self.alpha * reward * features[feature]

    def train(self, num_games, initial_board):
        for i in range(num_games):
            board = deepcopy(initial_board)
            game_result = self.practice_game(board)
            self.learn_from_game_result(board, game_result)

    def learn_from_game_result(self, board, game_result):
        if game_result == self.side:
            reward = 1.0
        elif game_result == 0:
            reward = 0.5
        else:
            reward = -1.0
        features = self.calculate_features(board)
        self.update_weights(features, reward)

    def practice_game(self, board):
        self.move_count = 0
        self.consecutive_passes = 0

        while not self.is_game_over(board):
            move = self.move(board)
            if not move:
                break
        return self.determine_game_result(board)


def read_input():
    with open('input.txt', 'r') as f:
        color = int(f.readline().strip())
        previous_board = [list(map(int, list(f.readline().strip()))) for _ in range(BOARD_SIZE)]
        current_board = [list(map(int, list(f.readline().strip()))) for _ in range(BOARD_SIZE)]
    return color, previous_board, current_board


def write_output(move):
    with open('output.txt', 'w') as f:
        if move == "PASS":
            f.write("PASS\n")
        else:
            f.write(f"{move[0]},{move[1]}\n")


# Main game loop
if __name__ == '__main__':
    player, previous_board, current_board = read_input()

    learning_minimax_agent = LearningMinimaxAgent(side=player, previous_board=previous_board)

    opponent_move = learning_minimax_agent.detect_opponent_move(previous_board, current_board)

    # training_games = 1
    # training_board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    # learning_minimax_agent.train(training_games, training_board)

    learning_minimax_agent.consecutive_passes = 0
    learning_minimax_agent.move_count = 0
    if learning_minimax_agent.is_game_over(current_board):
        write_output("PASS")
    else:
        best_move = learning_minimax_agent.move(current_board, opponent_move=opponent_move)
        if best_move:
            write_output(best_move)
        else:
            write_output("PASS")
