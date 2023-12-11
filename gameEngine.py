import torch

class GameEngine:
    # returns the starting position
    def startingPosition(self):
        raise NotImplementedError

    # returns a hash of the given position
    def hash(self, state):
        del state
        raise NotImplementedError

    # returns a list of all possible moves
    def allMoves(self):
        raise NotImplementedError

    # returns True if the game is over
    def gameOver(self, state):
        del state
        raise NotImplementedError

    # returns 1 if the active player wins, -1 if the non-active player wins, 0 otherwise
    def outcome(self, state):
        del state
        raise NotImplementedError

    # returns the list of legal moves, the order must always be the same
    def legalMoves(self, state) -> list:
        del state
        raise NotImplementedError

    # plays move inplace
    def makeMove(self, state, move):
        del state, move
        raise NotImplementedError

    def copy(self, state):
        del state
        raise NotImplementedError

    def undoMove(self, state):
        del state
        raise NotImplementedError

    # Returns an encoded version of the last game state to be used as model input
    def encodeState(self, state, device=None):
        del state, device
        raise NotImplementedError

    # Returns an encoded version of the last game state that includes the outputs to predict
    def encodeStateAndOutput(self, state, policy, evaluation, device=None):
        del state, policy, evaluation, device
        raise NotImplementedError

    # Pretty prints the game
    def print(self, state, event=("?", "?"), players='', reversed=False):
        del state, event, players, reversed
        raise NotImplementedError


class InvalidBoardState(Exception):
    def __init__(self, state):
        super().__init__(state)


class InvalidMove(Exception):
    pass


TicTacToeMoves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

class TicTacToe(GameEngine):
    def startingPosition(self):
        return [torch.zeros((3,3), dtype=torch.int)]

    def hash(self, game):
        return hash(tuple(game[-1].numpy().flatten()))

    def allMoves(self):
        return TicTacToeMoves

    def gameOver(self, game):
        return self.outcome(game) != 0 or 0 not in game[-1]

    def outcome(self, game):
        state = game[-1]
        winner = -2
        for i in range(3):
            if state[i][0] != 0 and state[i][0] == state[i][1] == state[i][2]:
                winner = max(winner, state[i][0])
            if state[0][i] != 0 and state[0][i] == state[1][i] == state[2][i]:
                winner = max(winner, state[0][i])
        if state[0][0] != 0 and state[0][0] == state[1][1] == state[2][2]:
            winner = max(winner, state[0][0])
        if state[0][2] != 0 and state[0][2] == state[1][1] == state[2][0]:
            winner = max(winner, state[0][2])
        if winner == 1:
            raise InvalidBoardState(game)
        return -1 if winner == -1 else 0

    def legalMoves(self, game):
        moves = []
        for i in range(3):
            for j in range(3):
                if game[-1][i][j] == 0:
                    moves.append((i, j))
        return moves

    def makeMove(self, game, move):
        state = game[-1].clone()
        i, j = move
        if state[i][j] != 0:
            raise InvalidMove
        state[i][j] = 1
        game.append(-state)
        return game

    def copy(self, game):
        return [state.clone() for state in game]

    def undoMove(self, game):
        game.pop()
        return game

    def encodeState(self, game, device=None):
        return game[-1]

    def encodeStateAndOutput(self, game, policy, evaluation, device=None):
        return (
            self.encodeState(game).to(device),
            torch.tensor([policy.get(a, 0.) for a in TicTacToeMoves], dtype=torch.float, device=device),
            torch.tensor(evaluation, dtype=torch.float, device=device)
        )

    def print(self, game, event=("?", "?"), players=None, reversed=False):
        output = [[' ']*4*len(game) for _ in range(3)]
        for i, state in enumerate(game):
            if reversed:
                state = -state
            state = ["X·O"[j+1] for j in ((-1) ** (i)) * state.flatten()]
            for j in range(3):
                for k in range(3):
                    output[j][4*i+k] = state[3*j+k]
        if players is not None:
            print(players)
        for l in output:
            print(''.join(l))
        print()





