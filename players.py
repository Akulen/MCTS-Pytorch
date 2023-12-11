import numpy as np
import os
import torch

from gameEngine import GameEngine, TicTacToeMoves
Ms = TicTacToeMoves

class Player:
    def __init__(self):
        raise NotImplementedError

    def random_move(self, game, gen: np.random.Generator):
        del game, gen
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class RandomPlayer(Player):
    def __init__(self, gameEngine: GameEngine, id=''):
        self.ge = gameEngine
        self.id = id

    def random_move(self, game, gen: np.random.Generator):
        moves = self.ge.legalMoves(game)
        return moves[gen.choice(len(moves))]

    def name(self):
        return f'Random{self.id}'

class NNPlayer(Player):
    def __init__(self, nnet, mcts, id=''):
        self.nnet = nnet
        self.mcts = mcts
        self.ge = mcts.ge
        self.id = id

    def gen_data(self, n_games=25_000, max_len=-1, progress=None, device=None):
        return self.mcts.gen_data(self.nnet, n_games=n_games, max_len=max_len, progress=progress, device=device)

    def random_move(self, game, gen: np.random.Generator):
        pi = self.mcts.policy(game, self.nnet, gen)

        moves = self.ge.legalMoves(game)
        p_pi = [pi[move] for move in moves]

        return moves[np.argmax(p_pi)]

    def best_move(self, game):
        pi = self.mcts.policy(game, self.nnet, None)

        moves = self.ge.legalMoves(game)
        p_pi = [pi[move] for move in moves]

        return moves[np.argmax(p_pi)]
        # return moves[gen.choice(len(moves), p=p_pi)]

    def name(self):
        return f"NN{self.id}"

def BestPlayer(mcts, config, gameEngine: GameEngine, id='') -> Player:
    id = f'Best{id}'
    if os.path.isfile(f'./models/{config["game"]}best.pt'):
        nnet = config['model']()
        nnet.load_state_dict(torch.load(f'./models/{config["game"]}best.pt'))
        return NNPlayer(nnet, mcts, id=id)

    return RandomPlayer(gameEngine, id=id)
