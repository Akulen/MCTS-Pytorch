import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle as pkl
import rich.progress as rp
import torch
from typing import Any

from gameEngine import GameEngine, TicTacToe
from models import BaseNN
from utils import get_freer_gpu
from players import Player, NNPlayer, RandomPlayer, BestPlayer

if False:
    config = {
        'game': 'ttt', # ttt
        'n_sim': 100, # 1600
        'n_gen_games': 90, # 1000
        'n_train_iter': 100, # 1000
        'batch_size': 1024,
        'n_games_eval': 45, # 100
        'n_iter': 10,
        'n_jobs': 90,
    }
else:
    config = {
        'game': 'ttt', # ttt
        'n_sim': 4,
        'n_gen_games': 51, # 1000
        'n_train_iter': 100, # 1000
        'batch_size': 1024,
        'n_games_eval': 100,
        'n_iter': 50,
        'n_jobs': 50,
    }

class MCTS:
    def __init__(self, gameEngine: GameEngine, c=1.0, n_sim=config['n_sim'], single_thread=False, device=None):
        super().__init__()

        self.ge = gameEngine
        self.device = "cpu" if device is None else device

        self.c = c
        self.n_sim = n_sim
        self.alpha = 0.03
        self.eps_exploration = 0.25
        self.num_sampling_moves = 30

        self.single_thread = single_thread
        if self.single_thread:
            self.model = dict()
        else:
            manager = multiprocessing.Manager()
            self.model = manager.dict()

    def puct_score(self, sqrt_N, n, p, q): # TODO: update c = log((1 + N + c_1) / c_1) + c_0
        return q + self.c * p * sqrt_N / (1 + n)

    def bestMove(self, moves, N, P, Q):
        max_u, best_a_i = -float("inf"), np.random.choice(len(moves))
        sqrt_N = np.sqrt(1+np.sum(N))
        s = []
        for a_i, (n, p, q) in enumerate(zip(N, P, Q)):
            u = self.puct_score(sqrt_N, n, p, q)
            s.append(u)
            if u > max_u:
                max_u = u
                best_a_i = a_i
        return best_a_i, moves[best_a_i]

    def expand(self, nnet, N, P, Q, s, hs=None, zero=False):
        if hs is None:
            hs = self.ge.hash(s)
        if zero or hs not in self.model:
            self.model[hs] = nnet.predict(self.ge.encodeState(s, device=self.device))
        model_v, model_P = self.model[hs]

        moves = self.ge.legalMoves(s)
        P[hs] = []
        for move in moves:
            i = -1
            if config['game'] == 'ttt':
                for j in range(len(self.ge.allMoves())):
                    if self.ge.allMoves()[j] == move:
                        i = j
                        break
            else:
                raise NotImplementedError
            assert i != -1
            P[hs].append(model_P[i])
        Q[hs] = [0.] * len(moves)
        N[hs] = [0] * len(moves)

        return model_v

    def search(self, s, nnet, N, P, Q):
        actions = []
        hs = self.ge.hash(s)

        while not self.ge.gameOver(s) and hs in N:
            moves = self.ge.legalMoves(s)
            move_i, move = self.bestMove(moves, N[hs], P[hs], Q[hs])

            actions.append((hs, move_i))
            s = self.ge.makeMove(s, move)
            hs = self.ge.hash(s)

        if self.ge.gameOver(s):
            v = self.ge.outcome(s)
        else:
            v = self.expand(nnet, N, P, Q, s, hs=hs, zero=True)

        for hs, move_i in actions[::-1]:
            v = -v
            Q[hs][move_i] = (Q[hs][move_i] * N[hs][move_i] + v) / (N[hs][move_i] + 1)
            N[hs][move_i] += 1

        return v

    def policy(self, s, nnet, gen: np.random.Generator | None, progress=None):
        hs = self.ge.hash(s)
        N = {}
        P = {} # P[s][a] is the probability evaluated by nnet of playing a in state s
        Q = {} # Q[s][a] is the average outcome of playing a in state s

        if gen is None:
            dir_noise = np.ones(len(self.ge.legalMoves(s)))
        else:
            dir_noise = gen.dirichlet([self.alpha] * len(self.ge.legalMoves(s)))

        self.expand(nnet, N, P, Q, s, hs=hs)
        for i, noise in enumerate(dir_noise):
            P[hs][i] = P[hs][i] * (1 - self.eps_exploration) + noise * self.eps_exploration

        task = None
        if progress is not None:
            task = progress[0].add_task(f"[cyan]{progress[1]}: [magenta]Generating Policy", total=self.n_sim)
        for _ in range(self.n_sim):
            self.search(self.ge.copy(s), nnet, N, P, Q)
            if progress is not None:
                progress[0].update(task, advance=1)

        total = np.sum(N[hs])
        moves = self.ge.legalMoves(s)
        return {
            a: N[hs][a_i] / total
            for a_i, a in enumerate(moves)
        }

    def gen_data(self, nnet, n_games=25_000, max_len=-1, progress=None, device=None):
        task = None
        if progress is not None:
            task = progress[0].add_task(f"[cyan]{progress[1]}: [green]Generating Data", total=n_games, arg1_n="#positions", arg1=0)

        def gen_game(gen: np.random.Generator):
            examples_per_game = []
            game = self.ge.startingPosition()
            while not self.ge.gameOver(game) and (max_len == -1 or len(examples_per_game) < max_len):
                pi = self.policy(game, nnet, gen)
                examples_per_game.append([game.copy(), pi])

                moves = self.ge.legalMoves(game)

                p_pi = [pi[move] for move in moves]
                if len(examples_per_game) < self.num_sampling_moves:
                    move = moves[gen.choice(len(moves), p=p_pi)]
                else:
                    move = moves[np.argmax(p_pi)]
                game = self.ge.makeMove(game, move)
            result = self.ge.outcome(game) * (-1) ** len(examples_per_game)
            for example in examples_per_game:
                example.append(result)
                result = -result
            return examples_per_game

        examples: Any = Parallel(
            n_jobs=config['n_jobs'],
            batch_size=1, # type: ignore
            return_as='generator'
        )(
            delayed(gen_game)(np.random.default_rng(np.random.randint(int(1e10))))
            for _ in range(n_games)
        )
        data = []
        for gameData in examples:
            data += [
                self.ge.encodeStateAndOutput(state, policy, evaluation, device=self.device if device is None else device)
                for state, policy, evaluation in gameData
            ]
            if progress is not None:
                progress[0].update(task, advance=1, arg1=len(data))

        return data

# Runs a game between model1 and model2 and returns the result
def play_game(game, player1: Player, player2: Player, gameEngine: GameEngine, gen: np.random.Generator):
    mult = 1
    while not gameEngine.gameOver(game):
        move = player1.random_move(game, gen)
        game = gameEngine.makeMove(game, move)
        player1, player2 = player2, player1
        mult = -mult

    return mult * gameEngine.outcome(game), game

def results_to_score(results):
    if isinstance(results, list) and len(results) == 3:
        return f"{(results[0]+results[1]/2) / sum(results):.5f}"
    return "N/A"

def pretty_results(results):
    return f'{results[0]}/{results[1]}/{results[2]} {results_to_score(results)}'

# Runs n games between player1 and player2 and returns the results
def pit(player1: Player, opponents: dict[str, Player], gameEngine: GameEngine, n_games=400, progress=None, n_display=5, step=-1):
    tasks = {}
    if progress is not None:
        tasks = {
            task_name: progress[0].add_task(
                f"[cyan]{progress[1]}:[/ cyan] Playing vs {task_name}",
                total=n_games,
                arg1_n="results",
                arg1="0/0/0 N/A"
            )
            for task_name in opponents.keys()
        }

    def playgames(i, task_name: str, player2: Player, gen):
        if i % 2 == 0:
            x, game = play_game(gameEngine.startingPosition(), player2, player1, gameEngine, gen)
            x = -x
            game = (False, game)
        else:
            x, game = play_game(gameEngine.startingPosition(), player1, player2, gameEngine, gen)
            game = (True, game)
        return task_name, player2.name(), x, game

    rresults: Any = Parallel(
        n_jobs=config['n_jobs'],
        batch_size=1, # type: ignore
        return_as='generator'
    )(
        delayed(playgames)(i, task_name, player2, np.random.default_rng(np.random.randint(int(1e10))))
        for i in range(n_games)
        for task_name, player2 in opponents.items()
    )
    results = {
        task_name: [0, 0, 0]
        for task_name in opponents.keys()
    }
    displayed = {
        task_name: [0, 0]
        for task_name in opponents.keys()
    }
    for task_name, player2_name, x, (normalOrder, game) in rresults:
        if displayed[task_name][normalOrder] < n_display and x == -1:
            players = f'{player1.name()}/{player2_name}' if normalOrder else f'{player2_name}/{player1.name()}'
            gameEngine.print(game, ("MCTS Training", step), players, reversed=not normalOrder)
            displayed[task_name][normalOrder] += 1
        results[task_name][1-x] += 1
        if progress is not None:
            progress[0].update(tasks[task_name], advance=1, arg1=pretty_results(results[task_name]))
    return results

def finalnet(gameEngine, iterations=200, n_games_eval=400, device=[None, None]):
    nnet = config['model']().to(device[1])
    oldPlayer = NNPlayer(nnet, MCTS(gameEngine, device=device[1]), id="init")
    randomPlayer = RandomPlayer(gameEngine)
    bestPlayer = BestPlayer(MCTS(gameEngine, device=device[1]), config, gameEngine)

    dpi = 96
    steps = []
    plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    with rp.Progress(
        *rp.Progress.get_default_columns(),
        rp.TimeElapsedColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("{task.fields[arg1_n]}: {task.fields[arg1]}"),
        refresh_per_second=1
    ) as progress:
        task = progress.add_task("[cyan]Stepping", total=iterations, arg1_n="perf", arg1="N/A")
        for i in range(iterations):
            data = oldPlayer.gen_data(n_games=config['n_gen_games'], max_len=512, progress=(progress, i), device=device[0])

            with open(f'data/games/{config["game"]}-it{i:03}-{nnet.id()}.out', 'wb') as f:
                pkl.dump({
                    'data': data
                }, f)

            # new_nnet = BaseNN().to(device)
            new_nnet = copy.deepcopy(nnet).to(device[0])

            losses = new_nnet.fit(
                data,
                n_iter=config['n_train_iter'],
                batch_size=config['batch_size'],
                progress=(progress, i),
            )

            newPlayer = NNPlayer(new_nnet.to(device[1]), MCTS(gameEngine, device=device[1]), id=f'Step{i:03}')

            results = pit(
                newPlayer,
                {
                    'random': randomPlayer,
                    'self-play': oldPlayer,
                    'best': bestPlayer,
                },
                gameEngine,
                n_games=n_games_eval,
                progress=(progress, i),
                step=i+1
            )
            print(results)

            nnet = new_nnet
            oldPlayer = newPlayer

            if i % 5 == 0:
                plt.plot([loss.sum() for loss in losses], label=f'Iteration {i}')
            steps.append((results, losses[-1]))
            progress.update(task, advance=1, arg1=f'{results_to_score(results["random"])}(random) {results_to_score(results["self-play"])}(self) {results_to_score(results["best"])}(best)')

    plt.legend()
    plt.savefig(f'plots/loss/{config["game"]}-{nnet.id()}.svg')

    fig, ax1 = plt.subplots()
    fig.set_size_inches((1920/dpi, 1080/dpi))
    fig.set_dpi(dpi)
    ax1.set_xlabel('step')

    color = 'tab:red'
    ax1.set_ylabel('score', color=color)
    results = {
        tp: [step[tp] for step, _ in steps]
        for tp in steps[0][0]
    }
    ls = {
        'random': '--',
        'self-play': ':',
        'best': '-'
    }
    ax1.plot([0.5] * len(results['best']), ':', color='grey')
    for tp, scores in results.items():
        ax1.plot([float(results_to_score(result)) for result in scores], ls[tp], color=color, label=tp)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0., 1.])
    plt.legend()

    ax2 = ax1.twinx()  # instantiate a second ax that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    ax2.plot([l1 for _, (l1, _) in steps], ':', color=color, label='mse')
    ax2.plot([l2 for _, (_, l2) in steps], '--', color=color, label='cross entropy')
    ax2.plot([l1+l2 for _, (l1, l2) in steps], color=color, label='loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    plt.savefig(f'plots/steps/{config["game"]}-{nnet.id()}.svg')

    return nnet

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device, device2 = "cpu", "cpu"
    if torch.cuda.is_available():
        a, b = get_freer_gpu(2)
        device = f"cuda:{a}" # Training Device
        # device2 = f"cuda:{b}" # MCTS Device

    if config['game'] == 'ttt':
        gameEngine = TicTacToe()
        config['model'] = BaseNN
    else:
        raise NotImplementedError
    nnet = finalnet(gameEngine, iterations=config['n_iter'], n_games_eval=config['n_games_eval'], device=[device, device2])
    torch.save(nnet.state_dict(), f'./models/{config["game"]}best.pt')
