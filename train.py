import torch
from torch import nn, optim
import numpy as np

from adversaries.trained_adversary.pi import Pi

from tic_tac_environment import TicTacEnv 

def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    gamma = 0.99
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main(episodes=10000,learning_rate=0.005):
    env = TicTacEnv()

    # in_dim is the state dimension
    in_dim = env.observation_space.shape[0]

    # out_dim is the action dimension, we have max 9 possible moves
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    pi.train()
    
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)
    games = [] # this will store how a game in the episode `epi` ended, win, lose, or draw
    # episodes=10
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = pi.act(state)
            state, reward, done, game_status_tag = env.step(action)
            pi.rewards.append(reward)
            if done:
                games.append(game_status_tag) #
                break

        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)

        pi.onpolicy_reset()

        if len(games) > 500:
            games = games[-500:]
        if episode % 1000 == 0:
            DRAW='='
            WIN='+'
            LOSE='-'
            pretty = lambda x : ''.join({'W':WIN,'L':LOSE,'D':DRAW}[i] for i in x)
            last_500 = pretty(''.join(games[-500:]))
            last_100 = pretty(''.join(games[-100:]))
            print('\n', last_100, '\n')
            for batch in [last_100, last_500]:
                print(
                    f'LAST {len(batch)} GAMES:',
                    'WINS',
                    str(batch.count(WIN)).ljust(3),
                    'DRAW',
                    str(batch.count(DRAW)).ljust(3),
                    'LOST',
                    str(batch.count(LOSE)).ljust(3),
                    )
            print('\n', f'Episode {episode}, loss: {loss}, total reward: {total_reward}')
    return pi

if __name__ == '__main__':
    model = main(episodes=1000000, learning_rate=0.005)
    torch.save(model.state_dict(), './adversaries/trained_adversary/models/v1.pt')