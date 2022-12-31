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

def main(episodes=100000):
    env = TicTacEnv()

    # in_dim is the state dimension
    in_dim = env.observation_space.shape[0]

    # out_dim is the action dimension, we have max 9 possible moves
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    pi.train()
    
    optimizer = optim.Adam(pi.parameters(), lr=0.0001)
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

        if episode % 1000 == 0:
            last_500 = ''.join(games[-500:])
            last_100 = ''.join(games[-100:])
            print('\n',''.join(games[-100:]))
            print(
                'WIN_100',
                last_100.count('W'),
                'DRAW_100',
                last_100.count('D'),
                'LOST_100',
                last_100.count('L'),
                'INVAL_100',
                last_100.count('-'),
                )
            print(
                'WIN_500',
                last_500.count('W'),
                'DRAW_500',
                last_500.count('D'),
                'LOST_500',
                last_500.count('L'),
                'INVAL_100',
                last_500.count('-'),
                )
            print(f'Episode {episode}, loss: {loss}, total reward: {total_reward}')
    return pi

if __name__ == '__main__':
    model = main()
    torch.save(model.state_dict(), './adversaries/trained_adversary/models/v1.pt')