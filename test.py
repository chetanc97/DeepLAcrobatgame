 
import numpy as np
import gym
import copy
from collections import deque
import random
import  pickle
 
bsize = 32
q_in_size = 4
q_hl1_size = 40
q_hl2_size = 40
q_out_size = 1
miu_in_size = 3
miu_hl1_size = 40
miu_hl2_size = 40
miu_out_size = 1
replay_size = 100*1000
episodes_num = 500
iters_num = 1000
gamma = 0.99
upd_r = 0.1
lr_actor = 3*1e-3
lr_critic = 1e-2
seed = 105
np.random.seed(seed)
random.seed(seed)
running_r = None
version = 1
demo = True
resume = demo
render = demo
allow_writing = not demo
 
print(bsize, replay_size, gamma, upd_r, lr_actor, lr_critic, seed, version, demo)
 
if resume:
    Q = pickle.load(open('Q-pendulum-%d' % version, 'rb'))
    Miu = pickle.load(open('Miu-pendulum-%d' % version, 'rb'))
else:
    Q = {}
    Q['W1'] = np.random.uniform(-1., 1., (q_in_size, q_hl1_size)) / np.sqrt(q_in_size)
    Q['W2'] = np.random.uniform(-1., 1., (q_hl1_size, q_hl2_size)) / np.sqrt(q_hl1_size)
    Q['W3'] = np.random.uniform(-3*1e-4, 3*1e-4, (q_hl2_size, q_out_size))
 
    Miu = {}
    Miu['W1'] = np.random.uniform(-1., 1., (miu_in_size, miu_hl1_size)) / np.sqrt(miu_in_size)
    Miu['W2'] = np.random.uniform(-1., 1., (miu_hl1_size, miu_hl2_size)) / np.sqrt(miu_hl1_size)
    Miu['W3'] = np.random.uniform(-3*1e-3, 3*1e-3, (miu_hl2_size, miu_out_size))
 
Q_tar = copy.deepcopy(Q)
Miu_tar = copy.deepcopy(Miu)
 
Qgrad = {}
Qgrad_sq = {}
for k, v in Q.items(): Qgrad[k] = np.zeros_like(v)
for k, v in Q.items(): Qgrad_sq[k] = np.zeros_like(v)
 
Miugrad = {}
Miugrad_sq = {}
for k, v in Miu.items(): Miugrad[k] = np.zeros_like(v)
for k, v in Miu.items(): Miugrad_sq[k] = np.zeros_like(v)
 
R = deque([], replay_size)
env = gym.make('Pendulum-v0')
 
def sample_batch(R, bsize):
    batch = random.sample(list(R), bsize)
    D_array = np.array(batch)
 
    states1 = np.array([data[0] for data in D_array])
    actions1 = np.array([data[1] for data in D_array])
    rewards = np.array([[data[2]] for data in D_array])
    states2 = np.array([data[3] for data in D_array])
    dones = np.array([data[4] for data in D_array])
 
    return states1, actions1, rewards, states2, dones
 
def relu(x):
    return np.maximum(0, x)
 
def tanh(x):
    e1 = np.exp(x)
    e2 = np.exp(-x)
    return (e1 - e2) / (e1 + e2)
 
def actions_Miu(states, Miu):
    hl1 = np.matmul(states, Miu['W1'])
    hl1 = relu(hl1)
    hl2 = np.matmul(hl1, Miu['W2'])
    hl2 = relu(hl2)
    outs = np.matmul(hl2, Miu['W3'])
    actions = 2 * tanh(outs)
 
    return actions
 
def values_Q(states, actions, Q):
    inputs = np.concatenate([states, actions], axis=1)
    hl1 = np.matmul(inputs, Q['W1'])
    hl1 = relu(hl1)
    hl2 = np.matmul(hl1, Q['W2'])
    hl2 = relu(hl2)
    values = np.matmul(hl2, Q['W3'])
 
    return values, hl2, hl1
 
def train_Q(douts, hl2, hl1, states, actions, Q):
    inputs = np.concatenate([states, actions], axis=1)
    dhl2 = np.matmul(douts, Q['W3'].transpose())
    dhl2[hl2 <= 0] = 0
    dhl1 = np.matmul(dhl2, Q['W2'].transpose())
    dhl1[hl1 <= 0] = 0
    d = {}
    d['W3'] = np.matmul(hl2.transpose(), douts)
    d['W2'] = np.matmul(hl1.transpose(), dhl2)
    d['W1'] = np.matmul(inputs.transpose(), dhl1)
 
    for k in Qgrad: Qgrad[k] = Qgrad[k] * 0.9 + d[k] * 0.1
    for k in Qgrad_sq: Qgrad_sq[k] = Qgrad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in Q: Q[k] -= lr_critic * Qgrad[k] / (np.sqrt(Qgrad_sq[k]) + 1e-5)
 
def train_Miu(states, Miu, Q):
    mhl1 = np.matmul(states, Miu['W1'])
    mhl1 = relu(mhl1)
    mhl2 = np.matmul(mhl1, Miu['W2'])
    mhl2 = relu(mhl2)
    outs = np.matmul(mhl2, Miu['W3'])
    actions = 2 * tanh(outs)
 
    inputs = np.concatenate([states, actions], axis=1)
    qhl1 = np.matmul(inputs, Q['W1'])
    qhl1 = relu(qhl1)
    qhl2 = np.matmul(qhl1, Q['W2'])
    qhl2 = relu(qhl2)
 
    dvalues = np.ones((bsize, q_out_size))
    dqhl2 = np.matmul(dvalues, Q['W3'].transpose())
    dqhl2[qhl2 <= 0] = 0
    dqhl1 = np.matmul(dqhl2, Q['W2'].transpose())
    dqhl1[qhl1 <= 0] = 0
    dinputs = np.matmul(dqhl1, Q['W1'].transpose())
    dactions = dinputs[:, 3:4]
    dactions /= bsize
 
    douts = dactions * 2 * (1 + actions/2) * (1 - actions/2)
    dmhl2 = np.matmul(douts, Miu['W3'].transpose())
    dmhl2[mhl2 <= 0] = 0
    dmhl1 = np.matmul(dmhl2, Miu['W2'].transpose())
    dmhl1[mhl1 <= 0] = 0
 
    d = {}
    d['W3'] = np.matmul(mhl2.transpose(), douts)
    d['W2'] = np.matmul(mhl1.transpose(), dmhl2)
    d['W1'] = np.matmul(states.transpose(), dmhl1)
 
    for k in Miugrad: Miugrad[k] = Miugrad[k] * 0.9 + d[k] * 0.1
    for k in Miugrad_sq: Miugrad_sq[k] = Miugrad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in Miu: Miu[k] += lr_actor * Miugrad[k] / (np.sqrt(Miugrad_sq[k]) + 1e-5)
 
def noise(episode):
    if demo:
        return 0.
    if np.random.randint(2) == 0:
        return (1. / (1. + episode/4))
    else:
        return -(1. / (1. + episode/4))
 
arr_values_rewards = []
 
for episode in range(1, episodes_num+1):
    state1 = env.reset()
    ep_reward = 0.
    value, _, _ = values_Q([state1], actions_Miu([state1], Miu), Q)
    for iter in range(1, iters_num+1):
        if render: env.render()
        action = actions_Miu(state1, Miu)
        action += noise(episode)
        state2, reward, done, _ = env.step(action)
        R.append([state1, action, reward, state2, done])
        ep_reward += reward
        state1 = state2
 
        if(len(R) > bsize) and not demo:
            states1, actions1, rewards, states2, dones = sample_batch(R, bsize)
            actions2 = actions_Miu(states2, Miu_tar)
            values, _, _ = values_Q(states2, actions2, Q_tar)
            second_term = gamma * values
            second_term[dones] = 0
            y = rewards + second_term
 
            outs, hl2, hl1 = values_Q(states1, actions1, Q)
            douts = (outs - y) / bsize
            train_Q(douts, hl2, hl1, states1, actions1, Q)
            train_Miu(states1, Miu, Q)
 
            for k, v in Q.items(): Q_tar[k] = upd_r * v + (1-upd_r) * Q_tar[k]
            for k, v in Miu.items(): Miu_tar[k] = upd_r * v + (1-upd_r) * Miu_tar[k]
 
        if done or iter == iters_num:
            running_r = (running_r * 0.9 + ep_reward * 0.1) if running_r != None else ep_reward
            arr_values_rewards.append([value, ep_reward])
            if episode % 1 == 0:
                print(np.mean(Q['W1']), np.mean(Q['W2']), np.mean(Q['W3']))
                print(np.mean(Miu['W1']), np.mean(Miu['W2']), np.mean(Miu['W3']))
                print('ep: %d, iters: %d, reward %f, run aver: %f' % \
                      (episode, iter, ep_reward, running_r))
            if episode % 10 == 0 and allow_writing:
                pickle.dump(Q, open('Q-pendulum-%d' % version, 'wb'))
                pickle.dump(Miu, open('Miu-pendulum-%d' % version, 'wb'))
                pickle.dump(arr_values_rewards, open('VR-pendulum-%d' % version, 'wb'))
            break