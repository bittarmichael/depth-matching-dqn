import os
import numpy as np
import torch as T
import argparse
from torch.utils.tensorboard import SummaryWriter
from env import Env
from model import DuelingDeepQNetwork


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, length,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-6,
                 replace=1000, chkpt_path='models/'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.length = length
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.chkpt_path = chkpt_path

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   length=self.length,
                                   chkpt_path=self.chkpt_path+'q_eval/')

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   length=self.length,
                                   chkpt_path=self.chkpt_path+'q_next/')

    def choose_action(self, observation, test=False):
        if np.random.random() > self.epsilon or test == True:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, i):
        self.q_eval.save_checkpoint(i)
        self.q_next.save_checkpoint(i)

    def load_models(self, i):
        self.q_eval.load_checkpoint(i)
        self.q_next.load_checkpoint(i)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)


        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)        

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss
    
    def run_test_episode(self, agent):
        score = 0
        done = None
        observation = self.reset()
        while not done:
            action = agent.choose_action(observation, test=True)
            observation_, reward, done, _ = self.step(action)
            score += reward
            observation = observation_

        return score
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", type=bool, default=False)
    parser.add_argument("-train_again", type=bool, default=False)
    args = parser.parse_args()

    test = args.test
    train_again = args.train_again
    
    window_size = 3000
    env = Env(window_size=window_size, test=test, no_rand=True)
    num_games = 50000
    
    chkpt_path = "weights/"
    
    if train_again:
        epsilon = 0.01
    else:
        epsilon = 1.0

    agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.0001,
                  input_dims=(2, window_size), length=window_size, n_actions=7, mem_size=100000, eps_min=0.01,
                  batch_size=32, eps_dec=5e-6, replace=1000, chkpt_path=chkpt_path)
    
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)
        os.makedirs(chkpt_path + '/q_eval')
        os.makedirs(chkpt_path + '/q_next')

    if test:
        agent.load_models(49950)
        
    if train_again == True and test == False:
        agent.load_models()
        
    scores = []
    eps_history = []
    n_steps = 0

    writer = SummaryWriter(f'runs/{chkpt_path}/')

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation, test)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not test:
                agent.store_transition(observation, action,
                                    reward, observation_, int(done))
                loss = agent.learn()

                if loss is not None:
                    writer.add_scalar('loss', loss.detach(), agent.memory.mem_cntr)

            observation = observation_
        
        if test:
            env.render()
        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])

        writer.add_scalar('avg_score', avg_score, i)
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon,
            env.tracker_position, env.match_point)
        
        if i > 0 and i % 50 == 0 and not test:
            test_env = Env(window_size=window_size, test=True, no_rand=True)
            test_score = test_env.run_test_episode(agent)
            writer.add_scalar('test_score', test_score, i)
            agent.save_models(i)

        eps_history.append(agent.epsilon)

