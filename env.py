import os
import numpy as np
import matplotlib.pyplot as plt
import random


class Env:
    def __init__(self, window_size=3000, test=False, no_rand=True):
        self.window_size = window_size
        self.action_space = [-100, -10, -1, 0, 1, 10, 100]
        self.test = test
        self.no_rand = no_rand
        self.reset()
        
    def reset(self):
        self.steps_in_env = 0
        reference, target, labels = self.get_random_well(self.test)
        self.reference = reference
        self.target = target
        self.labels = labels
        self.len = len(self.reference)
        
        self.starting_position = random.randint((self.len // 3), 2 * (self.len // 3))
        
        if self.no_rand:
            self.starting_position = 4500

        self.patch = self.target[self.starting_position -
                                 self.window_size // 2:self.starting_position +
                                 self.window_size // 2]
        
        self.match_point = int(self.labels[self.starting_position])
        self.tracker_position = self.starting_position

        return self.get_state()

    def get_state(self):
        copy_reference = self.reference.copy()
        curr_view = copy_reference[self.tracker_position -
                               self.window_size // 2:self.tracker_position +
                               self.window_size // 2]
        
        observation = [curr_view, self.patch]

        return observation

    def agent_in_bounds(self, tracker):
        if (tracker - self.window_size // 2) >= 0 and (tracker + self.window_size // 2) < len(self.reference):
            return True
        else:
            return False

    def is_terminal_state(self, action):
        if self.match_point == self.tracker_position and action == 3:
            return True, 20
        elif self.steps_in_env == 20:
            return True, 0
        else:
            return False, 0

    def step(self, action):
        self.steps_in_env += 1
        
        expected_position = self.tracker_position + self.action_space[
            action]
        if self.agent_in_bounds(expected_position):
            self.tracker_position = expected_position        
            
        next_state = self.get_state()
        reward = float((np.abs(self.match_point - self.tracker_position) * -1) / 100)
        done, add = self.is_terminal_state(action)
        return next_state, (reward + add), done, None

    def get_random_well(self, test=False):
        cwd = os.getcwd()
        well_path = os.path.join(cwd, "train")
        if test:
            well_path = os.path.join(cwd, "test")

        well_txt = random.choice(os.listdir(well_path))
        w = os.path.join(well_path, well_txt)

        well_data = np.loadtxt(w)
        ref_well, target_well, label = well_data
        return ref_well, target_well, label
    
    def run_test_episode(self, agent, render=False, ground_truth=True):
        score = 0
        done = None
        observation = self.reset()
        
        if not ground_truth:
            done = 0   
        count = 0       
        while not done and count < 20:
            action = agent.choose_action(observation, test=True)
            if not ground_truth:
                observation_, reward, _, _, = self.step(action)
                count += 1
                print(count, action)
            else:
                observation_, reward, done, _ = self.step(action)
            score += reward
            observation = observation_
            
        if render:
            self.render()
            
        magnitude = abs(self.starting_position - self.tracker_position)
        print(magnitude)

        return score

    def render(self):
        curr_state, match_state = self.get_state()
        curr_view = curr_state
        match_view = match_state
        
        figure = plt.figure()
           
        figure.add_subplot(411)
        plt.plot(self.reference)
        tracker = plt.Rectangle((self.tracker_position - 3000 // 2, min(self.reference)),
                               3000,
                               max(self.reference),
                                transform=plt.gca().transData)
        
        line, = plt.plot(self.reference,
                        color = 'green',
                        linewidth = 3,
                        alpha = 0.5,
                        label = "reference_well")
        
        line.set_clip_path(tracker)
        
        plt.axvline(x=self.match_point, c='k', linestyle='--')
        plt.axvline(x=self.tracker_position, c='r', linestyle='--')
          
        figure.add_subplot(412)
        plt.plot(self.target)
        tracker = plt.Rectangle((self.match_point - 3000 // 2, min(self.reference)),
                               3000,
                               max(self.reference),
                                transform=plt.gca().transData)
        
        line, = plt.plot(self.target,
                        color = 'red',
                        linewidth = 3,
                        alpha = 0.5,
                        label = "target_well")
        
        line.set_clip_path(tracker)
        
        figure.add_subplot(413)
        plt.plot(curr_view)
        plt.axvline(x=len(curr_view) // 2.4, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 4, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 2.4 + 500, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 12, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // (4/3), c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // (4/3) + 500, c='sienna', linestyle=':')

        figure.add_subplot(414)
        plt.plot(match_view)
        plt.axvline(x=len(curr_view) // 2.4, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 4, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 2.4 + 500, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // 12, c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // (4/3), c='sienna', linestyle=':')
        plt.axvline(x=len(curr_view) // (4/3) + 500, c='sienna', linestyle=':')
        plt.show()
        

if __name__ == "__main__":
    env = Env()
    action = 6
    state = env.get_state()
    next_state, reward, done, _ = env.step(action)
    breakpoint()

