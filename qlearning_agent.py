# qlearningAgents.py
# Q-Learning implementation for Pacman with Learning Rate Decay

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory
from util import manhattanDistance

import backend
import gridworld
import random
import util
import math
import numpy as np
import copy
import pickle
import os

###############################################################################
# HYPERPARAMETER AND REWARD CONFIGURATION
###############################################################################

HYPERPARAMETERS = {
    'config_1': {
        'learning_rate': 0.3,
        'learning_rate_min': 0.01,
        'learning_rate_decay': 0.9995,
        'discount_factor': 0.92,
        'epsilon_start': 1.0,
        'epsilon_end': 0.02,
        'epsilon_decay': 0.97
    },
    'config_2': {
        'learning_rate': 0.4,
        'learning_rate_min': 0.01,
        'learning_rate_decay': 0.999,
        'discount_factor': 0.88,
        'epsilon_start': 0.9,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.9995
    },
    'config_3': {
        'learning_rate': 0.25,
        'learning_rate_min': 0.005,
        'learning_rate_decay': 0.9998,
        'discount_factor': 0.98,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.9998
    }
}

REWARD_SYSTEMS = {
    'model_1': {
        'step': -1,
        'eat_food': +10,
        'eat_power_pellet': +20,
        'eat_ghost': +40,
        'win': +100,
        'lost': -100
    },
    'model_2': {
        'step': 0,
        'eat_food': +5,
        'eat_power_pellet': +30,
        'eat_ghost': +50,
        'win': +200,
        'lost': -100
    },
    'model_3': {
        'step': -5,
        'eat_food': +15,
        'eat_power_pellet': +50,
        'eat_ghost': +150,
        'win': +250,
        'lost': -100
    }
}

###############################################################################
# Q-LEARNING AGENT WITH LEARNING RATE DECAY
###############################################################################

class QLearningAgent(ReinforcementAgent):
    
    def __init__(self, 
                 reward_model='model_1',
                 hyperparameter_config='config_1',
                 **args):
        
        ReinforcementAgent.__init__(self, **args)
        
        self.reward_model = REWARD_SYSTEMS[reward_model]
        self.reward_model_name = reward_model
        
        config = HYPERPARAMETERS[hyperparameter_config]
        
        self.alpha_initial = config['learning_rate']
        self.alpha = config['learning_rate']
        self.alpha_min = config['learning_rate_min']
        self.alpha_decay = config['learning_rate_decay']
        
        self.discount = config['discount_factor']
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        
        self.q_table = util.Counter()
        
        self.last_state = None
        self.last_action = None
        self.episode_rewards = []
        self.q_table_file = f"q_table_{reward_model}_{hyperparameter_config}.pkl"
        
        print(f"\n{'='*70}")
        print(f"Q-Learning Agent Initialized with Learning Rate Decay")
        print(f"{'='*70}")
        print(f"Reward Model: {reward_model}")
        print(f"Hyperparameters: {hyperparameter_config}")
        print(f"  - Learning Rate: {self.alpha_initial:.3f} -> {self.alpha_min:.3f}")
        print(f"  - Alpha Decay: {self.alpha_decay}")
        print(f"  - Discount Factor: {self.discount}")
        print(f"  - Epsilon Start: {self.epsilon_start}")
        print(f"  - Epsilon End: {self.epsilon_end}")
        print(f"  - Epsilon Decay: {self.epsilon_decay}")
        print(f"Q-table file: {self.q_table_file}")
        print(f"{'='*70}\n")
        
        self.loadQTable()

    def getStateRepresentation(self, state):
        
        pacman_pos = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        ghost_states = state.getGhostStates()
        food = state.getFood().asList()
        walls = state.getWalls()
        capsules = state.getCapsules()
        
        # 1. Ghost proximity (0-4)
        min_ghost_dist = min([manhattanDistance(pacman_pos, g) for g in ghosts])
        if min_ghost_dist <= 1:
            ghost_prox = 0  # IMMEDIATE DANGER
        elif min_ghost_dist <= 3:
            ghost_prox = 1  # VERY CLOSE
        elif min_ghost_dist <= 6:
            ghost_prox = 2  # CLOSE
        elif min_ghost_dist <= 10:
            ghost_prox = 3  # MEDIUM
        else:
            ghost_prox = 4  # SAFE
        
        # 2. Ghost scared state (0-1) 
        closest_ghost_idx = ghosts.index(min(ghosts, key=lambda g: manhattanDistance(pacman_pos, g)))
        ghost_scared = 1 if ghost_states[closest_ghost_idx].scaredTimer > 0 else 0
        
        # 3. Ghost direction (0-4)
        if min_ghost_dist <= 8:
            closest_ghost = min(ghosts, key=lambda g: manhattanDistance(pacman_pos, g))
            dx = closest_ghost[0] - pacman_pos[0]
            dy = closest_ghost[1] - pacman_pos[1]
            if abs(dy) > abs(dx):
                ghost_dir = 0 if dy > 0 else 1
            else:
                ghost_dir = 2 if dx > 0 else 3
        else:
            ghost_dir = 4
        
        # 4. Food direction (0-4)
        if food:
            closest_food = min(food, key=lambda f: manhattanDistance(pacman_pos, f))
            dx = closest_food[0] - pacman_pos[0]
            dy = closest_food[1] - pacman_pos[1]
            if abs(dy) > abs(dx):
                food_dir = 0 if dy > 0 else 1
            else:
                food_dir = 2 if dx > 0 else 3
        else:
            food_dir = 4
        
        # 5. Food distance (0-3)
        if food:
            food_dist = manhattanDistance(pacman_pos, closest_food)
            if food_dist <= 2:
                food_prox = 0  # VERY CLOSE
            elif food_dist <= 5:
                food_prox = 1  # CLOSE
            elif food_dist <= 10:
                food_prox = 2  # MEDIUM
            else:
                food_prox = 3  # FAR
        else:
            food_prox = 3
        
        # 6. Capsule nearby (0-1)
        capsule_nearby = 0
        if capsules:
            min_capsule_dist = min([manhattanDistance(pacman_pos, c) for c in capsules])
            capsule_nearby = 1 if min_capsule_dist <= 5 else 0
        
        # 7. Number of surrounding walls (0-2) - NEW
        x, y = int(pacman_pos[0]), int(pacman_pos[1])
        wall_count = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            if walls[x+dx][y+dy]:
                wall_count += 1
        
        if wall_count <= 1:
            surrounding_walls = 0  # Open space
        elif wall_count == 2:
            surrounding_walls = 1  # Corridor
        else:
            surrounding_walls = 2  # Corner/tight spot
        
        # STATE: 5 × 2 × 5 × 5 × 4 × 2 × 3 = 6000 states
        return (ghost_prox, ghost_scared, ghost_dir, food_dir, food_prox, capsule_nearby, surrounding_walls)

    def getQValue(self, state, action):
        state_rep = self.getStateRepresentation(state)
        return self.q_table[(state_rep, action)]

    def computeValueFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return 0.0
        max_q_value = max([self.getQValue(state, action) for action in legal_actions])
        return max_q_value

    def computeActionFromQValues(self, state):
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        max_q_value = self.computeValueFromQValues(state)
        best_actions = [action for action in legal_actions 
                       if self.getQValue(state, action) == max_q_value]
        return random.choice(best_actions)

    def getAction(self, state):
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        
        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.computeActionFromQValues(state)
        
        self.last_state = state
        self.last_action = action
        
        return action

    def update(self, state, action, nextState, reward):
        state_rep = self.getStateRepresentation(state)
        
        current_q = self.q_table[(state_rep, action)]
        next_max_q = self.computeValueFromQValues(nextState)
        
        td_target = reward + self.discount * next_max_q
        td_error = td_target - current_q
        
        new_q = current_q + self.alpha * td_error
        
        self.q_table[(state_rep, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def final(self, state):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        ReinforcementAgent.final(self, state)
        
        if self.episodesSoFar % 100 == 0:
            print(f"Episode {self.episodesSoFar}: "
                  f"epsilon = {self.epsilon:.3f}, "
                  f"alpha = {self.alpha:.3f}, "
                  f"Q-table size = {len(self.q_table)} entries, "
                  f"Score = {state.getScore()}")
        
        if self.episodesSoFar % 500 == 0:
            self.saveQTable()
        
        if self.episodesSoFar == self.numTraining:
            self.saveQTable()
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"Total episodes: {self.episodesSoFar}")
            print(f"Q-table size: {len(self.q_table)} entries")
            print(f"Final epsilon: {self.epsilon:.4f}")
            print(f"Final alpha: {self.alpha:.4f}")
            print(f"Q-table saved to: {self.q_table_file}")
            print(f"{'='*70}\n")
    
    def saveQTable(self):
        try:
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            return True
        except Exception as e:
            print(f"Error saving Q-table: {e}")
            return False
    
    def loadQTable(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_table = pickle.load(f)
                    self.q_table = util.Counter(loaded_table)
                print(f"Q-table loaded from {self.q_table_file}")
                print(f"  Loaded {len(self.q_table)} Q-values")
                return True
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                return False
        else:
            print(f"No existing Q-table found. Starting fresh.")
            return False


###############################################################################
# PACMAN AGENT CLASS
###############################################################################

class SmartPacmanAgent(QLearningAgent):

    def __init__(self, 
                 epsilon=0.05,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=0,
                 reward_model='model_1',
                 hyperparameter_config='config_1',
                 **args):
        
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0
        
        QLearningAgent.__init__(
            self, 
            reward_model=reward_model,
            hyperparameter_config=hyperparameter_config,
            **args
        )

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action
    
    def final(self, state):
        QLearningAgent.final(self, state)