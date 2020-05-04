#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import random
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

ACTION_IDX = {
    'left' : 0,
    'right' : 1,
    'forward' : 2,
    'pickup' : 3,
    'drop' : 4,
    'toggle' : 5,
    'enter' : 6
}

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    # new code
    o = obs["image"]
    o = o.transpose()
    # print(o)
    o = preprocess(o)

    input_state = rdf(o)

    print("***********************input state*********************************** ")
    print(input_state)
    leaf_state = []
    root.reward = reward
    state_node = root.traverse(input_state, leaf_state)
    print("Action Taken: ", state_node.assertAction)
    # if action is None:
    #     action = random.randint(0, len(ACTION_TO_IDX) - 1)

    action = state_node.assertAction

    while not done:
        # key_handler.key = action
        redraw(obs)

        obs, reward, done, info = env.step(ACTION_IDX[action])
        print('step=%s, reward=%.2f' % (env.step_count, reward))

        o = obs["image"]
        o = o.transpose()
        # print(o)
        o = preprocess(o)
        input_state = rdf(o)
        print(input_state)
        leaf_state = []
        root.reward = reward
        state_node = root.traverse(input_state, leaf_state)
        print("Action Taken: ", state_node.assertAction)
        action = state_node.assertAction
        # if action is None:
        #     action = random.randint(0, len(ACTION_TO_IDX)-1)

        if done:
            print("done")
            break;

    # old code

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def step_next(action):

    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    # new code
    o = obs["image"]
    o = o.transpose()
    print(o)

    o = preprocess(o)

    input_state = rdf(o)

    print("***********************input state*********************************** ")
    print(input_state)
    leaf_state = []
    root.reward = reward
    root.traverse(input_state, leaf_state)

# Map of object type to integers old
OBJECT_TO_IDX_OLD = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10
}

# Map of object type to integers new
OBJECT_TO_IDX_NEW = {
    'agent': 0,
    'key': 1,
    'door': 2,
    'goal': 3
}

def preprocess(old_state):
    dim = len(OBJECT_TO_IDX_NEW)
    old_key_list = list(OBJECT_TO_IDX_OLD.keys())
    old_val_list = list(OBJECT_TO_IDX_OLD.values())
    obs = (3, dim, dim)
    obs = np.zeros(obs, dtype=int)

    visible = (dim, dim)
    visible = np.zeros(visible, dtype=int)

    carrying = (dim, dim)
    carrying = np.zeros(carrying, dtype=int)

    locked = (dim, dim)
    locked = np.zeros(locked, dtype=int)

    for key in OBJECT_TO_IDX_NEW:
        old_index = OBJECT_TO_IDX_OLD[key]
        new_index = OBJECT_TO_IDX_NEW[key]
        found = np.where(old_state[0] == old_index)
        print("agent ", old_state[0][6][3])
        print("key ", old_index)

        if (found[0].size > 0):
            visible[0][new_index] = 1

    obs[0] = visible

    if (old_state[0][6][3] != 1):
        carrying_object = old_key_list[old_val_list.index(old_state[0][6][3])]
        new_index = OBJECT_TO_IDX_NEW[carrying_object]
        carrying[0][new_index] = 1

    obs[1] = carrying

    is_door_locked = np.where(old_state[2] == 2)
    door_index = OBJECT_TO_IDX_NEW['door']
    if (is_door_locked[0].size > 0):
        locked[0][door_index] = 1

    obs[2] = locked

    print(obs)

    return obs


def rdf(o):
    state = []

    visible = o[0]
    carrying = o[1]
    locked = o[2]

    objects_visible = np.where(visible[0] == 1)
    objects_carrying = np.where(carrying[0] == 1)
    door_locked = np.where(locked[0] == 1)

    key_list = list(OBJECT_TO_IDX_NEW.keys())
    val_list = list(OBJECT_TO_IDX_NEW.values())

    print(locked)

    for b in objects_visible[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "visible", object))

    for b in objects_carrying[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "carrying", object))

    for b in door_locked[0]:
        object = key_list[val_list[b]]
        state.append(("agent", "locked", object))

    for s, p, o in state:
        print((s, p, o))
    return state


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

ACTION_TO_IDX = {
    0 : 'left',
    1 : 'right',
    2 : 'forward',
    3 : 'pickup',
    4 : 'drop',
    5 : 'toggle',
    6 : 'enter'
}


class Tree:
    def __init__(self):
        self.__root = None


class TreeNode:
    def __init__(self, predicate, obj, n=0, assertAction=""):

        self.nodeType = n  ##  test, 1 -> #leaf
        self.parent = None
        self.yes = None
        self.no = None
        self.reward = 0
        self.learning_rate = 0.1
        self.discount_fact = 0.1
        self.last_state = []
        self.assertactn = " "

        # for test nodes
        if self.nodeType == 0:
            self.predicate = predicate
            self.obj = obj

        # for leaf nodes
        else:
            self.expression = []
            self.assertAction = assertAction
            self.Q_val = 0
            self.Q_val_list = list(0 for i in range(len(ACTION_TO_IDX)))

    def insert(self, side, val, assertAction=" "):
        # Compare the new value with the parent node
        if len(val) != 0:
            if side == "yes":
                self.yes = TreeNode(val[0], val[1])
                self.yes.parent = self
            else:
                self.no = TreeNode(val[0], val[1])
                self.no.parent = self
        else:
            if side == "yes":
                self.yes = TreeNode("", "", 1, assertAction)
                self.yes.parent = self
            else:
                self.no = TreeNode("", "", 1, assertAction)
                self.yes.parent = self

    def print(self):
        if self.nodeType == 0:
            print("Test node ", self.predicate, self.obj)
        else:
            print("Leaf Node ", self.assertAction, self.expression, self.Q_val)
        if self.yes:
            self.yes.print()
        if self.no:
            self.no.print()

    def get_action(self):
        if self.assertAction == " ":
            # select random action
            action = random.randint(0, len(ACTION_TO_IDX)-1)
            self.assertAction = ACTION_TO_IDX[action]
            # self.assertactn = self.assertAction
        else:
            # do an q-update
            # 1. get max of Q_val from possible states with each action
            # let's default that with some value now
            max_val = 0.1
            # TODO --> calculate the 'estimate of optimal future value'
            '''QUESTION  (TODO) ---> is the understanding correct for the calculation?
            Are we going to do a test here to take which predicate to split and create 'yes'
            and 'no' child to get their Q-val and see if there are any changes in the Q-val-vec of child.
            If parent/child Q_val-vec remains same, we don't split, 
            otherwise we split the node, and get the MAX Q-val for future val from the child nodes.
            '''

            for i in range(len(self.Q_val_list)):
                self.Q_val_list[i] = self.Q_val_list[i] + self.learning_rate * (
                    self.reward + self.discount_fact * max_val - self.Q_val_list[i]
                )
            action = self.Q_val_list.index(max(self.Q_val_list))
            self.Q_val = self.Q_val_list[action]
            self.assertAction = ACTION_TO_IDX[action]
            # self.assertactn = self.assertAction


    def traverse(self, predList, state_exp):
        if self.nodeType == 1:
            self.expression = state_exp
            print("LEAF NODE FOUND, @ State ", self.expression)
            self.get_action()
            print("Best Q-Value State Action Pair: ", self.Q_val, self.assertAction)
            return self
        predFound = 0
        # state_exp = []
        for s, p, o in predList:
            if self.predicate == p and self.obj == o:
                if self.yes:
                    state_exp.append([s, p, o])
                    node = self.yes.traverse(predList, state_exp)
                predFound = 1
                # print([s, p, o])
                break;

        if predFound == 0:
            if self.no:
                # state_exp.append([s, p, o])
                node = self.no.traverse(predList, state_exp)

        return node

def create_tree():
    root_node = TreeNode("visible", "key")
    root_node.insert("yes", ["carrying", "key"])
    root_node.insert("no", [], " ")
    left = root_node.yes

    left.insert("yes", ["visible", "door"])
    left.insert("no", [], " ")

    left = left.yes

    left.insert("yes", ["locked", "door"])
    left.insert("no", [], " ")

    left = left.yes

    left.insert("yes", [], " ")
    left.insert("no", ["visible", "door"])

    right = left.no

    right.insert("yes", ["visible", "goal"])
    right.insert("no", [], " ")

    left = right.yes

    left.insert("yes", [], " ")
    left.insert("no", [], " ")

    root_node.print()
    return root_node

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)

root = create_tree()


if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
