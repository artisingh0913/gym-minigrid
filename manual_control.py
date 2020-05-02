#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
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


def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    # new code
    o = obs["image"]
    o = o.transpose()
    print(o)

    o = preprocess(o)

    rdf(o)

    # old code

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)


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


# class TreeNode:
#     def __init__(self, data, old_state):
#         # data represents the feature upon which the node was split when fitting the training data
#         # data = None for leaf node
#         self.data = data
#         # children of a node are stored as a dicticionary with key being the value of feature upon which the node was split
#         # and the corresponding value stores the child TreeNode
#         self.children = {}
#         # output represents the old_state at this instance of the decision tree
#         self.output = old_state
#         # index will be used to assign a unique index to each node
#         self.index = -1

#     def add_child(self, feature_value, obj):
#         self.children[feature_value] = obj

#     def find_split(self):
#         for c in range(self.col_count):
#             self.find_better_split(c)
#         if self.is_leaf: return
#         x = self.split_col
#         lhs = np.nonzero(x <= self.split)[0]
#         rhs = np.nonzero(x > self.split)[0]
#         self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)
#         self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)

# class DecisionTree:
#     def __init__(self):
#         # root represents the root node of the decision tree built after fitting the training data
#         self.__root = None

#     def __decision_tree
#     OBJECT_TO_IDX_OLD

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
    'left': 0,
    'right': 1,
    'forward': 2,
    'toggle': 3,
    'pickup': 4,
    'drop': 5,
    'enter': 6
}

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


class Tree:
    def __init__(self):
        self.__root = None


class TestNode:
    def __init__(self, predicate, obj, n=0):

        self.nodeType = n #test, 1 -> #leaf
        self.parent = None
        self.yes = None
        self.no = None

        # for test nodes
        if self.nodeType == 0:
            self.predicate = predicate
            self.obj = obj

        # for leaf nodes
        else:
            self.expression = []
            self.Q_val = list(0 for i in range(len(ACTION_TO_IDX)))

    def insert(self, side, val):
        # Compare the new value with the parent node
        if len(val) != 0:
            if side == "yes":
                self.yes = TestNode(val[0], val[1])
                self.yes.parent = self
            else:
                self.no = TestNode(val[0], val[1])
                self.no.parent = self
        else:
            if side == "yes":
                self.yes = TestNode("", "", 1)
                self.yes.parent = self
            else:
                self.no = TestNode("", "", 1)
                self.yes.parent = self

    def print(self):
        if self.yes:
            self.yes.print()
        print(self.predicate, self.obj),
        if self.no:
            self.no.print()



#
# class LeafNode:
#     def __index__(self):
#


args = parser.parse_args()

env = gym.make(args.env)

# decisionTree = Tree()
root = TestNode("visible", "key")
root.insert("yes", ["carrying", "key"])
root.insert("no", [])
left = root.yes
# right = root.no

left.insert("yes", ["visible", "door"])
left.insert("no", [])

left = left.yes
#right

left.insert("yes", ["locked", "door"])
left.insert("no", [])

root.print()


if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
