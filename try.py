import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from gaNN import *
LR = 1e-2
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 50000


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

model = neural_network_model(10)
agent = model_to_vector(model)
vector_to_model(agent[0],agent[1],model,lr)
child = crossover(np.array([agent[0],np.zeros(len(agent[0]))],np.float32),np.array([5,len(agent[0])]))
childBias = crossover(np.array([agent[1],np.zeros(len(agent[1]))],np.float32),np.array([5,len(agent[1])]))
mutChild =mutation(childBias,20)
print(child)
print(mutChild)