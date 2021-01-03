import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tensorflow.python.framework import ops
from statistics import stdev
import os
from itertools import combinations  

def createModel(training_data,lr,model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = np.array([i[1] for i in training_data])
    nnSize=[len(X[0]),128,256,512,256,128,2]
    if not model:
        model = neural_network_model(nnSize,lr)
    else:
        model = vector_to_model(model[0],model[1],nnSize,lr)
    return model

def train_model(training_data, n_epoch,modelName, lr,model):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = np.array([i[1] for i in training_data])
    model.fit( X, Y, n_epoch=n_epoch, snapshot_step=500, show_metric=True, run_id=modelName)
    return model


def neural_network_model(modelSize,lr):
    ops.reset_default_graph()
    network= None
    for size in modelSize[:-1]:
        if network is None:
            network = input_data(shape=[None, size, 1], name='input')
        else:
            network = fully_connected(network, size, activation='relu')
            network = dropout(network, 0.8)
    network = fully_connected(network, modelSize[-1], activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model

# Converting each solution from mmodel with matrix to vector.
def model_to_vector(model):
    modelIter=iter(model.get_train_vars())
    agentWeights=np.zeros(0,np.float32)
    agentBias=np.zeros(0,np.float32)
    # zip same layer weights and bias togheter
    for (weights,bias) in zip(modelIter,modelIter):
        for weightNode in model.get_weights(weights):
            agentWeights=np.append(agentWeights,weightNode, 0)
        agentBias=np.append(agentBias,model.get_weights(bias),0)
    return (agentWeights,agentBias,model.get_train_vars())

# Converting each solution from vector to matrix.
def vector_to_model(agentWeights, agentBias, modelSize, lr):
    # zip same layer weights and bias togheter
    model = neural_network_model(modelSize,lr)
    modelIter=iter(model.get_train_vars())
    prevArray=None
    prevArray2=None
    for (weights,bias) in zip(modelIter,modelIter):
        prevArray=model.get_weights(weights)
        prevArray2=model.get_weights(bias)
        layerMatrix=np.empty((0,weights.shape[1]),np.float32)
        for _ in range(weights.shape[0]):
            layerMatrix=np.append(layerMatrix,agentWeights[:weights.shape[1]].reshape(1,weights.shape[1]),axis=0)
            agentWeights=agentWeights[weights.shape[1]:]
        model.set_weights(weights,layerMatrix)
        model.set_weights(bias,agentBias[:bias.shape[0]])
        # print("Prev ")
        # print(prevArray)
        # print("\n  now")
        # print(layerMatrix)
        agentBias=agentBias[bias.shape[0]:]
    return model
            # layerWeights.append(agentWeights[])
def fitness(modelScores):
    fitness=[]
    for score in modelScores:
        fitness.append(score["avg"])
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents =[]
    sort = np.argsort(fitness)
    for i in range(num_parents):
        parents.append(pop[sort[-i]])
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint32(offspring_size[1]/2)
    k=0
    for (parent1_idx,parent2_idx) in combinations(range(len(parents)), 2)  :
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        k+=1
        if(k>=offspring_size[0]):
            break
    return offspring


def mutation(offspring_crossover, mutation_percent):
    num_mutations = np.uint32(((mutation_percent*offspring_crossover.shape[1]))/100)
    if(num_mutations!=0):
        mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
        return offspring_crossover
    return offspring_crossover


def score_model(env,model,n_games,render=False):
    scores =[]
    choices = []
    for game in range(n_games):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        while True:
            if render:
                env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)
    return {"avg": sum(scores)/len(scores),
            "std": stdev(scores),
            "scores": scores}


def getTrainingData(env,initial_games,score_requirement,goal_steps,file):
    if os.path.isfile(file):
        training_data = np.load(file,allow_pickle=True)
        return training_data
    else:
        # [OBS, MOVES]
        training_data = []
        # all scores:
        scores = []
        # just the scores that met our threshold:
        accepted_scores = []
        # iterate through however many games we want:
        for i in range(initial_games):
            print("games left {}".format(initial_games-i))
            score = 0
            # moves specifically from this environment:
            game_memory = []
            # previous observation that we saw
            prev_observation = []
            # for each frame in 200
            for _ in range(goal_steps):
                # choose random action (0 or 1)
                action = random.randrange(0,2)
                # do it!
                observation, reward, done, info = env.step(action)
                
                # notice that the observation is returned FROM the action
                # so we'll store the previous observation here, pairing
                # the prev observation to the action we'll take.
                if len(prev_observation) > 0 :
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score+=reward
                if done: break

            # IF our score is higher than our threshold, we'd like to save
            # every move we made
            # NOTE the reinforcement methodology here. 
            # all we're doing is reinforcing the score, we're not trying 
            # to influence the machine in any way as to HOW that score is 
            # reached.
            if score >= score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    # convert to one-hot (this is the output layer for our neural network)
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                        
                    # saving our training data
                    training_data.append([data[0], output])

            # reset env to play again
            env.reset()
            # save overall scores
            scores.append(score)
        
        # just in case you wanted to reference later
        training_data_save = np.array(training_data)
        np.save(file, training_data_save, allow_pickle=True)
        
        # some stats here, to further illustrate the neural network magic!
        # print('Average accepted score:',mean(accepted_scores))
        # print('Median score for accepted scores:',median(accepted_scores))
        # print(Counter(accepted_scores))
        
        return training_data