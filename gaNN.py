import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tensorflow.python.framework import ops
lr=1e-3
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
def vector_to_model(agentWeights, agentBias, model, lr):
    # zip same layer weights and bias togheter
    modelSize=[]
    modelIter=iter(model.get_train_vars())
    for (weights,_) in zip(modelIter,modelIter):
        if modelSize == []:
            modelSize.append(weights.shape[0])
        modelSize.append(weights.shape[1])
    print(modelSize)
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

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint32(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, mutation_percent):
    num_mutations = np.uint32(((mutation_percent*offspring_crossover.shape[1]))/100)
    print("NUM MUTS")
    print(num_mutations)
    if(num_mutations!=0):
        mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
        return offspring_crossover
    return offspring_crossover
