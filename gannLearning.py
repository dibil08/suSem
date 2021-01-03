import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from gaNN import *

lr = 1e-3
gym.envs.register(
    id='CartPoleExtraLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=5000,
)

env = gym.make("CartPoleExtraLong-v0")
env.reset()
goal_steps = 5000
score_requirement = 500
initial_games = 50000
scoringGamesNumber=30
training_data_file="saved.npy"
bestModelScore=0
bestModelScoreText=''

totalGenerationNumber=20
offspring_size=10
top_fitnes_parents_number=4

n_epoch=1


training_data=getTrainingData(env,initial_games,score_requirement,goal_steps,training_data_file)

parameters=[
    #     {
    #     "totalGenerationNumber":30,
    #     "offspring_size":50,
    #     "top_fitnes_parents_number":11,
    #     "n_epoch":1,
    #     "useMutation":True,
    #     "mutation_percentWeight":0.001,
    #     "mutation_percentBias":0.1,
    #     "useTraining":False,
    # }
    # {
    #     "totalGenerationNumber":15,
    #     "offspring_size":10,
    #     "top_fitnes_parents_number":4,
    #     "n_epoch":1,
    #     "useMutation":True,
    #     "mutation_percentWeight":0.1,
    #     "mutation_percentBias":0.1,
    #     "useTraining":True,
    # },{
    #     "totalGenerationNumber":15,
    #     "offspring_size":10,
    #     "top_fitnes_parents_number":4,
    #     "n_epoch":1,
    #     "useMutation":True,
    #     "mutation_percentWeight":0.01,
    #     "mutation_percentBias":0.1,
    #     "useTraining":True,
    # },{
    #     "totalGenerationNumber":15,
    #     "offspring_size":10,
    #     "top_fitnes_parents_number":4,
    #     "n_epoch":1,
    #     "useMutation":True,
    #     "mutation_percentWeight":0.001,
    #     "mutation_percentBias":0.1,
    #     "useTraining":True,
    # },
    # {
    #     "totalGenerationNumber":15,
    #     "offspring_size":10,
    #     "top_fitnes_parents_number":4,
    #     "n_epoch":2,
    #     "useMutation":False,
    #     "mutation_percentWeight":0.001,
    #     "mutation_percentBias":0.1,
    #     "useTraining":True,
    # },{
    #     "totalGenerationNumber":10,
    #     "offspring_size":10,
    #     "top_fitnes_parents_number":4,
    #     "n_epoch":5,
    #     "useMutation":False,
    #     "mutation_percentWeight":0.001,
    #     "mutation_percentBias":0.1,
    #     "useTraining":True,
    # },
    {
        "totalGenerationNumber":10,
        "offspring_size":10,
        "top_fitnes_parents_number":5,
        "n_epoch":10,
        "useMutation":False,
        "mutation_percentWeight":0.001,
        "mutation_percentBias":0.1,
        "useTraining":True,
    },
]

for parameter in parameters:
    totalGenerationNumber=parameter["totalGenerationNumber"]
    offspring_size=parameter["offspring_size"]
    top_fitnes_parents_number=parameter["top_fitnes_parents_number"]
    n_epoch = parameter["n_epoch"]
    useMutation = parameter["useMutation"]
    mutation_percentWeight= parameter["mutation_percentWeight"]
    mutation_percentBias= parameter["mutation_percentBias"]
    useTraining=parameter["useTraining"]
    generationModels=[False]*offspring_size
    generationAgents=[False]*offspring_size
    allGenerationScores=[]
    with open('output.txt', 'a') as f:
        print("{}".format(parameter), file=f)

    for currentGeneration in range(totalGenerationNumber):
        generationScores=([0]*offspring_size)
        weights=[]
        bias=[]
        for subjectId in range(offspring_size):
            if useTraining:
                generationModels[subjectId]=train_model(training_data,n_epoch,"Gen: {} Subject: {}".format(currentGeneration,subjectId),lr,createModel(training_data,lr,generationAgents[subjectId]))
            else:
                generationModels[subjectId]=createModel(training_data,lr,generationAgents[subjectId])
            generationScores[subjectId]=score_model(env,generationModels[subjectId],scoringGamesNumber)
        
        bestFitnes=select_mating_pool(generationModels,fitness(generationScores),top_fitnes_parents_number)
        parentWeights=[]
        parentBias=[]
        for model in bestFitnes:
            (w,b,layers) = model_to_vector(model)
            parentWeights.append(w)
            parentBias.append(b)
        parentWeights=np.array(parentWeights)
        parentBias=np.array(parentBias)
        offspingsWeights=crossover(parentWeights,[offspring_size,len(parentWeights[0])])
        offspingsBias=crossover(parentBias,[offspring_size,len(parentBias[0])])
        
        
        if useMutation:
            offspingsWeights=mutation(offspingsWeights,mutation_percentWeight)
            offspingsBias=mutation(offspingsBias,mutation_percentBias)
        
        for subjectId in range(offspring_size):
            generationAgents[subjectId]=(offspingsWeights[subjectId],offspingsBias[subjectId])

        allGenerationScores.append(generationScores)
        with open('output.txt', 'a') as f:
            avg =0
            std=0
            genScores=[]
            m=0
            for score in allGenerationScores[currentGeneration]:
                avg+=score["avg"]
                std+=score["std"]
                genScores.append(score["avg"])
            print("{} ; {} ; {}; {} ; {}\n".format(currentGeneration,avg/offspring_size,max(genScores),std/offspring_size,genScores), file=f)

    

