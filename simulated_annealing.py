import numpy as np
from KNN_spikes import KNN_classifier
import random

# Simulated annealing function
def anneal(solution, demand, alpha, iterations, var, trainD,trainLab,testD, testLab):
    oldCost = cost(solution, demand, trainD,trainLab, testD, testLab)
    costValues = list()
    costValues.append(oldCost)
    T = 1.0
    T_min = 0.001
    while T > T_min:
        i = 1
        while i <= iterations:
            print("Iteration : " + str(i) + " Cost : " + str(oldCost))
            new_solution = neighbour(solution, var)
            newCost = cost(new_solution, demand, trainD, trainLab, testD, testLab)
            ap = acceptanceProbability(oldCost, newCost, T)
            if ap > random.random():
                solution = new_solution
                oldCost = newCost
            i += 1
            costValues.append(oldCost)
        T = T * alpha
    return solution, oldCost, costValues

def acceptanceProbability(oldCost, newCost, T):
    return np.exp((oldCost - newCost) / T)

# Cost function
def cost(supply, demand,train_data,labels,test_data, test_labels):
    # Find error between demand and supply
    Pnc = int(round(supply[0]))
    Knn = int(round(supply[1]))
    Knp = int(round(supply[2]))
    Pred, _ = KNN_classifier(train_data,labels,test_data,Pnc, Knn, Knp)
    Score = Performance(test_data,test_labels,Pred, Evaluation = False)
    DCost = demand - Score
    return DCost

def neighbour(solution, d):
    delta = np.random.random((3, 1))
    scale = np.full((3, 1), 2 * d)
    offset = np.full((3, 1), 1.0 - d)
    var = np.multiply(delta, scale)
    m = np.add(var, offset)
    new_solution = []
    for i in range(len(solution)):
        parameter = m[i][0] * solution[i]
        # If any of the solutions PCA components, Kn neighbours, P norm
        # is below 0.5 make it 1 as it will give an error.
        if parameter < 0.5:
            parameter = 1
        new_solution.append(parameter)
    return new_solution