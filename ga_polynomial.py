from random import randint, random, choice
from math import ceil
import logging
import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

def createIndividual(min, max, length):
    """
    Function creates a single individual of the population using a random integer between the min/ max parameter inputs.
    :param length: number of random ints to generate
    :param min: min int of range
    :param max: max int of range
    :return: returns list of size 'length' random numbers within min/max range
    """
    return np.random.randint(min, max, length)

def createPopulation(count, min, max, length):
    """
    Creates a population of individuals.
    :param count: the number of individuals in the population
    :param length: the number of values per individual
    :param min: the minimum possible value in an individual's list of values
    :param max: the maximum possible value in an individual's list of values
    :return: returns a list of individuals of size 'length
    """
    return [createIndividual(min, max, length) for x in range(count)]

def generatePredictions(individual):

    # Extract polynomial coefficients from individual
    a1 = individual[0]
    a2 = individual[1]
    a3 = individual[2]
    a4 = individual[3]
    a5 = individual[4]
    c = individual[5]

    yTarget = np.array([])
    yPrediction = np.array([])

    # For a range of values of x, calculate the target and predicted value of y
    for x in np.linspace(-1,1, 20):
        yTarget = np.append(yTarget, 25*x**5 + 18*x**4 + 0 - 14*x**2 + 7*x -19)
        yPrediction = np.append(yPrediction, a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x + c)

    return yTarget, yPrediction

def getMSE(individual):
    """
    Take in individual, which are effectively coefficient predictions, then use a range of x values to generate a corresponding y value
    with the same equation pattern as the polynomial we are trying to fit. Return the mean square error by comparing with y values
    generated from target polynomial.
    :param individual:
    :return:
    """

    yTarget, yPrediction = generatePredictions(individual)

    # Calculate error as difference between each y value prediction and corresponding target
    errors = np.subtract(yTarget, yPrediction)

    # Square each error term
    squaredErrors = np.power(errors, 2)

    # Calculate weighted average of the squared errors array
    meanSquaredErrors = np.average(squaredErrors)

    return meanSquaredErrors

def getAvgAbsError(individual):

    yTarget, yPrediction = generatePredictions(individual)

    errors = np.subtract(yTarget, yPrediction)
    absErrors = np.absolute(errors)

    return np.average(absErrors)

def getFitnesses(errors):

    target = np.zeros_like(errors)
    compare = np.equal(target, errors)


    if np.sum(errors) == 0:
        raise NotImplementedError
    else:
        fitnesses = np.divide(1,np.sum(errors))

    return fitnesses, np.average(fitnesses)

def getAvgPopulationFitness(population, errorMethod='mse'):
    """
    Determines the fitness of a given individual by calculating MSE between prediction and target over a range of values of x.
    :param population: population of individuals
    :param errorMethod: error calculation methodology
    :return: average fitness of population
    """
    if 'mse' in errorMethod:
        # Calculate MSE for each individual in population (pre-fitness)
        errors = [getMSE(individual) for individual in population]

        # Return weighted average of raw fitnesses as fitness for entire population
        _, avgFitness = getFitnesses(errors)
        return avgFitness

    elif 'abs' in errorMethod:
        # Calculate average abs error for each individual in population
        errors = [getAvgAbsError(individual) for individual in population]

        return np.average(errors)

    else:
        raise ValueError("Please specify error calculation methodology that has been implemented.")

def selectByRoulette(population, retain, errorMethod):

    # Calculate limit for number of parent chromosomes to select
    poolSize = ceil(len(population) * retain)
    logging.info("Desired mating pool size: {}".format(poolSize))

    errors = np.array([])

    if 'mse' in errorMethod:
        # Get total fitness of population (cumulative sum of each individual's error)
        for individual in population:
            # Get errors for each individual
            errors = np.append(errors, getMSE(individual))

    elif 'abs' in errorMethod:
        # Get total fitness of population (cumulative sum of each individual's error)
        for individual in population:
            # Get errors for each individual
            errors = np.append(errors, getAvgAbsError(individual))

    else:
        raise NotImplementedError("Please select an error method that has been implemented.")

    logging.info("{} errors calculated for the {} individuals.".format(len(errors), len(population)))

    # Get fitness, append to list of fitnesses
    fitnesses, _ = getFitnesses(errors)

    # Calculate selection probability for each individual as a proportion of total fitness for population
    selectProbabilities = np.divide(fitnesses, np.sum(fitnesses))

    # Assert probabilities all are floats within range [0,1] and sum to 1
    assert 0 <= selectProbabilities.all() <= 1
    assert abs(np.sum(selectProbabilities) - 1) < 0.0001

    # Generate cumulative sum, used to ensure at least one individual is always chosen
    selectProbabilitiesCS = np.cumsum(selectProbabilities)

    matingPool = []

    # Select individuals to add to mating pool using probability based on fitness of individual
    for individual, selectProbability in zip(population, selectProbabilitiesCS):
        if selectProbability > random():
            matingPool.append(individual)
        else:
            pass

        # Break when mating pool has reached size specified by retain parameter
        if len(matingPool) >= poolSize:
            break
        else:
            pass

    logging.info("Returning from selection stage with mating pool size {}.".format(len(matingPool)))

    assert len(matingPool) > 1

    # Cast list to numpy array
    matingPool = np.array(matingPool)

    # Return mating pool
    return matingPool.astype(int)

def selectByRandom(population, random_select, retain, errorMethod='mse'):

    # Calculate number of individuals to retain as matingPool of the next generation
    retainLength = ceil(len(population)*retain)

    if 'mse' in errorMethod:
        # Grade each individual within the population
        individualGrades = [(getMSE(individual), individual) for individual in population]
    elif 'abs' in errorMethod:
        # Grade each individual within the population
        individualGrades = [(getAvgAbsError(individual), individual) for individual in population]
    else:
        raise NotImplementedError("Please select an error method that has been implemented.")

    # Rank population based on individual grades
    graded = [x[1] for x in sorted(individualGrades)]

    # Retain the desired number of elements from the leading side of the list as the matingPool of next generation
    matingPool = graded[:retainLength]

    # Add other individuals to matingPool group to promote genetic diversity
    for individual in graded[retainLength:]:

        # Configured probability used to decide if each individual from remaining population is added to parent gene pool
        if random_select > random():
            matingPool.append(individual)

    return matingPool

def mutateByRandom(matingPool, mutate):

    mutateCount = 0

    logging.info("Mating pool: {}".format(matingPool))

    # Mutate some individuals by introducing a random integer within the range of the individuals min/ max values
    # at a random index of the chromosome
    for individual in matingPool:

        # Configured probability used to decide if each individual from parent gene pool is mutated
        if mutate > random():

            logging.info("Individual: {}".format(individual))

            # Generate value to randomly index individual's contents
            randIndex = choice(range(len(individual)))

            # Generate a random integer between the individual's min/ max and store value using random index
            individual[randIndex] = randint(min(individual), max(individual))

            mutateCount += 1

    logging.info("{} chromosomes mutated.".format(mutateCount))

    return matingPool

def evolve(population, retain=0.2, random_select=0.05, mutate=0.01, femalePortion=0.5, select='roulette', errorMethod='abs'):

    # Assert all fractions within range [0,1]
    assert 0 <= femalePortion <= 1
    assert 0 <= random_select <= 1
    assert 0 <= mutate <= 1
    assert 0 <= retain <= 1

    if 'roulette' in select:
        matingPool = selectByRoulette(population, retain, errorMethod)
    elif 'random' in select:
        matingPool = selectByRandom(population, random_select, retain, errorMethod)
    else:
        raise ValueError("Please specify a selection methodology that has been implemented.")

    # Mutate some of the individuals within the mating pool
    matingPool = mutateByRandom(matingPool, mutate)

    children = []
    poolSize = len(matingPool)
    desiredChildren = len(population) - poolSize

    # Create children from high performing parent group to maintain population size
    while len(children) < desiredChildren:

        # Get female and male index
        femaleIndex = randint(0, poolSize-1)
        maleIndex = randint(0, poolSize-1)

        if maleIndex == femaleIndex:
            continue

        else:

            # Get female and male values
            female = matingPool[femaleIndex]
            male = matingPool[maleIndex]

            # Determine proportion of female chromosomes to
            split = int(len(female)*femalePortion)

            # Split individual chromosomes based on desired female portion of chromosomes
            # Create child with half of male chromosomes and other half of female's chromosomes
            child = np.append(male[:split], female[split:])

            # Add child to list
            children.append(child)

    # Return list of children and parent individuals to be next generation
    nextGeneration = np.vstack((matingPool, children))

    return nextGeneration
