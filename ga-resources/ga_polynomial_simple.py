from random import randint, random, choice
from math import ceil
import logging
import numpy as np
import pandas as pd

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

def logMatingPoolInfo(matingPool, stage):
    logging.info("Length of mating pool after {}: {}".format(stage, len(matingPool)))
    logging.info("First individual: {}".format(matingPool[0]))
    logging.info("Last individual: {}".format(matingPool[-1]))

    return None

def createIndividual(min, max, individualLength):
    """
    Function creates a single individual of the population using a random integer between the min/ max parameter inputs.
    :param individualLength: number of random ints to generate
    :param min: min int of range
    :param max: max int of range
    :return: returns list of size 'length' random numbers within min/max range
    """
    return np.random.randint(min, max, individualLength)

def createPopulation(count, min, max, individualLength):
    """
    Creates a population of individuals.
    :param count: the number of individuals in the population
    :param individualLength: the number of values per individual
    :param min: the minimum possible value in an individual's list of values
    :param max: the maximum possible value in an individual's list of values
    :return: returns a list of individuals of size 'length
    """
    return [createIndividual(min, max, individualLength) for x in range(count)]

def generatePointsFromCoefficients(individual, xRange):
    """
    Generate y values based on coefficient predictions from given individual for a range of x values
    :param individual:
    :param xRange:
    :return: returns array of y values that represent predicted polynomial
    """

    # Extract polynomial coefficients from individual
    a1 = individual[0]
    a2 = individual[1]
    a3 = individual[2]
    a4 = individual[3]
    a5 = individual[4]
    c = individual[5]

    yPrediction = np.array([])

    # For a range of values of x, calculate the target and predicted value of y
    for x in xRange:
        yPrediction = np.append(yPrediction, a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x + c)

    return yPrediction

def getSumAbsError(individual, target):
    """
    Gets the sum of absolute errors between individual's chromosome's values and target values
    :param individual: individual/ chromosome
    :param target: target chromosome values
    :return: single value for sum of absolute errors
    """

    # Get array of errors between individual's chromosome values and target chromosome values
    errors = np.subtract(individual, target)
    # Find absolute values of each error
    absErrors = np.absolute(errors)
    sumAbs = np.sum(absErrors)

    try:
        assert isinstance(sumAbs, float)
    except AssertionError:
        logging.warning("Expected float. Got {}".format(type(sumAbs)))
        raise AssertionError

    # Return the sum of all errors
    return sumAbs

def getAvgPopulationError(population, target):
    """
    Determines the fitness of a given individual by calculating MSE between prediction and target over a range of values of x.
    :param target: target individual chromosome values
    :param population: a population of individuals
    :return: average fitness of population
    """
    # Calculate average abs error for each individual in population
    errors = [getSumAbsError(individual, target) for individual in population]

    return np.average(errors)

def selectByRoulette(population, target, retain):

    logging.info("ENTERING STAGE: Selection - roulette.")

    # Calculate limit for number of parent chromosomes to select
    poolSize = ceil(len(population) * retain)
    logging.info("Desired mating pool size: {}".format(poolSize))

    errors = np.array([])

    # Get total fitness of population (cumulative sum of each individual's error)
    for individual in population:
        # Get errors for each individual
        errors = np.append(errors, getSumAbsError(individual, target))

    # Find sum of errors for all individuals
    errorSum = np.sum(errors)

    # Calculate selection probability for each individual as a proportion of total fitness for population: 1 - errors/errorSum
    # Largest errors have smallest probabilities
    selectProbabilities = np.divide(np.subtract(errorSum, errors), errorSum*100)

    # Assert probabilities all are floats within range [0,1]
    try:
        assert 0 <= selectProbabilities.all() <= 1
    except AssertionError:
        print("Errors: \n", errors)
        print("Errors sum: \n", errorSum)
        print("Selection probabilities: \n", selectProbabilities)
        raise AssertionError("Something not right about the probabilities...")

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

def selectByRandom(population, target, random_select, retain):

    logging.info("ENTERING STAGE: Selection - random.")

    # Calculate number of individuals to retain as matingPool of the next generation
    retainLength = ceil(len(population)*retain)

    logging.info("Retaining {} individuals.".format(retainLength))

    # Create dataframe from population list
    df = pd.DataFrame({'individual': population})

    # Create second column with grade for each individual in population
    df['grade'] = df.apply(lambda x: getSumAbsError(x['individual'], target), axis=1)

    # Sort rows based on grade of individual
    graded = df.sort_values('grade')

    # Create list of best performers to retain
    matingPool = graded.iloc[:retainLength, 0].tolist()

    # Select subset of rows from graded dataframe with remaining individuals
    remaining = graded.iloc[retainLength:, :]

    # Create new column with random float between range [0,1] for each remaining individual
    remaining.insert(2, 'random', np.random.uniform(0.0, 1.0, len(remaining)))

    # Randomly select a subset of these individuals by comparing to random_select float
    mask = remaining['random'] < random_select
    selected = remaining.loc[mask]

    # Cast column with individual's chromosome values to list to append to mating pool
    matingPool.extend(selected['individual'].tolist())

    logging.info("Added an additional {} randomly selected individuals to increase mating pool size to {}.".format(len(selected), len(matingPool)))
    logMatingPoolInfo(matingPool, 'selection')

    # Return list of combined parents with randomly selected individuals from remaining pool
    return matingPool

def mutateByRandom(matingPool, mutate):

    logging.info("ENTERING STAGE: Mutation - random.")

    mutateCount = 0

    # Mutate some individuals by introducing a random integer within the range of the individuals min/ max values
    # at a random index of the chromosome
    for individual in matingPool:

        # Configured probability used to decide if each individual from parent gene pool is mutated
        if mutate > random():

            logging.info("Individual selected for mutation: {}".format(individual))

            # Generate value to randomly index individual's contents
            randIndex = choice(range(len(individual)))

            # Generate a random integer between the individual's min/ max and store value using random index
            individual[randIndex] = randint(min(individual), max(individual))

            logging.info("Individual after mutation: {}".format(individual))

            mutateCount += 1

    logging.info("{} chromosomes mutated in this cycle of evolution.".format(mutateCount))
    logMatingPoolInfo(matingPool, 'mutation')

    return matingPool

def evolve(population,
           target,
           retain=0.2,
           random_select=0.05,
           mutate=0.01,
           femalePortion=0.5,
           select='roulette'):

    # Assert all fractions within range [0,1]
    assert 0 <= femalePortion <= 1
    assert 0 <= random_select <= 1
    assert 0 <= mutate <= 1
    assert 0 <= retain <= 1

    if 'roulette' in select:
        matingPool = selectByRoulette(population, target, retain)
    elif 'random' in select:
        matingPool = selectByRandom(population, target, random_select, retain)
    else:
        raise ValueError("Please specify a selection methodology that has been implemented.")

    # Mutate some of the individuals within the mating pool
    matingPool = mutateByRandom(matingPool, mutate)

    children = []
    poolSize = len(matingPool)
    desiredChildren = len(population) - poolSize

    logging.info("ENTERING STAGE: Crossover. {} children desired to fill population size back to {}.".format(desiredChildren, poolSize+desiredChildren))

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
            child = male[:split] + female[split:]

            # Add child to list
            children.append(child)

    # Return list of children and parent individuals to be next generation
    matingPool.extend(children)

    logging.info("{} children created from crossover.".format(len(children)))
    logMatingPoolInfo(matingPool, 'crossover')
    logging.info("Evolution cycle complete!")

    return matingPool
