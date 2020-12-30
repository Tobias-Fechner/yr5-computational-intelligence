from random import randint, random, choice
from math import ceil
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def individual(length, min, max):
    """
    Function creates a single individual of the population using a random integer between the min/ max parameter inputs.
    :param length: number of random ints to generate
    :param min: min int of range
    :param max: max int of range
    :return: returns list of size 'length' random numbers within min/max range
    """
    return [ randint(min,max) for x in range(length) ]

def population(count, length, min, max):
    """
    Creates a population of individuals.
    :param count: the number of individuals in the population
    :param length: the number of values per individual
    :param min: the minimum possible value in an individual's list of values
    :param max: the maximum possible value in an individual's list of values
    :return: returns a list of individuals of size 'length
    """
    return [ individual(length, min, max) for x in range(count) ]

def fitness(individual, target):
    """
    Determines the fitness of a given individual. Higher is better.
    :param individual: individual to be measured
    :param target: target value
    :return: fitness metric
    """

    return abs(target-sum(individual))

def grade(population, target):
    """
    Grade the overall population by finding the average fitness.
    :param population:
    :param target:
    :return:
    """
    return sum(fitness(x, target) for x in population) / len(population)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01, femalePortion=0.5):

    # Assert all fractions between 0-1
    assert 0 <= femalePortion <= 1
    assert 0 <= random_select <= 1
    assert 0 <= mutate <= 1
    assert 0 <= retain <= 1

    # Grade each individual within the population
    individualGrades = [(fitness(x, target), x) for x in pop]

    # Rank population based on individual grades
    graded = [x[1] for x in sorted(individualGrades)]

    # Calculate number of individuals to retain as parents of the next generation
    retain_length = ceil(len(graded)*retain)

    # Retain the desired number of elements from the leading side of the list as the parents of next generation
    parents = graded[:retain_length]

    # Add other individuals to parents group to promote genetic diversity
    for individual in graded[retain_length:]:

        # Configured probability used to decide if each individual from remaining population is added to parent gene pool
        if random_select > random():
            parents.append(individual)

    # Mutate some individuals
    for individual in parents:

        # Configured probability used to decide if each individual from parent gene pool is mutated
        if mutate > random():

            # Generate value to randomly index individual's contents
            randIndex = choice(range(len(individual)))

            # Generate a random integer between the individual's min/ max and store value using random index
            individual[randIndex] = randint(min(individual), max(individual))

    children = []
    lenParents = len(parents)
    desiredChildren = len(pop) - lenParents

    # Create children from high performing parent group to maintain population size
    while len(children) < desiredChildren:

        # Get female and male index
        femaleIndex = randint(0, lenParents-1)
        maleIndex = randint(0, lenParents-1)

        if maleIndex == femaleIndex:
            pass

        else:

            # Get female and male values
            female = parents[femaleIndex]
            male = parents[maleIndex]

            # Determine proportion of female chromosomes to
            split = int(len(female)*femalePortion)
            logging.info("Split at {}. Parent chromosomes: {}; Child gets {} from female and {} from male.".format(
                split, len(male), len(female[:split]), len(male[split:])))

            # Split individual chromosomes based on desired female portion of chromosomes
            # Create child with half of male chromosomes and other half of female's chromosomes
            child = male[:split] + female[split:]


            # Add child to list
            children.append(child)

    # Return list of children and parent individuals to be next generation
    parents.extend(children)
    return parents
