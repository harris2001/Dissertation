import numpy as np
from itertools import combinations

def generate_population(n_agents, n_strategies):
  """Generates a population of agents with different strategies.

  Args:
    n_agents: The number of agents in the population.
    n_strategies: The number of strategies available to each agent.

  Returns:
    A list of agents, each with a different strategy.
  """

  # Initialize the population of agents.
  agents = []
  for i in range(n_agents):
    agents.append(Agent(n_strategies))

  # Generate a list of all possible combinations of strategies.
  combinations_of_strategies = list(combinations(range(n_strategies), n_agents))

  # Assign a strategy to each agent.
  for agent, combination in zip(agents, combinations_of_strategies):
    agent.strategy = combination

  # Calculate the diversity of the population.
  diversity = 0
  for i in range(n_agents):
    for j in range(i + 1, n_agents):
      if np.array_equal(agents[i].strategy, agents[j].strategy):
        diversity += 1

  # If the diversity is too low, re-generate the population.
  while diversity < n_agents:
    agents = []
    for i in range(n_agents):
      agents.append(Agent(n_strategies))

    combinations_of_strategies = list(combinations(range(n_strategies), n_agents))

    for agent, combination in zip(agents, combinations_of_strategies):
      agent.strategy = combination

    diversity = 0
    for i in range(n_agents):
      for j in range(i + 1, n_agents):
        if np.array_equal(agents[i].strategy, agents[j].strategy):
          diversity += 1

  return agents

class Agent:
  """A class representing an agent with a strategy."""

  def __init__(self, n_strategies):
    """Initializes an agent with a strategy.

    Args:
      n_strategies: The number of strategies available to the agent.
    """

    self.strategy = np.random.choice(n_strategies)

  def get_strategy(self):
    """Returns the agent's strategy."""

    return self.strategy

