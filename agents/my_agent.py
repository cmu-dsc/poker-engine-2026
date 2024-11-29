import random
from agents.agent import Agent
from gym_env import PokerEnv
from agents.prob_agent import ProbabilityAgent

action_types = PokerEnv.ActionType

class MyAgent(Agent):
    def __name__(self):
        return "MyAgent"

    def act(self, observation, reward, terminated, truncated, info):
        prob_agent = ProbabilityAgent()
        return prob_agent.act(observation, reward, terminated, truncated, info)
        # valid_actions = observation["valid_actions"]
        # action_type = random.choice(valid_actions)
        # raise_amount = observation["min_raise"]
        # card_to_discard = -1
        # return action_type, raise_amount, card_to_discard
