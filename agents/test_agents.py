import random
from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType


class FoldAgent(Agent):
    def __name__(self):
        return "FoldAgent"

    def act(self, observation, reward, terminated, truncated, info):
        action_type = action_types.FOLD.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


# TODO: Implement the following agents
class CallingStationAgent(Agent):
    # Always calls/checks
    def __name__(self):
        return "CallingStationAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        elif observation["valid_actions"][action_types.CHECK.value]:
            action_type = action_types.CHECK.value
        raise_amount = 0
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard

class AllInAgent(Agent):
    # Always goes all in
    def __name__(self):
        return "AllInAgent"

    def act(self, observation, reward, terminated, truncated, info):
        if observation["valid_actions"][action_types.RAISE.value]:
            action_type = action_types.RAISE.value
        elif observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        elif observation["valid_actions"][action_types.CHECK.value]:
            action_type = action_types.CHECK.value

        raise_amount = observation["max_raise"]
        card_to_discard = -1
        return action_type, raise_amount, card_to_discard


class RandomAgent(Agent):
    # Randomly chooses an action
    def __name__(self):
        return "RandomAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = [n for i, n in enumerate(observation["valid_actions"]) if i == 1]
        action_type = random.choice(valid_actions)
        if action_type == action_types.RAISE.value:
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
        else:
            raise_amount = 0
        card_to_discard = random.randint(-1, 1)
        return action_type, raise_amount, card_to_discard


all_agent_classes = (FoldAgent, CallingStationAgent, AllInAgent, RandomAgent)