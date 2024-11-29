import random
from agents.agent import Agent
from treys import Evaluator
from gym_env import PokerEnv


action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class ProbabilityAgent(Agent):
    # Chooses an action based on the probability of winning
    def __init__(self, logger=None):
        super().__init__(logger)
        self.evaluator = Evaluator()         

    def __name__(self):
        return "ProbabilityAgent"

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = observation["my_cards"]
        community_cards = observation["community_cards"]
        opp_discarded_card = observation["opp_discarded_card"]
        opp_drawn_card = observation["opp_drawn_card"]
        
        # Remove unshown cards and convert to list of integers
        my_cards = [int(card) for card in my_cards]
        community_cards = [card for card in community_cards if card != -1]
        opp_discarded_card = [opp_discarded_card] if opp_discarded_card != -1 else []
        opp_drawn_card = [opp_drawn_card] if opp_drawn_card != -1 else []

        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        non_shown_cards = [i for i in range(27) if i not in shown_cards]

        def evaluate_hand(cards):
            # Given a list of cards, return True if my hand is better than opp's hand
            # Cards are in the order of my_cards, community_cards, opp_cards
            my_cards, opp_cards, community_cards = cards
            my_cards = [int(card) for card in my_cards]
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)

            return my_hand_rank < opp_hand_rank

        res = 0
        num_simulations = 1000
        self.logger.debug("Running Monte Carlo Simulation with %d simulations", num_simulations)

        for _ in range(num_simulations):
            num_cards_to_draw = 7 - len(community_cards) - len(opp_drawn_card)
            drawn_cards = random.sample(non_shown_cards, num_cards_to_draw)
            res += evaluate_hand((my_cards,opp_drawn_card + drawn_cards[:2-len(opp_drawn_card)],community_cards + drawn_cards[2-len(opp_drawn_card):]))        
        equity = res / num_simulations

        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size)

        self.logger.debug(f"Equity: {equity}, Pot Odds: {pot_odds}")

        raise_amount = 0
        card_to_discard = -1

        if equity > 0.8 and observation["valid_actions"][action_types.RAISE.value]:
            raise_amount = min(int(pot_size*0.75), observation["max_raise"])
            raise_amount = max(raise_amount, observation["min_raise"])
            action_type = action_types.RAISE.value
        elif equity >= pot_odds and observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        elif observation["valid_actions"][action_types.CHECK.value]:
            action_type = action_types.CHECK.value
        elif observation["valid_actions"][action_types.DISCARD.value]:
            action_type = action_types.DISCARD.value
            card_to_discard = random.randint(0, 1)
        else:
            action_type = action_types.FOLD.value

        self.logger.debug(f"Action: {action_type}, Raise Amount: {raise_amount}, Card to Discard: {card_to_discard}")

        return action_type, raise_amount, card_to_discard



