import random
from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class PlayerAgent(Agent):
    def __name__(self):
        return "ProbabilityPlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.evaluator = PokerEnv().evaluator

    def _compute_equity(self, my_cards, community_cards, opp_discarded, opp_drawn,
                        num_simulations=500, replacement_card=None, discard_idx=None):
        """
        Monte Carlo equity estimate. If discard_idx is set, simulates discarding that card
        and drawing a random replacement (replacement_card=None means draw unknown).
        """
        if discard_idx is not None:
            discarded = my_cards[discard_idx]
            kept = my_cards[1 - discard_idx]
            # Build shown set including the discarded card (removed from play)
            shown = {kept, discarded}
        else:
            shown = set(my_cards)

        for c in community_cards:
            shown.add(c)
        if opp_discarded != -1:
            shown.add(opp_discarded)
        if opp_drawn != -1:
            shown.add(opp_drawn)

        non_shown = [i for i in range(27) if i not in shown]

        opp_known = []
        if opp_drawn != -1:
            opp_known.append(opp_drawn)

        wins = 0
        valid = 0
        for _ in range(num_simulations):
            opp_needed = 2 - len(opp_known)
            board_needed = 5 - len(community_cards)

            if discard_idx is not None:
                # Need 1 extra card for replacement
                sample_size = 1 + opp_needed + board_needed
            else:
                sample_size = opp_needed + board_needed

            if sample_size > len(non_shown):
                continue

            sample = random.sample(non_shown, sample_size)

            if discard_idx is not None:
                new_card = sample[0]
                rest = sample[1:]
                effective_my_cards = [new_card, kept] if discard_idx == 0 else [kept, new_card]
            else:
                rest = sample
                effective_my_cards = my_cards

            opp_cards = opp_known + rest[:opp_needed]
            full_board = community_cards + rest[opp_needed:]

            my_hand = list(map(int_to_card, effective_my_cards))
            opp_hand = list(map(int_to_card, opp_cards))
            board = list(map(int_to_card, full_board))

            my_rank = self.evaluator.evaluate(my_hand, board)
            opp_rank = self.evaluator.evaluate(opp_hand, board)
            if my_rank < opp_rank:
                wins += 1
            valid += 1

        return wins / valid if valid > 0 else 0.0

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = list(observation["my_cards"])
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_discarded = observation["opp_discarded_card"]
        opp_drawn = observation["opp_drawn_card"]
        valid_actions = observation["valid_actions"]

        # Proactive discard: check if discarding either card improves expected equity
        if valid_actions[action_types.DISCARD.value]:
            base_equity = self._compute_equity(my_cards, community_cards, opp_discarded, opp_drawn,
                                               num_simulations=300)
            eq0 = self._compute_equity(my_cards, community_cards, opp_discarded, opp_drawn,
                                       num_simulations=300, discard_idx=0)
            eq1 = self._compute_equity(my_cards, community_cards, opp_discarded, opp_drawn,
                                       num_simulations=300, discard_idx=1)

            best_discard = None
            best_equity = base_equity
            # Only discard if improvement is meaningful (>0.05 threshold to avoid info leakage for minor gains)
            if eq0 > best_equity + 0.05:
                best_equity = eq0
                best_discard = 0
            if eq1 > best_equity + 0.05:
                best_equity = eq1
                best_discard = 1

            if best_discard is not None:
                self.logger.debug(f"Discarding card {best_discard}: equity {base_equity:.2f} -> {best_equity:.2f}")
                return action_types.DISCARD.value, 0, best_discard

        # Betting decision based on equity vs pot odds
        equity = self._compute_equity(my_cards, community_cards, opp_discarded, opp_drawn,
                                      num_simulations=500)

        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        self.logger.debug(f"Street {observation['street']}: equity={equity:.2f}, pot_odds={pot_odds:.2f}")

        if equity > 0.75 and valid_actions[action_types.RAISE.value]:
            raise_amount = int(pot_size * 0.75)
            raise_amount = max(raise_amount, observation["min_raise"])
            raise_amount = min(raise_amount, observation["max_raise"])
            return action_types.RAISE.value, raise_amount, -1
        elif equity >= pot_odds and valid_actions[action_types.CALL.value]:
            return action_types.CALL.value, 0, -1
        elif valid_actions[action_types.CHECK.value]:
            return action_types.CHECK.value, 0, -1
        else:
            return action_types.FOLD.value, 0, -1

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand ended with reward: {reward}")
