"""
CMU Poker Bot Competition Game Engine 2025

People working on this code, please refer to:
https://gymnasium.farama.org
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

Keep in mind gym doesn't inherently support multi-agent environments.
We will have to use the Tuple space to represent the observation space and
action space for each agent.
"""

import logging
import os
from enum import Enum

import gym
import numpy as np
from gym import spaces
from treys import Card, Evaluator

class WrappedEval(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, hand: list[int], board: list[int]) -> int:
        """
        This is the function that the user calls to get a hand rank.

        No input validation because that's cycles!
        """

        def ace_to_ten(treys_card: int):
            """Convert trey's representation of an Ace to trey's representation of a Ten"""
            as_str = Card.int_to_str(treys_card)
            alt = as_str.replace("A", "T")  # treys uses "T" for ten
            alt_as_treys = Card.new(alt)
            return alt_as_treys

        # check for the edge case of Ace used as high card after a 9
        alt_hand = list(map(ace_to_ten, hand))
        alt_board = list(map(ace_to_ten, board))

        reg_score = super().evaluate(hand, board)  # regular score
        alt_score = super().evaluate(
            alt_hand, alt_board
        )  # score if aces were tens

        if alt_score < reg_score:
            # explicit branch for pytorch coverage
            return alt_score

        return reg_score

class PokerEnv(gym.Env):
    SMALL_BLIND_PLAYER = 0
    BIG_BLIND_PLAYER = 1
    MAX_PLAYER_BET = 100

    RANKS = "23456789A"
    SUITS = "dhs"  # diamonds hearts spade

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer desiged for fast lookup & comparison
        """
        return Card.new(PokerEnv.int_card_to_str(card_int))

    @staticmethod
    def int_card_to_str(card_int: int):
        RANKS, SUITS = PokerEnv.RANKS, PokerEnv.SUITS
        rank = RANKS[card_int % len(RANKS)]
        suit = SUITS[card_int // len(RANKS)]
        return rank + suit

    class ActionType(Enum):
        FOLD = 0
        RAISE = 1
        CHECK = 2
        CALL = 3
        DISCARD = 4
        INVALID = 5

    def __init__(self, logger=None, small_blind_amount=1, num_hands=1, num_players=6):
        """
        Represents a single hand of poker.

        Args:
            logger: Optional logger instance
            small_blind_amount: Amount for small blind (deprecated for bomb-pot)
            num_hands: Number of hands in the match
            num_players: Number of players (1-6, default 6)
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.num_players = min(max(1, num_players), 6)  # Clamp between 1 and 6
        self.num_hands = num_hands
        self.current_hand = 0
        self._reset_count = 0  # Track number of resets

        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = small_blind_amount * 2
        self.min_raise = self.big_blind_amount
        self.acting_agent = PokerEnv.SMALL_BLIND_PLAYER
        self.last_street_bet = None
        self.evaluator = WrappedEval()

        # Action space is a Tuple (action_type, raise_amount, card_to_discard)
        # where action is a Discrete(4), raise_amount is a Discrete(100), and card_to_discard is a Discrete(3) (-1 means no card is discarded)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(len(self.ActionType) - 1),  # user can pass any besides INVALID
                spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                spaces.Discrete(3, start=-1),
            ]
        )

        # Card space is a Discrete(28), -1 means the card is not shown
        cards_space = spaces.Discrete((len(self.SUITS) * len(self.RANKS)) + 1, start=-1)

        # Single observation space is a Dict.
        # Since we have two players, acting_agent is a Discrete(2)
        # Make sure to check (acting_agent == agent_num) before taking an action
        # opp_shown_card is "0" if the opp's card is not shown
        # Two players, so the observation space is a Tuple of two single_observation_spaces
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),
                "acting_agent": spaces.Discrete(2),
                "my_cards": spaces.Tuple([cards_space for _ in range(2)]),
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_bet": spaces.Discrete(self.MAX_PLAYER_BET, start=1),
                "opp_discarded_card": cards_space,
                "opp_drawn_card": cards_space,
                "min_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "max_raise": spaces.Discrete(self.MAX_PLAYER_BET, start=2),
                "valid_actions": spaces.MultiBinary(len(self.ActionType) - 1),
            }
        )

        # Since we have two players, the observation space is a tuple of
        # (observation_space_one_player, observation_space_one_player)
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(2)])

        # New episode
        self.reset(seed=int.from_bytes(os.urandom(32)))

    def _get_valid_actions(self, player_num: int):
        """
        Returns valid action mask for 6-player bomb-pot rules.

        Street 0 (card selection): No betting actions, only card selection (handled separately)
        Street 1 (betting): CHECK, CALL, RAISE, FOLD
        Street 2+ (showdown): No actions

        Returns:
            List of 5 binary values indicating valid actions:
            [FOLD, RAISE, CHECK, CALL, DISCARD]
        """
        valid_actions = [1, 1, 1, 1, 1]

        # Street 0 (card selection): disable betting actions
        if self.street == 0:
            valid_actions[self.ActionType.FOLD.value] = 0
            valid_actions[self.ActionType.RAISE.value] = 0
            valid_actions[self.ActionType.CHECK.value] = 0
            valid_actions[self.ActionType.CALL.value] = 0
            # Card selection is handled separately, not via DISCARD action
            valid_actions[self.ActionType.DISCARD.value] = 0
            return valid_actions

        # Street 2+ (showdown): no actions
        if self.street >= 2:
            return [0, 0, 0, 0, 0]

        # Street 1 (betting round) - disable DISCARD
        valid_actions[self.ActionType.DISCARD.value] = 0

        # Get max bet across all players
        max_bet = max(self.bets)
        my_bet = self.bets[player_num]

        # Can't CHECK if facing a bet
        if my_bet < max_bet:
            valid_actions[self.ActionType.CHECK.value] = 0

        # Can't CALL if bets are equal
        if my_bet == max_bet:
            valid_actions[self.ActionType.CALL.value] = 0

        # Can't RAISE if at bet cap
        if max_bet >= self.MAX_PLAYER_BET:
            valid_actions[self.ActionType.RAISE.value] = 0

        return valid_actions

    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for a single player matching PlayerObservation type.

        Args:
            player_num: Seat index (0-5)

        Returns:
            PlayerObservation dict with all required fields
        """
        from poker_types import NUM_SEATS, NUM_BOARDS, BET_CAP, BOARD_CARDS_PER_BOARD

        # Community cards visibility based on street
        # Street 0 (card selection): cards hidden
        # Street 1+ (betting/showdown): cards revealed
        if self.street == 0:
            community_cards = [[-1] * BOARD_CARDS_PER_BOARD for _ in range(NUM_BOARDS)]
        else:
            # Reveal all 15 community cards across 3 boards
            community_cards = [
                self.community_cards[i*BOARD_CARDS_PER_BOARD:(i+1)*BOARD_CARDS_PER_BOARD]
                for i in range(NUM_BOARDS)
            ]

        # Calculate total pot
        pot_total = sum(self.bets)

        # Valid actions
        valid_actions = self._get_valid_actions(player_num)

        # Build observation matching PlayerObservation type
        obs = {
            'seat': player_num,
            'acting_seat': self.acting_agent,
            'street': self.street,
            'hole_cards': list(self.player_cards[player_num]),
            'community_cards': community_cards,
            'bets': list(self.bets),
            'my_stack': self.stacks[player_num],
            'all_stacks': list(self.stacks),
            'button_position': self.button_position,
            'pot_total': pot_total,
            'min_raise': self.min_raise,
            'max_raise': min(BET_CAP, self.MAX_PLAYER_BET - max(self.bets)),
            'valid_actions': valid_actions,
            'time_used': 0.0,  # TODO: Implement time tracking
            'time_left': 300.0,  # TODO: Implement time tracking
            'hand_number': self.current_hand,
        }

        # Add kept_cards if card selection phase is complete
        if self.kept_cards[player_num]:
            obs['kept_cards'] = list(self.kept_cards[player_num])

        # All-in situation (cap min_raise to max_raise)
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]

        return obs

    def _get_obs(self, winner, invalid_action=False):
        """
        Returns the observation for both players.
        """
        obs0, info0 = self._get_single_player_obs(0)
        obs1, info1 = self._get_single_player_obs(1)
        if winner == 0:
            reward = (min(self.bets), -min(self.bets))
        elif winner == 1:
            reward = (-min(self.bets), min(self.bets))
        else:
            reward = (0, 0)
        terminated = winner is not None
        truncated = False

        is_showdown = terminated and self.street > 3
        info = (
            {
                "player_0_cards": info0["player_cards"],
                "player_1_cards": info1["player_cards"],
                "community_cards": info0["community_cards"],
                "invalid_action": invalid_action,
            }
            if is_showdown
            else {}
        )

        return (obs0, obs1), reward, terminated, truncated, info

    def _draw_card(self):
        drawn_card = self.cards[0]
        self.cards = self.cards[1:]
        return drawn_card

    def reset(self, *, seed=None, options=None):
        """
        Resets the game for a new hand.

        For 6-player triple-board bomb-pot:
        - Deals 5 hole cards to each active player
        - Deals 15 community cards (3 boards × 5 cards)
        - Forces $1 bomb-pot ante from all players
        - Starts at street 0 (card selection phase)

        Args:
            seed: Random seed for reproducibility
            options: Dict with optional keys:
                - 'cards': List of card integers to rig the deck
                - 'button_seat': Override button position

        Returns:
            observations: Tuple of 6 PlayerObservation dicts (None for empty seats)
            info: Dict with game metadata
        """
        from poker_types import NUM_SEATS, NUM_BOARDS, HOLE_CARDS_PER_PLAYER, BOMB_POT_ANTE

        super().reset(seed=seed)

        # Button position (rotates each hand)
        self.button_position = self.current_hand % NUM_SEATS

        # Initialize game state
        self.street = 0  # 0 = card selection, 1 = betting, 2+ = showdown
        self.bets = [0] * NUM_SEATS
        self.stacks = [0] * NUM_SEATS  # Net gain/loss tracking (infinite bankroll)
        self.folded_players = set()  # Track who has folded
        self.kept_cards = [[] for _ in range(NUM_SEATS)]  # Track card selection

        # Set up deck
        DECK_SIZE = 52
        self.cards = np.arange(DECK_SIZE)
        np.random.shuffle(self.cards)

        # Override options if provided
        if options is not None:
            self.cards = options.get("cards", self.cards)
            self.button_position = options.get("button_seat", self.button_position)

        # Deal hole cards (5 per player, only to active players)
        self.player_cards = []
        for seat in range(NUM_SEATS):
            if seat < self.num_players:
                self.player_cards.append([self._draw_card() for _ in range(HOLE_CARDS_PER_PLAYER)])
            else:
                self.player_cards.append([])  # Empty for inactive seats

        # Deal community cards (15 total = 3 boards × 5 cards)
        self.community_cards = [self._draw_card() for _ in range(15)]

        # Apply bomb-pot ante for active players
        for seat in range(self.num_players):
            self.bets[seat] = BOMB_POT_ANTE
            self.stacks[seat] -= BOMB_POT_ANTE  # Deduct from stack

        # First action is card selection, starts left of button
        self.acting_agent = (self.button_position + 1) % NUM_SEATS
        self.min_raise = BOMB_POT_ANTE
        self.last_street_bet = BOMB_POT_ANTE

        # Generate observations for all seats
        observations = []
        for seat in range(NUM_SEATS):
            if seat < self.num_players:
                obs = self._get_single_player_obs(seat)
                observations.append(obs)
            else:
                observations.append(None)  # Null observation for empty seat

        info = {
            'hand_number': self.current_hand,
            'button_position': self.button_position,
        }

        # Increment hand counter for next reset (skip first reset in __init__)
        self._reset_count += 1
        if self._reset_count > 1:
            self.current_hand += 1

        return tuple(observations), info

    def _next_street(self):
        """
        Update to the next street of the game.
        """
        self.street += 1
        self.min_raise = self.big_blind_amount
        assert self.bets[0] == self.bets[1], self.logger.log(f"Bet amounts are not equal: {self.bets}")
        self.last_street_bet = self.bets[0]
        self.acting_agent = self.small_blind_player

    def _get_winner(self):
        """
        Returns the winner of the game.
        """
        board_cards = list(map(self.int_to_card, self.community_cards))
        player_0_cards = list(map(self.int_to_card, [c for c in self.player_cards[0] if c != -1]))
        player_1_cards = list(map(self.int_to_card, [c for c in self.player_cards[1] if c != -1]))
        assert len(player_0_cards) == 2 and len(player_1_cards) == 2 and len(board_cards) == 5
        player_0_hand_rank = self.evaluator.evaluate(
            player_0_cards,
            board_cards,
        )
        player_1_hand_rank = self.evaluator.evaluate(
            player_1_cards,
            board_cards,
        )

        self.logger.debug(f"(get winner) Player 0 cards: {list(map(Card.int_to_str, player_0_cards))}; Player 1 cards: {list(map(Card.int_to_str, player_1_cards))}")
        self.logger.debug(f"Determined winner based on hand scores; p0 score: {player_0_hand_rank}; p1 score: {player_1_hand_rank}")

        # showdown
        if player_0_hand_rank == player_1_hand_rank:
            winner = -1  # tie
        elif player_1_hand_rank < player_0_hand_rank:
            winner = 1
        else:
            winner = 0
        return winner

    def step(self, action: tuple[int, int, int]):
        """
        Takes a step in the game, given the action taken by the active player.

        `action`: (action_type, raise_amount, card_to_discard)
            - `action_type`: `int`, index of the action type
            - `raise_amount`: `int`, how much to raise, or 0 for a check or call
            - `card_to_discard`: `int`, index of the card which you would like to discard (0, or 1) or -1
        """
        action_type, raise_amount, card_to_discard = action
        valid_actions = self._get_valid_actions(self.acting_agent)
        self.logger.debug(f"Action type: {action_type}, Valid actions: {valid_actions}, Street: {self.street}, Bets: {self.bets}")

        # Handle invalid actions
        if not valid_actions[action_type]:
            action_name = self.ActionType(action_type).name
            valid_action_names = [self.ActionType(i).name for i, is_valid in enumerate(valid_actions) if is_valid]
            self.logger.error(f"Player {self.acting_agent} attempted invalid action: {action_name}. Valid actions are: {valid_action_names}")
            action_type = self.ActionType.INVALID.value

        if action_type == self.ActionType.RAISE.value and not (self.min_raise <= raise_amount <= (self.MAX_PLAYER_BET - max(self.bets))):
            self.logger.error(f"Player {self.acting_agent} attempted invalid raise amount: {raise_amount}. Must be between {self.min_raise} and {self.MAX_PLAYER_BET - max(self.bets)}")
            action_type = self.ActionType.INVALID.value

        winner = None

        new_street = False
        if action_type in (self.ActionType.FOLD.value, self.ActionType.INVALID.value):
            # We consider invalid actions as a fold
            self.logger.debug(f"Player {self.acting_agent} Folded")
            winner = 1 - self.acting_agent
        elif action_type == self.ActionType.CALL.value:
            self.bets[self.acting_agent] = self.bets[1 - self.acting_agent]
            if not (self.street == 0 and self.acting_agent == self.small_blind_player and self.bets[self.acting_agent] == self.big_blind_amount):
                # on the first street, the little blind can "call" the big blind's bet of 2
                new_street = True
        elif action_type == self.ActionType.CHECK.value:
            if self.acting_agent == self.big_blind_player:
                new_street = True  # big blind checks mean next street
        elif action_type == self.ActionType.RAISE.value:
            assert (
                self.bets[1 - self.acting_agent] >= self.bets[self.acting_agent]
            ), "Expected the opponent to have bet at least as much as current player given current player is raising"
            self.bets[self.acting_agent] = self.bets[1 - self.acting_agent] + raise_amount
            raise_so_far = self.bets[1 - self.acting_agent] - self.last_street_bet

            max_raise = self.MAX_PLAYER_BET - max(self.bets)
            min_raise_no_limit = raise_so_far + raise_amount
            self.min_raise = min(min_raise_no_limit, max_raise)
        else:
            # Must be DISCARD at this point
            assert action_type == self.ActionType.DISCARD.value, f"Unexpected action type: {action_type}"
            if card_to_discard != -1:
                self.discarded_cards[self.acting_agent] = self.player_cards[self.acting_agent][card_to_discard]
                drawn_card = self._draw_card()
                self.drawn_cards[self.acting_agent] = drawn_card
                self.player_cards[self.acting_agent][card_to_discard] = drawn_card

        if new_street:
            self._next_street()
            if self.street > 3:
                winner = self._get_winner()

        if not new_street and action_type != self.ActionType.DISCARD.value:
            self.acting_agent = 1 - self.acting_agent

        self.min_raise = min(self.min_raise, self.MAX_PLAYER_BET - max(self.bets))
        obs, reward, terminated, truncated, info = self._get_obs(winner, action_type == self.ActionType.INVALID.value)
        if terminated:
            self.logger.debug(
                f"Game is terminated. P0 cards: {list(map(self.int_card_to_str, self.player_cards[0]))}; P1 cards: {list(map(self.int_card_to_str, self.player_cards[1]))}; board cards: {list(map(self.int_card_to_str, self.community_cards))}"
            )

            if winner == 0:
                reward = (min(self.bets), -min(self.bets))
            elif winner == 1:
                reward = (-min(self.bets), min(self.bets))
            else:
                # tie
                reward = (0, 0)

            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
