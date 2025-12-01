"""
CMU Poker Bot Competition Game Engine 2026

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

class PokerEnv(gym.Env):
    NUM_PLAYERS = 6
    ANTE_AMOUNT = 1
    MAX_PLAYER_BET = 100
    HAND_SIZE = 5  # Each player gets 5 cards initially

    RANKS = "23456789TJQKA"  # 13 ranks
    SUITS = "cdhs"  # 4 suits = 52 card deck

    @staticmethod
    def int_to_card(card_int: int):
        """
        Convert from our encoding of a card, an integer on [0, 52)
        to the trey's encoding of a card, an integer designed for fast lookup & comparison
        """
        if card_int == -1:
            raise ValueError("Cannot convert -1 to card")
        return Card.new(PokerEnv.int_card_to_str(card_int))

    @staticmethod
    def int_card_to_str(card_int: int):
        """Convert card int (0-51) to string like '2c', 'Ah', etc."""
        if card_int == -1:
            return "??"
        RANKS, SUITS = PokerEnv.RANKS, PokerEnv.SUITS
        rank = RANKS[card_int % len(RANKS)]
        suit = SUITS[card_int // len(RANKS)]
        return rank + suit

    @staticmethod
    def card_to_obs_encoding(card_int: int):
        """
        Convert card int to observation space encoding.
        -1 (hidden) -> 0
        0-51 (cards) -> 1-52
        """
        return card_int + 1

    @staticmethod
    def obs_encoding_to_card(obs_val: int):
        """
        Convert observation space encoding to card int.
        0 -> -1 (hidden)
        1-52 -> 0-51 (cards)
        """
        return obs_val - 1

    class ActionType(Enum):
        FOLD = 0
        RAISE = 1
        CHECK = 2
        CALL = 3
        SELECT_CARDS = 4  # Select 2 cards from 5 to keep
        INVALID = 5

    def __init__(self, logger=None, ante_amount=1, num_hands=1):
        """
        Represents a single hand of 6-player Bomb Pot PLO.
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.ante_amount = ante_amount
        self.min_raise = ante_amount
        self.acting_agent = 0
        self.last_street_bet = None
        self.evaluator = Evaluator()
        self.dealer_button = 0  # Tracks dealer position

        # Action space: (action_type, raise_amount, card_idx_1, card_idx_2)
        # - action_type: FOLD, RAISE, CHECK, CALL, SELECT_CARDS
        # - raise_amount: amount to raise (only used for RAISE action)
        # - card_idx_1, card_idx_2: indices of 2 cards to keep (only used for SELECT_CARDS)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(len(self.ActionType) - 1),  # user can pass any besides INVALID
                spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET (raise amounts 1 to MAX_PLAYER_BET)
                spaces.Discrete(self.HAND_SIZE),  # card index 1 (0-4)
                spaces.Discrete(self.HAND_SIZE),  # card index 2 (0-4)
            ]
        )

        # Card space: Discrete(53) represents -1 to 51
        # We'll map: 0 -> -1 (hidden), 1 -> 0 (first card), ..., 52 -> 51 (last card)
        cards_space = spaces.Discrete(53)  # 0 represents -1, 1-52 represent cards 0-51

        # Observation space for one player
        # Each player sees their own cards, community cards, and aggregate info about opponents
        observation_space_one_player = spaces.Dict(
            {
                "street": spaces.Discrete(4),  # 0=preflop(unused), 1=flop, 2=turn, 3=river
                "acting_agent": spaces.Discrete(self.NUM_PLAYERS),
                "seat": spaces.Discrete(self.NUM_PLAYERS),  # This player's seat number
                "my_cards": spaces.Tuple([cards_space for _ in range(self.HAND_SIZE)]),
                "has_fixed": spaces.Discrete(2),  # 0=not fixed, 1=fixed
                "community_cards": spaces.Tuple([cards_space for _ in range(5)]),
                "my_bet": spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET
                "pot": spaces.Discrete(self.MAX_PLAYER_BET * self.NUM_PLAYERS + 1),  # 0 to MAX_PLAYER_BET * 6
                "opponent_bets": spaces.Tuple([spaces.Discrete(self.MAX_PLAYER_BET + 1) for _ in range(self.NUM_PLAYERS)]),
                "opponent_fixed": spaces.Tuple([spaces.Discrete(2) for _ in range(self.NUM_PLAYERS)]),
                "players_active": spaces.Tuple([spaces.Discrete(2) for _ in range(self.NUM_PLAYERS)]),
                "min_raise": spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET (1 to MAX_PLAYER_BET are valid)
                "max_raise": spaces.Discrete(self.MAX_PLAYER_BET + 1),  # 0 to MAX_PLAYER_BET
                "valid_actions": spaces.MultiBinary(len(self.ActionType) - 1),
            }
        )

        # Observation space is a tuple of 6 player observations
        self.observation_space = spaces.Tuple([observation_space_one_player for _ in range(self.NUM_PLAYERS)])

        # New episode
        import random
        self.reset(seed=random.randint(0, 2**32 - 1))

    def _get_valid_actions(self, player_num: int):
        """
        Returns a list of valid actions for the given player.
        [FOLD, RAISE, CHECK, CALL, SELECT_CARDS]
        """
        # Card selection phase - only SELECT_CARDS is valid
        if self.selection_phase:
            valid_actions = [0] * (len(self.ActionType) - 1)
            if player_num == self.acting_agent and not self.has_fixed[player_num]:
                valid_actions[self.ActionType.SELECT_CARDS.value] = 1
            return valid_actions

        # Not active anymore (folded), no valid actions
        if not self.players_active[player_num]:
            return [0] * (len(self.ActionType) - 1)

        # Betting phase
        valid_actions = [1, 1, 1, 1, 0]  # FOLD, RAISE, CHECK, CALL always considered, SELECT_CARDS disabled

        max_bet = max(self.bets)
        my_bet = self.bets[player_num]

        # Can't CHECK if someone has a larger bet than you
        if my_bet < max_bet:
            valid_actions[self.ActionType.CHECK.value] = 0

        # Can't CALL if you already have the max bet
        if my_bet == max_bet:
            valid_actions[self.ActionType.CALL.value] = 0

        # Can't RAISE if already at max bet limit
        if max_bet >= self.MAX_PLAYER_BET:
            valid_actions[self.ActionType.RAISE.value] = 0

        return valid_actions
    
    def _handle_hand_selection(self, action: tuple[int, int, int, int]):
        """
        Handle the card fixing phase where players select 2 cards from 5 to keep.
        This happens on the flop (street 1) before betting begins.
        """
        action_type, _, card_idx_1, card_idx_2 = action
        p = self.acting_agent

        # Validate action type
        if action_type != self.ActionType.SELECT_CARDS.value:
            self.logger.error(f"Player {p} must select cards, chose action {action_type}; defaulting to (0,1)")
            card_idx_1, card_idx_2 = 0, 1

        # Validate card indices
        if not (0 <= card_idx_1 < self.HAND_SIZE and 0 <= card_idx_2 < self.HAND_SIZE and card_idx_1 != card_idx_2):
            self.logger.error(f"Player {p} invalid selection {card_idx_1},{card_idx_2}; defaulting to (0,1)")
            card_idx_1, card_idx_2 = 0, 1

        # Record chosen pair
        self.chosen_pairs[p] = (card_idx_1, card_idx_2)
        self.has_fixed[p] = True

        # Mask the 3 unchosen cards to -1 (keep only the 2 selected)
        keep = {card_idx_1, card_idx_2}
        for k in range(self.HAND_SIZE):
            if k not in keep:
                self.player_cards[p][k] = -1

        # Check if all players have fixed their cards
        if all(self.has_fixed):
            self.selection_phase = False
            # Start betting from left of dealer button
            self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS
            # Skip folded players (shouldn't happen during selection, but just in case)
            while not self.players_active[self.acting_agent]:
                self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS
        else:
            # Move to next player who hasn't fixed yet
            self.acting_agent = (p + 1) % self.NUM_PLAYERS
            while self.has_fixed[self.acting_agent]:
                self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS

        # Return observation (no winner yet)
        return self._get_obs(winner=None, invalid_action=False)


    def _get_single_player_obs(self, player_num: int):
        """
        Returns the observation for a specific player.
        """
        # In bomb pot, all 5 community cards are visible from street 1 (flop)
        if self.street == 0:
            num_cards_to_reveal = 0  # Shouldn't happen in bomb pot
        else:
            # Street 1 (flop) = 3 cards, Street 2 (turn) = 4 cards, Street 3 (river) = 5 cards
            num_cards_to_reveal = self.street + 2

        # Calculate pot
        pot = sum(self.bets)

        # Encode cards for observation space (0 = hidden, 1-52 = cards 0-51)
        my_cards_encoded = tuple(self.card_to_obs_encoding(c) for c in self.player_cards[player_num])
        community_cards_raw = self.community_cards[:num_cards_to_reveal] + [-1] * (5 - num_cards_to_reveal)
        community_cards_encoded = tuple(self.card_to_obs_encoding(c) for c in community_cards_raw)

        obs = {
            "street": self.street,
            "acting_agent": self.acting_agent,
            "seat": player_num,
            "my_cards": my_cards_encoded,
            "has_fixed": int(self.has_fixed[player_num]),
            "community_cards": community_cards_encoded,
            "my_bet": self.bets[player_num],
            "pot": pot,
            "opponent_bets": tuple(self.bets),
            "opponent_fixed": tuple([int(f) for f in self.has_fixed]),
            "players_active": tuple([int(a) for a in self.players_active]),
            "min_raise": self.min_raise,
            "max_raise": self.MAX_PLAYER_BET - max(self.bets),
            "valid_actions": self._get_valid_actions(player_num),
        }

        # All-in situation: cap min_raise at max_raise
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]

        # Info uses raw card values for human readability
        info = {
            "player_cards": [self.int_card_to_str(card) for card in self.player_cards[player_num] if card != -1],
            "community_cards": [self.int_card_to_str(card) for card in self.community_cards[:num_cards_to_reveal] if card != -1],
        }
        return obs, info

    def _get_obs(self, winner, invalid_action=False):
        """
        Returns observations for all 6 players, rewards, terminated status, and info.

        winner can be:
        - None: game continues
        - int (0-5): single winner
        - list of ints: multiple winners (tie)
        """
        # Get observations for all players
        obs_list = []
        info_list = []
        for p in range(self.NUM_PLAYERS):
            obs, info = self._get_single_player_obs(p)
            obs_list.append(obs)
            info_list.append(info)

        # Calculate rewards
        pot = sum(self.bets)
        rewards = [0] * self.NUM_PLAYERS

        if winner is not None:
            # Handle ties (winner is a list)
            if isinstance(winner, list):
                # Split pot among winners
                pot_share = pot / len(winner)
                for p in range(self.NUM_PLAYERS):
                    if p in winner:
                        rewards[p] = pot_share - self.bets[p]
                    else:
                        rewards[p] = -self.bets[p]
            else:
                # Single winner
                for p in range(self.NUM_PLAYERS):
                    if p == winner:
                        rewards[p] = pot - self.bets[p]
                    else:
                        rewards[p] = -self.bets[p]

        terminated = winner is not None
        truncated = False

        # Info dict for showdown
        is_showdown = terminated and self.street >= 1
        info = {}
        if is_showdown:
            for p in range(self.NUM_PLAYERS):
                info[f"player_{p}_cards"] = info_list[p]["player_cards"]
            info["community_cards"] = info_list[0]["community_cards"]
            info["pot"] = pot
            info["winner"] = winner
            info["invalid_action"] = invalid_action

        return tuple(obs_list), tuple(rewards), terminated, truncated, info

    def _draw_card(self):
        return self.cards.pop()

    def reset(self, *, seed=None, options=None):
        """
        Resets the game for a new hand of 6-player Bomb Pot PLO.

        Bomb Pot: All players post equal antes, no pre-flop betting.
        Game starts at street 1 (flop) with all 5 community cards visible.
        Players then select 2 cards from their 5 hole cards to keep.

        options is a dict with the following keys:
        - cards: a list of 52 cards to be used in the game (optional)
        - dealer_button: which player is the dealer (optional, default=0)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        DECK_SIZE = len(self.RANKS) * len(self.SUITS)  # 52 cards

        # Initialize game state variables
        self.street = 1  # Start at flop (skip pre-flop in bomb pot)
        self.selection_phase = True  # Players must fix cards before betting
        self.chosen_pairs = [None] * self.NUM_PLAYERS
        self.has_fixed = [False] * self.NUM_PLAYERS
        self.players_active = [True] * self.NUM_PLAYERS
        self.bets = [self.ANTE_AMOUNT] * self.NUM_PLAYERS  # Everyone posts ante
        self.min_raise = self.ANTE_AMOUNT
        self.last_street_bet = self.ANTE_AMOUNT

        # Initialize deck
        self.cards = list(np.arange(DECK_SIZE))
        np.random.shuffle(self.cards)

        # Override with any provided options
        if options is not None:
            if "cards" in options:
                self.cards = list(options["cards"])
            if "dealer_button" in options:
                self.dealer_button = options["dealer_button"]

        # Deal 5 cards to each player
        self.player_cards = [[self._draw_card() for _ in range(self.HAND_SIZE)] for _ in range(self.NUM_PLAYERS)]

        # Deal 5 community cards (all visible on flop in bomb pot)
        self.community_cards = [self._draw_card() for _ in range(5)]

        # First player to act (left of dealer button)
        self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS

        # Generate observations for all players
        obs_list_raw = [self._get_single_player_obs(p) for p in range(self.NUM_PLAYERS)]
        obs_list_filtered = tuple(obs_item for (obs_item, _) in obs_list_raw)
        info = {}

        return obs_list_filtered, info

    def _next_street(self):
        """
        Advance to the next street of the game.
        Streets: 1 (flop) -> 2 (turn) -> 3 (river) -> showdown
        """
        self.street += 1
        self.min_raise = self.ANTE_AMOUNT

        # Verify all active players have equal bets
        active_bets = [self.bets[p] for p in range(self.NUM_PLAYERS) if self.players_active[p]]
        if active_bets:
            max_bet = max(active_bets)
            for bet in active_bets:
                assert bet == max_bet, f"Bet amounts not equal when advancing street: {self.bets}"

        self.last_street_bet = max(self.bets)

        # First player to act is left of dealer
        self.acting_agent = (self.dealer_button + 1) % self.NUM_PLAYERS
        # Skip inactive players
        while not self.players_active[self.acting_agent]:
            self.acting_agent = (self.acting_agent + 1) % self.NUM_PLAYERS

    def _evaluate_plo_hand(self, player_cards, board_cards):
        """
        Evaluate a PLO hand: must use exactly 2 from hand and 3 from board.

        Args:
            player_cards: List of 2 card ints (already filtered to non -1)
            board_cards: List of 5 card ints

        Returns:
            Best hand rank (lower is better in treys)
        """
        from itertools import combinations

        # Convert to treys card format
        hole_cards_treys = [self.int_to_card(c) for c in player_cards]
        board_cards_treys = [self.int_to_card(c) for c in board_cards]

        # In PLO with 2 hole cards, there's only 1 way to use both
        # But there are C(5,3) = 10 ways to choose 3 from board
        best_rank = float('inf')

        for board_combo in combinations(board_cards_treys, 3):
            # Evaluate 5-card hand: 2 hole + 3 board
            hand = hole_cards_treys + list(board_combo)
            rank = self.evaluator.evaluate(hand, [])
            if rank < best_rank:
                best_rank = rank

        return best_rank

    def _get_winner(self):
        """
        Returns the winner(s) of the game using PLO evaluation.

        Returns:
            - int (0-5): single winner
            - list of ints: multiple winners (tie)
        """
        board_cards = self.community_cards[:5]  # Use first 5 community cards

        # Evaluate each active player's hand
        player_ranks = {}
        for p in range(self.NUM_PLAYERS):
            if self.players_active[p]:
                # Get the 2 cards this player kept (non -1 cards)
                player_cards = [c for c in self.player_cards[p] if c != -1]
                assert len(player_cards) == 2, f"Player {p} should have exactly 2 cards, has {len(player_cards)}"

                # Evaluate PLO hand
                rank = self._evaluate_plo_hand(player_cards, board_cards)
                player_ranks[p] = rank

                self.logger.debug(f"Player {p} cards: {[self.int_card_to_str(c) for c in player_cards]}, rank: {rank}")

        # Find winner(s) - lowest rank wins
        if not player_ranks:
            return None

        best_rank = min(player_ranks.values())
        winners = [p for p, rank in player_ranks.items() if rank == best_rank]

        # Return single winner or list of winners for tie
        if len(winners) == 1:
            return winners[0]
        else:
            return winners

    def step(self, action: tuple[int, int, int, int]):
        """
        Takes a step in the game, given the action taken by the active player.

        `action`: (action_type, raise_amount, card_idx_1, card_idx_2)
            - `action_type`: int, index of the action type [FOLD, RAISE, CHECK, CALL, SELECT_CARDS]
            - `raise_amount`: int, how much to raise (only for RAISE action)
            - `card_idx_1`: int, first card index to keep (only for SELECT_CARDS)
            - `card_idx_2`: int, second card index to keep (only for SELECT_CARDS)
        """
        # Handle card selection phase
        if self.selection_phase:
            return self._handle_hand_selection(action)

        # Betting phase
        action_type, raise_amount, _, _ = action
        current_player = self.acting_agent
        valid_actions = self._get_valid_actions(current_player)

        self.logger.debug(f"Player {current_player} action: {action_type}, Valid: {valid_actions}, Street: {self.street}, Bets: {self.bets}")

        # Validate action
        if not valid_actions[action_type]:
            action_name = self.ActionType(action_type).name
            valid_action_names = [self.ActionType(i).name for i, is_valid in enumerate(valid_actions) if is_valid]
            self.logger.error(f"Player {current_player} invalid action: {action_name}. Valid: {valid_action_names}")
            action_type = self.ActionType.INVALID.value

        # Validate raise amount
        if action_type == self.ActionType.RAISE.value:
            max_raise = self.MAX_PLAYER_BET - max(self.bets)
            if not (self.min_raise <= raise_amount <= max_raise):
                self.logger.error(f"Player {current_player} invalid raise: {raise_amount}. Range: [{self.min_raise}, {max_raise}]")
                action_type = self.ActionType.INVALID.value

        winner = None

        # Process action
        if action_type in (self.ActionType.FOLD.value, self.ActionType.INVALID.value):
            # Fold or invalid action
            self.logger.debug(f"Player {current_player} folded")
            self.players_active[current_player] = False

            # Check if only one player remains
            active_players = [p for p in range(self.NUM_PLAYERS) if self.players_active[p]]
            if len(active_players) == 1:
                winner = active_players[0]

        elif action_type == self.ActionType.CALL.value:
            # Match the highest bet
            max_bet = max(self.bets)
            self.bets[current_player] = max_bet

        elif action_type == self.ActionType.CHECK.value:
            # No action, just pass
            pass

        elif action_type == self.ActionType.RAISE.value:
            # Raise
            max_bet = max(self.bets)
            self.bets[current_player] = max_bet + raise_amount

            # Update min_raise for future raises
            raise_amount_added = raise_amount
            max_raise_available = self.MAX_PLAYER_BET - max(self.bets)
            self.min_raise = min(raise_amount_added * 2, max_raise_available)

        # Advance to next player
        next_player = (current_player + 1) % self.NUM_PLAYERS
        # Skip inactive (folded) players
        while not self.players_active[next_player] and len([p for p in range(self.NUM_PLAYERS) if self.players_active[p]]) > 1:
            next_player = (next_player + 1) % self.NUM_PLAYERS

        # Check if betting round is complete
        betting_round_complete = self._is_betting_round_complete()

        if betting_round_complete and winner is None:
            self._next_street()
            if self.street > 3:
                # Game over, determine winner
                winner = self._get_winner()
        else:
            self.acting_agent = next_player

        # Cap min_raise
        self.min_raise = min(self.min_raise, self.MAX_PLAYER_BET - max(self.bets))

        # Get observations and rewards
        obs, reward, terminated, truncated, info = self._get_obs(winner, action_type == self.ActionType.INVALID.value)

        if terminated:
            self.logger.debug(f"Game terminated. Winner: {winner}, Pot: {sum(self.bets)}")
            for p in range(self.NUM_PLAYERS):
                cards_str = [self.int_card_to_str(c) for c in self.player_cards[p] if c != -1]
                self.logger.debug(f"Player {p} cards: {cards_str}")

        return obs, reward, terminated, truncated, info

    def _is_betting_round_complete(self):
        """
        Check if the current betting round is complete.
        A round is complete when all active players have equal bets and have had a chance to act.
        """
        active_players = [p for p in range(self.NUM_PLAYERS) if self.players_active[p]]

        # If only one player remains, round is complete
        if len(active_players) <= 1:
            return True

        # Check if all active players have equal bets
        active_bets = [self.bets[p] for p in active_players]
        max_bet = max(active_bets)

        # All must have equal bets
        for bet in active_bets:
            if bet != max_bet:
                return False

        # Check if we've cycled back to the first player who can act
        # Simple heuristic: if next player is (dealer_button + 1) and all bets equal, round complete
        next_player = (self.acting_agent + 1) % self.NUM_PLAYERS
        while not self.players_active[next_player] and next_player != self.acting_agent:
            next_player = (next_player + 1) % self.NUM_PLAYERS

        # If we've cycled back to the start position (left of dealer)
        first_to_act = (self.dealer_button + 1) % self.NUM_PLAYERS
        while not self.players_active[first_to_act]:
            first_to_act = (first_to_act + 1) % self.NUM_PLAYERS

        return next_player == first_to_act
