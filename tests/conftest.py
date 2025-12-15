"""
Pytest fixtures and configuration for gym_env_v2 tests
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.gym_env_v2 import PokerEnv


@pytest.fixture
def env():
    """Create a fresh PokerEnv instance for each test"""
    return PokerEnv()


@pytest.fixture
def env_with_seed():
    """Create a PokerEnv instance with a fixed seed for deterministic tests"""
    env = PokerEnv()
    env.reset(seed=42)
    return env


@pytest.fixture
def rigged_deck():
    """
    Factory fixture to create rigged decks for deterministic testing.

    Usage:
        def test_something(rigged_deck):
            deck = rigged_deck(['Ah', '2d', '3c', ...])
            env.reset(options={'cards': deck})
    """
    def _make_rigged_deck(card_strings):
        """
        Convert card strings like 'Ah', '2d' to integer encoding.

        Args:
            card_strings: List of card strings (e.g., ['Ah', '2d', '3h'])

        Returns:
            List of card integers
        """
        RANKS = "23456789TJQKA"
        SUITS = "cdhs"

        def card_str_to_int(card_str):
            rank, suit = card_str[0], card_str[1]
            return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)

        return [card_str_to_int(c) for c in card_strings]

    return _make_rigged_deck


@pytest.fixture
def complete_selection_phase():
    """
    Factory fixture to complete the card selection phase with default selections.

    Usage:
        def test_something(env, complete_selection_phase):
            obs, info = env.reset(seed=42)
            obs = complete_selection_phase(env)
            # Now in betting phase
    """
    def _complete_selection(env_instance):
        """
        Complete card selection phase by having all players select cards 0 and 1.

        Returns:
            Final observation tuple after selection phase completes
        """
        obs = None
        for _ in range(6):
            if env_instance.selection_phase:
                action = (4, 0, 0, 1)  # SELECT_CARDS, cards 0 and 1
                obs, rewards, terminated, truncated, info = env_instance.step(action)
        return obs

    return _complete_selection


@pytest.fixture
def action_helper():
    """
    Helper fixture to create valid action tuples.

    Usage:
        def test_something(action_helper):
            action = action_helper.fold()
            action = action_helper.raise_bet(10)
    """
    class ActionHelper:
        @staticmethod
        def fold():
            """FOLD action"""
            return (0, 0, 0, 0)

        @staticmethod
        def raise_bet(amount):
            """RAISE action with specified amount"""
            return (1, amount, 0, 0)

        @staticmethod
        def check():
            """CHECK action"""
            return (2, 0, 0, 0)

        @staticmethod
        def call():
            """CALL action"""
            return (3, 0, 0, 0)

        @staticmethod
        def select_cards(idx1, idx2):
            """SELECT_CARDS action"""
            return (4, 0, idx1, idx2)

    return ActionHelper()
