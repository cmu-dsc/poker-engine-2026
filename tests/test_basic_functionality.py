"""
Basic functionality tests for gym_env_v2.py

Tests environment initialization, reset, and basic game flow.
"""

import pytest
import numpy as np


class TestEnvironmentInitialization:
    """Test environment creation and initialization"""

    def test_env_creation(self, env):
        """Test that environment can be created"""
        assert env is not None
        assert env.NUM_PLAYERS == 6
        assert env.ANTE_AMOUNT == 1
        assert env.MAX_PLAYER_BET == 100

    def test_env_reset(self, env):
        """Test that environment can be reset"""
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert len(obs) == 6, "Should have 6 player observations"
        assert info == {}

    def test_env_reset_deterministic(self, env):
        """Test that reset with same seed gives same initial state"""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        # Check that first player's cards are the same
        assert obs1[0]['my_cards'] == obs2[0]['my_cards']
        assert obs1[0]['community_cards'] == obs2[0]['community_cards']

    def test_action_space(self, env):
        """Test action space definition"""
        assert env.action_space is not None
        # Action space is Tuple of (action_type, raise_amount, card_idx_1, card_idx_2)
        assert len(env.action_space.spaces) == 4

    def test_observation_space(self, env):
        """Test observation space definition"""
        assert env.observation_space is not None
        # Should be tuple of 6 player observations
        assert len(env.observation_space.spaces) == 6


class TestCardSelection:
    """Test the card selection phase"""

    def test_selection_phase_starts(self, env):
        """Test that game starts in selection phase"""
        obs, info = env.reset(seed=42)
        assert env.selection_phase == True
        assert env.street == 1  # Bomb pot starts at flop

    def test_valid_card_selection(self, env, action_helper):
        """Test that players can select cards"""
        obs, info = env.reset(seed=42)

        # First player selects cards
        acting_player = env.acting_agent
        action = action_helper.select_cards(0, 1)
        obs, rewards, terminated, truncated, info = env.step(action)

        assert not terminated
        assert env.has_fixed[acting_player] == True

    def test_invalid_card_selection_same_index(self, env):
        """Test that selecting same card twice defaults to (0,1)"""
        obs, info = env.reset(seed=42)

        # Acting player is 1 (left of dealer)
        acting_player = env.acting_agent

        # Try to select same card twice (invalid)
        action = (4, 0, 2, 2)  # Same index
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should default to cards 0,1 and log error
        assert env.chosen_pairs[acting_player] == (0, 1)

    def test_invalid_card_selection_out_of_range(self, env):
        """Test that out of range indices default to (0,1)"""
        obs, info = env.reset(seed=42)

        # Acting player is 1 (left of dealer)
        acting_player = env.acting_agent

        # Try to select out of range indices
        action = (4, 0, 0, 10)  # 10 is out of range [0-4]
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should default to cards 0,1
        assert env.chosen_pairs[acting_player] == (0, 1)

    def test_selection_phase_completes(self, env, complete_selection_phase):
        """Test that selection phase completes after all players select"""
        obs, info = env.reset(seed=42)
        obs = complete_selection_phase(env)

        assert env.selection_phase == False
        # Should have moved to betting phase
        assert env.street == 1

    def test_cards_masked_after_selection(self, env, complete_selection_phase):
        """Test that unchosen cards are masked to -1"""
        obs, info = env.reset(seed=42)
        obs = complete_selection_phase(env)

        # Check that each player has exactly 2 non-(-1) cards
        for p in range(6):
            non_masked = [c for c in env.player_cards[p] if c != -1]
            assert len(non_masked) == 2, f"Player {p} should have exactly 2 cards"


class TestBasicGameFlow:
    """Test basic game progression"""

    def test_check_around_advances_street(self, env, complete_selection_phase, action_helper):
        """Test that checking around advances to next street"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_street = env.street
        assert initial_street == 1  # Flop

        # All 6 players check
        for _ in range(6):
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        # Should advance to turn
        assert env.street == 2

    def test_game_reaches_showdown(self, env, complete_selection_phase, action_helper):
        """Test that game can reach showdown"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        terminated = False
        rounds = 0
        max_rounds = 50

        while not terminated and rounds < max_rounds:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)
            rounds += 1

        assert terminated, "Game should terminate at showdown"
        assert env.street == 4, "Should be past river (street 3)"

    def test_winner_determined_at_showdown(self, env, complete_selection_phase, action_helper):
        """Test that winner is determined at showdown"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Check through to showdown
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        # Check winner info
        assert 'winner' in info
        winner = info['winner']
        assert (isinstance(winner, int) and 0 <= winner < 6) or isinstance(winner, list)

    def test_rewards_sum_to_zero(self, env, complete_selection_phase, action_helper):
        """Test that rewards sum to zero (zero-sum game)"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Play to showdown
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        # Rewards should sum to 0 in zero-sum game
        assert sum(rewards) == 0, f"Rewards {rewards} should sum to 0"


class TestActionRotation:
    """Test that acting agent rotates correctly"""

    def test_acting_agent_rotates(self, env, complete_selection_phase, action_helper):
        """Test that acting agent advances to next player"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        first_agent = env.acting_agent
        action = action_helper.check()
        obs, rewards, terminated, truncated, info = env.step(action)

        second_agent = env.acting_agent
        # Should advance to next player (wrapping at 6)
        assert second_agent == (first_agent + 1) % 6

    def test_acting_agent_skips_folded_players(self, env, complete_selection_phase, action_helper):
        """Test that acting agent skips folded players"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Player 1 folds
        if env.acting_agent == 1:
            pass  # Already acting
        else:
            # Get to player 1
            while env.acting_agent != 1:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)
                if terminated:
                    pytest.skip("Game ended before reaching player 1")

        # Player 1 folds
        action = action_helper.fold()
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            # Acting agent should skip player 1
            assert env.acting_agent != 1
            assert env.players_active[1] == False
