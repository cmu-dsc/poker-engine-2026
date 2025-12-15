"""
Edge case tests for gym_env_v2.py

Tests unusual scenarios, invalid actions, and boundary conditions.
"""

import pytest


class TestInvalidActions:
    """Test invalid action handling"""

    def test_invalid_action_type_during_betting(self, env, complete_selection_phase):
        """Test that invalid action type is treated as fold"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        acting_player = env.acting_agent

        # Try to use SELECT_CARDS during betting phase (invalid)
        action = (4, 0, 0, 1)  # SELECT_CARDS
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should be treated as fold
        assert env.players_active[acting_player] == False

    def test_invalid_action_during_selection_phase(self, env):
        """Test that non-SELECT_CARDS action during selection defaults cards"""
        obs, info = env.reset(seed=42)

        # Acting player is 1 (left of dealer)
        acting_player = env.acting_agent

        # Try to FOLD during selection phase
        action = (0, 0, 0, 0)  # FOLD
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should default to selecting cards 0,1
        assert env.chosen_pairs[acting_player] == (0, 1)
        assert env.has_fixed[acting_player] == True

    def test_check_when_bet_is_higher_treated_as_fold(self, env, complete_selection_phase, action_helper):
        """Test that checking when facing a bet is treated as fold"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Player raises
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            acting_player = env.acting_agent

            # Try to check (invalid - need to call or fold)
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

            # Should be treated as fold
            assert env.players_active[acting_player] == False

    def test_call_when_bets_equal_treated_as_fold(self, env, complete_selection_phase, action_helper):
        """Test that calling when bets are equal is treated as fold"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        acting_player = env.acting_agent

        # Try to CALL when bets are equal (invalid)
        action = action_helper.call()
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should be treated as fold
        assert env.players_active[acting_player] == False


class TestAllInSituations:
    """Test all-in and max bet scenarios"""

    def test_cannot_raise_at_max_bet(self, env, complete_selection_phase, action_helper):
        """Test that raise is invalid when at MAX_PLAYER_BET"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Someone raises to near max
        action = action_helper.raise_bet(99)  # Raises to 100
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            # RAISE should now be invalid
            valid_actions = env._get_valid_actions(env.acting_agent)
            assert valid_actions[1] == 0  # RAISE is invalid

    def test_min_raise_capped_at_max_raise(self, env, complete_selection_phase):
        """Test that min_raise is capped when max_raise is smaller"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Get observation for acting player
        current_obs = obs[env.acting_agent]

        # In all-in situations, min_raise should not exceed max_raise
        if current_obs['min_raise'] > current_obs['max_raise']:
            pytest.fail("min_raise should never exceed max_raise")

        # This is enforced in _get_single_player_obs
        assert current_obs['min_raise'] <= current_obs['max_raise']


class TestZeroPlayers:
    """Test scenarios with unusual player counts"""

    def test_all_players_fold_simultaneously_not_possible(self, env, complete_selection_phase):
        """Test that at least one player must remain active"""
        # This is more of a sanity check - the game logic prevents this

        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Players fold one by one
        folds = 0
        while not env.selection_phase and sum(env.players_active) > 1:
            action = (0, 0, 0, 0)  # FOLD
            obs, rewards, terminated, truncated, info = env.step(action)
            folds += 1

            if terminated:
                break

        # Should have exactly 1 player remaining when game ends
        assert sum(env.players_active) >= 1


class TestDeterministicBehavior:
    """Test deterministic behavior with same seeds"""

    def test_same_seed_same_game(self, env):
        """Test that same seed produces same game"""
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=100)

        # Same initial observations
        assert obs1[0]['my_cards'] == obs2[0]['my_cards']
        assert obs1[0]['community_cards'] == obs2[0]['community_cards']

    def test_different_seed_different_game(self, env):
        """Test that different seeds produce different games"""
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=200)

        # Different initial observations (very likely)
        assert obs1[0]['my_cards'] != obs2[0]['my_cards']


class TestGameStateConsistency:
    """Test that game state remains consistent"""

    def test_num_active_players_decreases_on_fold(self, env, complete_selection_phase, action_helper):
        """Test that active player count decreases on fold"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_active = sum(env.players_active)

        # Player folds
        action = action_helper.fold()
        obs, rewards, terminated, truncated, info = env.step(action)

        new_active = sum(env.players_active)

        if not terminated:
            assert new_active == initial_active - 1
        else:
            # Game ended
            assert new_active >= 1

    def test_pot_increases_with_raises(self, env, complete_selection_phase, action_helper):
        """Test that pot size increases with raises"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_pot = sum(env.bets)

        # Player raises
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        new_pot = sum(env.bets)

        assert new_pot > initial_pot

    def test_bets_never_decrease(self, env, complete_selection_phase, action_helper):
        """Test that individual bets never decrease during a hand"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Track bets through several actions
        previous_bets = env.bets.copy()

        for _ in range(10):
            if not env.selection_phase:
                # Take valid action
                valid_actions = env._get_valid_actions(env.acting_agent)

                if valid_actions[2]:  # CHECK
                    action = action_helper.check()
                elif valid_actions[3]:  # CALL
                    action = action_helper.call()
                else:
                    action = action_helper.fold()

                obs, rewards, terminated, truncated, info = env.step(action)

                # Each player's bet should never decrease
                for p in range(6):
                    if env.players_active[p]:
                        assert env.bets[p] >= previous_bets[p]

                previous_bets = env.bets.copy()

                if terminated:
                    break


class TestCommunityCardVisibility:
    """Test that community cards are revealed progressively"""

    def test_flop_shows_3_cards(self, env, complete_selection_phase):
        """Test that flop shows exactly 3 community cards"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # At flop (street 1)
        assert env.street == 1

        # Check observation
        current_obs = obs[0]
        community_cards = current_obs['community_cards']

        # Count non-hidden cards (encoded as > 0)
        visible = [c for c in community_cards if c > 0]
        assert len(visible) == 3

    def test_turn_shows_4_cards(self, env, complete_selection_phase, action_helper):
        """Test that turn shows exactly 4 community cards"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Check to turn
        for _ in range(6):
            if env.street == 1:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        # At turn (street 2)
        assert env.street == 2

        # Check observation
        current_obs = obs[0]
        community_cards = current_obs['community_cards']

        # Count visible cards
        visible = [c for c in community_cards if c > 0]
        assert len(visible) == 4

    def test_river_shows_5_cards(self, env, complete_selection_phase, action_helper):
        """Test that river shows all 5 community cards"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Check to river
        for _ in range(12):  # 6 on flop, 6 on turn
            if env.street < 3:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        # At river (street 3)
        assert env.street == 3

        # Check observation
        current_obs = obs[0]
        community_cards = current_obs['community_cards']

        # Count visible cards
        visible = [c for c in community_cards if c > 0]
        assert len(visible) == 5


class TestRewardCalculation:
    """Test reward calculation correctness"""

    def test_rewards_sum_to_zero_simple(self, env, complete_selection_phase, action_helper):
        """Test zero-sum property with simple scenario"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # All check to showdown
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        assert sum(rewards) == 0

    def test_rewards_sum_to_zero_with_raises(self, env, complete_selection_phase, action_helper):
        """Test zero-sum property with raises"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # First player raises
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Others call or fold
        for _ in range(5):
            if not terminated:
                action = action_helper.call() if env._get_valid_actions(env.acting_agent)[3] else action_helper.fold()
                obs, rewards, terminated, truncated, info = env.step(action)

        # Play to showdown if needed
        while not terminated:
            action = action_helper.check() if env._get_valid_actions(env.acting_agent)[2] else action_helper.call()
            obs, rewards, terminated, truncated, info = env.step(action)

        assert sum(rewards) == 0
