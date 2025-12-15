"""
Betting logic tests for gym_env_v2.py

Tests FOLD, RAISE, CHECK, CALL actions and betting round progression.
"""

import pytest


class TestFoldAction:
    """Test FOLD action"""

    def test_fold_marks_player_inactive(self, env, complete_selection_phase, action_helper):
        """Test that folding marks player as inactive"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        acting_player = env.acting_agent
        assert env.players_active[acting_player] == True

        # Player folds
        action = action_helper.fold()
        obs, rewards, terminated, truncated, info = env.step(action)

        assert env.players_active[acting_player] == False

    def test_fold_with_only_two_players_ends_game(self, env, complete_selection_phase, action_helper):
        """Test that when only 2 players remain, one fold ends game"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # 4 players fold
        for i in range(4):
            if not env.selection_phase:
                action = action_helper.fold()
                obs, rewards, terminated, truncated, info = env.step(action)

        # Should still be 2 active players, game continues
        active_count = sum(env.players_active)
        assert active_count == 2
        assert not terminated

        # One more fold ends it
        action = action_helper.fold()
        obs, rewards, terminated, truncated, info = env.step(action)

        assert terminated
        assert 'winner' in info

    def test_all_fold_except_one(self, env, complete_selection_phase, action_helper):
        """Test that game ends when all but one player folds"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # 5 players fold
        terminated = False
        for i in range(5):
            if not terminated:
                action = action_helper.fold()
                obs, rewards, terminated, truncated, info = env.step(action)

        assert terminated
        assert 'winner' in info
        # Exactly one player should be active
        assert sum(env.players_active) == 1


class TestRaiseAction:
    """Test RAISE action"""

    def test_raise_increases_bet(self, env, complete_selection_phase, action_helper):
        """Test that raise increases player's bet"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        acting_player = env.acting_agent
        initial_bet = env.bets[acting_player]

        # Raise by 5
        action = action_helper.raise_bet(5)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Bet should increase
        assert env.bets[acting_player] == initial_bet + 5

    def test_raise_updates_min_raise(self, env, complete_selection_phase, action_helper):
        """Test that raise updates the min_raise for next player"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_min_raise = env.min_raise

        # Player raises by 10
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Min raise should be updated
        assert env.min_raise >= initial_min_raise

    def test_raise_requires_other_players_to_act(self, env, complete_selection_phase, action_helper):
        """Test that a raise requires other players to act again"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_street = env.street

        # Player 1 raises
        action = action_helper.raise_bet(5)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Next 5 players check (should not advance street)
        for i in range(5):
            if not terminated and env.street == initial_street:
                # Can't check with a raise, need to call or fold
                action = action_helper.call()
                obs, rewards, terminated, truncated, info = env.step(action)

        # After all call, street should advance
        if not terminated:
            assert env.street > initial_street

    def test_invalid_raise_amount_too_small(self, env, complete_selection_phase, action_helper):
        """Test that raise amount below min_raise is invalid"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        min_raise = env.min_raise
        acting_player = env.acting_agent

        # Try to raise less than min_raise (should be treated as fold)
        action = action_helper.raise_bet(0)  # Invalid
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should be treated as fold
        if not terminated:
            assert env.players_active[acting_player] == False
        else:
            # Game ended by elimination
            assert True

    def test_invalid_raise_amount_too_large(self, env, complete_selection_phase, action_helper):
        """Test that raise amount above max is invalid"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        acting_player = env.acting_agent

        # Try to raise more than allowed (above MAX_PLAYER_BET)
        action = action_helper.raise_bet(200)  # Way over max
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should be treated as fold
        if not terminated:
            assert env.players_active[acting_player] == False


class TestCheckAction:
    """Test CHECK action"""

    def test_check_with_equal_bets(self, env, complete_selection_phase, action_helper):
        """Test that check works when bets are equal"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # All bets should be equal (antes)
        assert len(set(env.bets)) == 1

        # Check should be valid
        acting_player = env.acting_agent
        valid_actions = env._get_valid_actions(acting_player)
        assert valid_actions[2] == 1  # CHECK is valid

        action = action_helper.check()
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should succeed (not fold)
        assert not terminated or 'winner' in info

    def test_check_invalid_with_unequal_bets(self, env, complete_selection_phase, action_helper):
        """Test that check is invalid when someone has raised"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # First player raises
        action = action_helper.raise_bet(5)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            # Next player's bets are not equal, CHECK should be invalid
            acting_player = env.acting_agent
            valid_actions = env._get_valid_actions(acting_player)
            assert valid_actions[2] == 0  # CHECK is invalid

    def test_all_check_advances_street(self, env, complete_selection_phase, action_helper):
        """Test that all players checking advances street"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_street = env.street
        terminated = False

        # All 6 players check
        for _ in range(6):
            if not terminated:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        # Should advance to next street
        if not terminated:
            assert env.street == initial_street + 1


class TestCallAction:
    """Test CALL action"""

    def test_call_matches_max_bet(self, env, complete_selection_phase, action_helper):
        """Test that call matches the current max bet"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # First player raises
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        max_bet = max(env.bets)

        # Next player calls
        acting_player = env.acting_agent
        action = action_helper.call()
        obs, rewards, terminated, truncated, info = env.step(action)

        # Their bet should now equal max_bet
        assert env.bets[acting_player] == max_bet

    def test_call_invalid_with_equal_bets(self, env, complete_selection_phase, action_helper):
        """Test that call is invalid when bets are already equal"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # All bets equal (antes), CALL should be invalid
        acting_player = env.acting_agent
        valid_actions = env._get_valid_actions(acting_player)
        assert valid_actions[3] == 0  # CALL is invalid

    def test_call_after_raise(self, env, complete_selection_phase, action_helper):
        """Test calling after a raise"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Player raises
        action = action_helper.raise_bet(7)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            # Next player should be able to call
            valid_actions = env._get_valid_actions(env.acting_agent)
            assert valid_actions[3] == 1  # CALL is valid


class TestBettingRounds:
    """Test complete betting rounds"""

    def test_betting_round_completes_when_all_bets_equal(self, env, complete_selection_phase, action_helper):
        """Test that betting round completes when all active players have equal bets"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_street = env.street

        # Player raises
        action = action_helper.raise_bet(5)
        obs, rewards, terminated, truncated, info = env.step(action)

        # All others call
        for _ in range(5):
            if not terminated and env.street == initial_street:
                action = action_helper.call()
                obs, rewards, terminated, truncated, info = env.step(action)

        # Street should advance after all call
        if not terminated:
            assert env.street > initial_street

    def test_multiple_raises_in_round(self, env, complete_selection_phase, action_helper):
        """Test multiple raises in same betting round"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_street = env.street

        # Player 1 raises
        action = action_helper.raise_bet(5)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            # Next player re-raises
            action = action_helper.raise_bet(5)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Should still be on same street
            if not terminated:
                assert env.street == initial_street

    def test_street_progression_flop_turn_river(self, env, complete_selection_phase, action_helper):
        """Test that streets progress correctly: flop -> turn -> river"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        assert env.street == 1  # Flop

        # Check around on flop
        for _ in range(6):
            if env.street == 1:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        assert env.street == 2  # Turn

        # Check around on turn
        for _ in range(6):
            if env.street == 2:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        assert env.street == 3  # River

        # Check around on river
        for _ in range(6):
            if env.street == 3:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        # Should reach showdown (street 4)
        assert env.street == 4
        assert terminated
