"""
Observation space tests for gym_env_v2.py

Tests that observations are properly formatted and contain correct information.
"""

import pytest


class TestObservationStructure:
    """Test observation dict structure"""

    def test_observation_has_all_required_fields(self, env):
        """Test that observation contains all required fields"""
        obs, info = env.reset(seed=42)

        required_fields = [
            'street', 'my_cards', 'my_hand', 'community_cards',
            'acting_agent', 'seat', 'button', 'bets',
            'min_raise', 'max_raise', 'valid_actions', 'players_active'
        ]

        for field in required_fields:
            assert field in obs[0], f"Missing field: {field}"

    def test_observation_tuple_length(self, env):
        """Test that observation tuple has 6 player observations"""
        obs, info = env.reset(seed=42)

        assert len(obs) == 6

    def test_each_player_has_unique_seat(self, env):
        """Test that each player observation has correct seat number"""
        obs, info = env.reset(seed=42)

        for i in range(6):
            assert obs[i]['seat'] == i

    def test_acting_agent_same_for_all_observations(self, env):
        """Test that acting_agent is same across all player observations"""
        obs, info = env.reset(seed=42)

        acting_agent = obs[0]['acting_agent']
        for i in range(6):
            assert obs[i]['acting_agent'] == acting_agent


class TestCardEncoding:
    """Test card encoding in observations"""

    def test_my_cards_encoded_correctly(self, env):
        """Test that my_cards are properly encoded"""
        obs, info = env.reset(seed=42)

        # Cards should be encoded: -1 -> 0, 0-51 -> 1-52
        my_cards = obs[0]['my_cards']

        assert len(my_cards) == 5  # 5 hole cards
        for card in my_cards:
            assert 0 <= card <= 52  # Encoded range

    def test_hidden_cards_encoded_as_zero(self, env, complete_selection_phase):
        """Test that hidden cards are encoded as 0"""
        obs, info = env.reset(seed=42)
        obs = complete_selection_phase(env)

        # After selection, 3 cards should be hidden (encoded as 0)
        my_cards = obs[0]['my_cards']
        hidden_count = sum(1 for c in my_cards if c == 0)

        assert hidden_count == 3  # 3 cards masked after selection

    def test_community_cards_encoded_correctly(self, env):
        """Test that community cards are properly encoded"""
        obs, info = env.reset(seed=42)

        community_cards = obs[0]['community_cards']

        assert len(community_cards) == 5  # Always 5 community cards
        for card in community_cards:
            assert 0 <= card <= 52  # Encoded range

    def test_my_hand_shows_selected_cards(self, env, complete_selection_phase):
        """Test that my_hand shows the 2 selected cards after selection"""
        obs, info = env.reset(seed=42)
        obs = complete_selection_phase(env)

        my_hand = obs[0]['my_hand']

        assert len(my_hand) == 2  # Exactly 2 cards
        # Both should be non-hidden
        assert all(c > 0 for c in my_hand)

    def test_my_hand_hidden_during_selection(self, env):
        """Test that my_hand is (0,0) during selection phase"""
        obs, info = env.reset(seed=42)

        # During selection phase, my_hand should be hidden
        # (unless player has already selected)
        my_hand = obs[0]['my_hand']

        # First player hasn't selected yet
        assert my_hand == (0, 0)


class TestBettingInformation:
    """Test betting-related observation fields"""

    def test_bets_tuple_length(self, env):
        """Test that bets tuple has 6 elements"""
        obs, info = env.reset(seed=42)

        bets = obs[0]['bets']
        assert len(bets) == 6

    def test_initial_bets_all_ante(self, env):
        """Test that all players start with ante bets"""
        obs, info = env.reset(seed=42)

        bets = obs[0]['bets']
        # All should have posted ante
        assert all(bet == 1 for bet in bets)

    def test_min_raise_reasonable(self, env):
        """Test that min_raise is a reasonable value"""
        obs, info = env.reset(seed=42)

        min_raise = obs[0]['min_raise']
        # Should be at least ante amount
        assert min_raise >= 1
        # Should not exceed max bet
        assert min_raise <= 100

    def test_max_raise_reasonable(self, env):
        """Test that max_raise is reasonable"""
        obs, info = env.reset(seed=42)

        max_raise = obs[0]['max_raise']
        # Should be non-negative
        assert max_raise >= 0
        # Should not exceed remaining to max bet
        assert max_raise <= 100

    def test_max_raise_decreases_with_bets(self, env, complete_selection_phase, action_helper):
        """Test that max_raise decreases as bets increase"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        initial_max_raise = obs[0]['max_raise']

        # Someone raises
        action = action_helper.raise_bet(20)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            new_max_raise = obs[0]['max_raise']
            # Max raise should decrease
            assert new_max_raise < initial_max_raise


class TestValidActions:
    """Test valid_actions field"""

    def test_valid_actions_length(self, env):
        """Test that valid_actions has correct length"""
        obs, info = env.reset(seed=42)

        valid_actions = obs[0]['valid_actions']
        # [FOLD, RAISE, CHECK, CALL, SELECT_CARDS]
        assert len(valid_actions) == 5

    def test_valid_actions_during_selection(self, env):
        """Test that only SELECT_CARDS is valid during selection"""
        obs, info = env.reset(seed=42)

        # For the acting player
        acting_player = env.acting_agent
        valid_actions = obs[acting_player]['valid_actions']

        # Only SELECT_CARDS should be valid
        assert valid_actions[4] == 1  # SELECT_CARDS
        assert valid_actions[0] == 0  # FOLD
        assert valid_actions[1] == 0  # RAISE
        assert valid_actions[2] == 0  # CHECK
        assert valid_actions[3] == 0  # CALL

    def test_valid_actions_during_betting_with_equal_bets(self, env, complete_selection_phase):
        """Test valid actions when bets are equal"""
        obs, info = env.reset(seed=42)
        obs = complete_selection_phase(env)

        acting_player = env.acting_agent
        valid_actions = obs[acting_player]['valid_actions']

        # When bets are equal:
        # - FOLD should be valid
        # - CHECK should be valid
        # - CALL should be invalid (bets already equal)
        # - RAISE should be valid
        # - SELECT_CARDS should be invalid

        assert valid_actions[0] == 1  # FOLD
        assert valid_actions[2] == 1  # CHECK
        assert valid_actions[3] == 0  # CALL invalid
        assert valid_actions[4] == 0  # SELECT_CARDS

    def test_valid_actions_after_raise(self, env, complete_selection_phase, action_helper):
        """Test valid actions after someone raises"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Someone raises
        action = action_helper.raise_bet(10)
        obs, rewards, terminated, truncated, info = env.step(action)

        if not terminated:
            acting_player = env.acting_agent
            valid_actions = obs[acting_player]['valid_actions']

            # After a raise:
            # - FOLD should be valid
            # - CHECK should be invalid (need to match bet)
            # - CALL should be valid
            # - RAISE should be valid

            assert valid_actions[0] == 1  # FOLD
            assert valid_actions[2] == 0  # CHECK invalid
            assert valid_actions[3] == 1  # CALL


class TestPlayersActive:
    """Test players_active field"""

    def test_players_active_length(self, env):
        """Test that players_active has 6 elements"""
        obs, info = env.reset(seed=42)

        players_active = obs[0]['players_active']
        assert len(players_active) == 6

    def test_all_players_active_initially(self, env):
        """Test that all players are active at start"""
        obs, info = env.reset(seed=42)

        players_active = obs[0]['players_active']
        assert all(players_active)

    def test_players_active_updates_on_fold(self, env, complete_selection_phase, action_helper):
        """Test that players_active updates when player folds"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        folding_player = env.acting_agent

        # Player folds
        action = action_helper.fold()
        obs, rewards, terminated, truncated, info = env.step(action)

        # Check players_active in all observations
        players_active = obs[0]['players_active']
        assert players_active[folding_player] == False


class TestStreetInformation:
    """Test street-related observation fields"""

    def test_street_starts_at_one(self, env):
        """Test that street starts at 1 (flop) for bomb pot"""
        obs, info = env.reset(seed=42)

        street = obs[0]['street']
        assert street == 1  # Bomb pot starts at flop

    def test_street_advances_correctly(self, env, complete_selection_phase, action_helper):
        """Test that street field updates correctly"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        assert obs[0]['street'] == 1  # Flop

        # Check around to advance street
        for _ in range(6):
            if obs[0]['street'] == 1:
                action = action_helper.check()
                obs, rewards, terminated, truncated, info = env.step(action)

        assert obs[0]['street'] == 2  # Turn


class TestButtonAndPosition:
    """Test button and position information"""

    def test_button_field_exists(self, env):
        """Test that button field is present"""
        obs, info = env.reset(seed=42)

        button = obs[0]['button']
        assert 0 <= button < 6

    def test_button_same_for_all_players(self, env):
        """Test that button is same across all observations"""
        obs, info = env.reset(seed=42)

        button = obs[0]['button']
        for i in range(6):
            assert obs[i]['button'] == button

    def test_acting_agent_is_valid_player(self, env):
        """Test that acting_agent is valid player index"""
        obs, info = env.reset(seed=42)

        acting_agent = obs[0]['acting_agent']
        assert 0 <= acting_agent < 6


class TestObservationConsistency:
    """Test that observations are consistent across different views"""

    def test_bets_same_across_all_observations(self, env):
        """Test that bets are same for all players"""
        obs, info = env.reset(seed=42)

        bets_0 = obs[0]['bets']
        for i in range(1, 6):
            assert obs[i]['bets'] == bets_0

    def test_players_active_same_across_all_observations(self, env):
        """Test that players_active is same for all players"""
        obs, info = env.reset(seed=42)

        players_active_0 = obs[0]['players_active']
        for i in range(1, 6):
            assert obs[i]['players_active'] == players_active_0

    def test_community_cards_same_across_all_observations(self, env):
        """Test that community cards are same for all players"""
        obs, info = env.reset(seed=42)

        community_cards_0 = obs[0]['community_cards']
        for i in range(1, 6):
            assert obs[i]['community_cards'] == community_cards_0

    def test_different_players_see_different_hole_cards(self, env):
        """Test that different players have different hole cards"""
        obs, info = env.reset(seed=42)

        # Each player should have different hole cards
        all_my_cards = [obs[i]['my_cards'] for i in range(6)]

        # At least some should be different
        unique_hands = set(all_my_cards)
        assert len(unique_hands) > 1
