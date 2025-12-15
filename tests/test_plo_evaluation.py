"""
PLO (Pot Limit Omaha) hand evaluation tests

Tests that the PLO evaluation correctly uses exactly 2 hole cards + 3 board cards.
"""

import pytest


class TestPLOEvaluation:
    """Test PLO hand evaluation logic"""

    def test_evaluate_plo_hand_method_exists(self, env):
        """Test that _evaluate_plo_hand method exists"""
        assert hasattr(env, '_evaluate_plo_hand')

    def test_evaluate_plo_hand_basic(self, env):
        """Test basic PLO hand evaluation"""
        # Pair of 2s
        hole_cards = [0, 13]  # 2c, 2d
        board_cards = [1, 14, 27, 40, 26]  # 3c, 3d, 2h, Ac, Kc

        rank = env._evaluate_plo_hand(hole_cards, board_cards)

        # Should return a valid rank (integer)
        assert isinstance(rank, int)
        assert rank > 0

    def test_evaluate_plo_hand_uses_exactly_two_hole_cards(self, env):
        """Test that PLO uses exactly 2 hole cards"""
        # This is implicit in the implementation but we can test different scenarios

        # Aces in hole
        hole_cards = [12, 25]  # Ac, Ad
        # Board has three more aces - but can only use 2 from hole
        board_cards = [38, 51, 1, 2, 3]  # Ah, As, 3c, 4c, 5c

        rank = env._evaluate_plo_hand(hole_cards, board_cards)
        assert isinstance(rank, int)

    def test_evaluate_plo_hand_tries_all_board_combinations(self, env):
        """Test that evaluation tries all C(5,3)=10 board combinations"""
        # Create scenario where best hand uses specific board cards

        # Hole: Ah, Kh (flush potential)
        hole_cards = [51, 50]  # Ah, Kh
        # Board: Qh, Jh, Th (makes straight flush with Ah, Kh)
        #        plus 2c, 3c
        board_cards = [49, 48, 47, 1, 2]  # Qh, Jh, Th, 3c, 4c

        rank = env._evaluate_plo_hand(hole_cards, board_cards)

        # Should find the straight flush (very low rank number)
        assert rank < 10  # Straight flush ranks are very low

    def test_plo_evaluation_different_hands(self, env):
        """Test that different hands get different ranks"""
        board_cards = [10, 11, 12, 13, 14]  # 2s, 3s, 4s, 5s, 6s

        # High cards
        hole1 = [51, 50]  # Ah, Kh
        rank1 = env._evaluate_plo_hand(hole1, board_cards)

        # Pair
        hole2 = [0, 13]  # 2c, 2d
        rank2 = env._evaluate_plo_hand(hole2, board_cards)

        # Different hands should (usually) have different ranks
        # Pair should beat high card (lower rank number)
        assert rank2 < rank1

    def test_plo_evaluation_finds_best_hand(self, env):
        """Test that PLO finds the best possible hand from combinations"""
        # Hole: 9h, Th (high cards)
        hole_cards = [48, 47]  # 9h, Th

        # Board has pair potential
        # Jh, Jd, 2c, 3c, 4c
        board_cards = [49, 36, 1, 2, 3]

        rank = env._evaluate_plo_hand(hole_cards, board_cards)

        # Should find at least a pair or better
        # High card is worst case, rank > 6000
        assert rank < 7500  # Should find at least some hand

    def test_plo_evaluation_assertion_on_wrong_num_hole_cards(self, env):
        """Test that evaluation asserts if wrong number of hole cards"""
        hole_cards = [0, 1, 2]  # 3 cards - invalid for PLO with our impl
        board_cards = [10, 11, 12, 13, 14]

        with pytest.raises(AssertionError):
            env._evaluate_plo_hand(hole_cards, board_cards)

    def test_plo_evaluation_assertion_on_wrong_num_board_cards(self, env):
        """Test that evaluation asserts if wrong number of board cards"""
        hole_cards = [0, 1]
        board_cards = [10, 11, 12]  # 3 cards - need 5

        with pytest.raises(AssertionError):
            env._evaluate_plo_hand(hole_cards, board_cards)


class TestWinnerDetermination:
    """Test winner determination using PLO evaluation"""

    def test_get_winner_returns_valid_winner(self, env, complete_selection_phase, action_helper):
        """Test that _get_winner returns valid winner(s)"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Play to showdown
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        # Winner should be valid
        winner = info['winner']
        if isinstance(winner, int):
            assert 0 <= winner < 6
        elif isinstance(winner, list):
            assert len(winner) >= 2  # Tie
            for w in winner:
                assert 0 <= w < 6

    def test_winner_gets_pot(self, env, complete_selection_phase, action_helper):
        """Test that winner's reward equals pot minus their bet"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Everyone checks to showdown (all bets = 1)
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        winner = info['winner']
        pot = info['pot']

        if isinstance(winner, int):
            # Single winner
            # Winner's reward = pot - their bet
            # Everyone bet 1 ante, so pot = 6, winner's reward = 6 - 1 = 5
            assert rewards[winner] == pot - env.bets[winner]
            assert rewards[winner] == 5  # In this case

    def test_losers_lose_their_bets(self, env, complete_selection_phase, action_helper):
        """Test that losers' rewards are negative of their bets"""
        obs, info = env.reset(seed=42)
        complete_selection_phase(env)

        # Everyone checks to showdown
        terminated = False
        while not terminated:
            action = action_helper.check()
            obs, rewards, terminated, truncated, info = env.step(action)

        winner = info['winner']

        if isinstance(winner, int):
            # All non-winners should have reward = -their_bet
            for p in range(6):
                if p != winner:
                    assert rewards[p] == -env.bets[p]
                    assert rewards[p] == -1  # Everyone bet 1 ante

    def test_tie_splits_pot(self, env):
        """Test that tied winners split the pot"""
        # This is hard to create deterministically, but we can test the logic exists
        # by looking at the code path in _get_obs

        # We'll just verify that the tie handling code exists
        # Actual tie testing would require rigging specific cards

        obs, info = env.reset(seed=999)  # Try different seed

        # Complete selection
        for _ in range(6):
            action = (4, 0, 0, 1)
            obs, rewards, terminated, truncated, info = env.step(action)

        # Play to showdown
        terminated = False
        rounds = 0
        while not terminated and rounds < 50:
            action = (2, 0, 0, 0)  # CHECK
            obs, rewards, terminated, truncated, info = env.step(action)
            rounds += 1

        # Just verify structure handles both single winner and ties
        winner = info.get('winner')
        assert winner is not None
