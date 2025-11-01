"""
Test suite for 6-player triple-board bomb-pot poker engine

Tests cover:
- 6-player game initialization
- Card selection phase (5 hole cards -> 2 kept)
- Single post-flop betting round
- Triple-board showdown and pot splitting
- Position tracking and button rotation
- Less than 6 players (with null seats)
"""

from gym_env import PokerEnv
from poker_types import NUM_SEATS, NUM_BOARDS, HOLE_CARDS_PER_PLAYER, BOMB_POT_ANTE
import logging

logging.basicConfig(level=logging.DEBUG)

# Card encoding helpers (same as original tests)
RANKS = "23456789A"
SUITS = "dhs"  # no clubs


def int_to_card_str(card_int: int):
    """Convert from integer encoding to card string (e.g., 0 -> '2d')"""
    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return rank + suit


def card_str_to_int(card_str: str):
    """Convert from card string to integer encoding (e.g., '2d' -> 0)"""
    rank, suit = card_str[0], card_str[1]
    return (SUITS.index(suit) * len(RANKS)) + RANKS.index(rank)


def check_observation(expected_obs: dict, got_obs: dict):
    """Helper to verify observation fields match expected values"""
    for field, value in expected_obs.items():
        assert field in got_obs, f"Field {field} was expected, but wasn't present in obs: {got_obs}"
        assert got_obs[field] == value, f"Field {field} failed: expected {value}, got {got_obs[field]}"


# ============================================================================
# RESET() TESTS - 6 Player Initialization
# ============================================================================

def test_reset_6_players_full_table():
    """
    Test that reset() correctly initializes a 6-player game:
    - 6 players seated at positions 0-5
    - Each player receives 5 hole cards
    - All players automatically commit $1 bomb-pot
    - 15 community cards dealt (hidden initially)
    - Button at position 0 for hand 0
    - Street 0 (card selection phase)
    """
    env = PokerEnv(num_players=6)

    # Create rigged deck with known cards
    # _draw_card() pops from index 0 (front of list)
    # Dealing order: P0 cards, P1 cards, ..., P5 cards, then community cards

    rigged_deck = []
    player_cards = []

    # Add player cards first (cards drawn first)
    card_index = 0
    for p in range(NUM_SEATS):
        player_hole = [card_index + i for i in range(HOLE_CARDS_PER_PLAYER)]
        player_cards.append(player_hole)
        rigged_deck.extend(player_hole)
        card_index += HOLE_CARDS_PER_PLAYER

    # Add community cards (15 total, drawn after player cards)
    # Board 0: cards 30-34
    # Board 1: cards 35-39
    # Board 2: cards 40-44
    for board in range(NUM_BOARDS):
        for card in range(5):
            rigged_deck.append(card_index)
            card_index += 1

    # Reset the environment
    observations, info = env.reset(options={"cards": rigged_deck})

    # Verify we get 6 observations (tuple of 6)
    assert len(observations) == NUM_SEATS, f"Expected {NUM_SEATS} observations, got {len(observations)}"

    # Verify each player's observation
    for seat in range(NUM_SEATS):
        obs = observations[seat]
        assert obs is not None, f"Player {seat} observation should not be None"

        # Expected hole cards for this player (already integers)
        expected_hole = player_cards[seat]

        expected_obs = {
            'seat': seat,
            'street': 0,  # Card selection phase
            'hole_cards': expected_hole,
            'my_stack': -BOMB_POT_ANTE,  # Net: started at 0, committed $1
            'pot_total': NUM_SEATS * BOMB_POT_ANTE,  # $6 total ($1 × 6 players)
            'button_position': 0,  # Button starts at seat 0
            'hand_number': 0,
        }

        # Check observation fields
        check_observation(expected_obs, obs)

        # Community cards should be hidden (-1) during card selection
        assert obs['community_cards'] == [[-1]*5 for _ in range(NUM_BOARDS)], \
            "Community cards should be hidden during card selection phase"

        # All players should have committed bomb pot
        assert obs['bets'] == [BOMB_POT_ANTE] * NUM_SEATS, \
            f"All players should have bet ${BOMB_POT_ANTE}, got {obs['bets']}"

        # Stacks should all be at -$1 after bomb-pot commitment (tracking net, infinite bankroll)
        assert obs['all_stacks'] == [-BOMB_POT_ANTE] * NUM_SEATS, \
            f"All stacks should be at -{BOMB_POT_ANTE} after bomb-pot, got {obs['all_stacks']}"


def test_reset_4_players_with_empty_seats():
    """
    Test that reset() handles less than 6 players correctly:
    - Only 4 players active (seats 0-3)
    - Seats 4-5 have null observations
    - Bomb pot only includes active players ($4 total)
    """
    env = PokerEnv(num_players=4)

    # Create simpler rigged deck for 4 players
    rigged_deck = []

    # Community cards (15 cards)
    for i in range(15):
        rigged_deck.append(i)

    # 4 players × 5 cards each
    for player in range(3, -1, -1):
        for card in range(5):
            rigged_deck.append(15 + player * 5 + card)

    observations, info = env.reset(options={"cards": rigged_deck})

    # Still returns 6-tuple, but last 2 are None
    assert len(observations) == NUM_SEATS, "Should always return 6-tuple"

    # First 4 observations should be valid
    for seat in range(4):
        assert observations[seat] is not None, f"Seat {seat} should have valid observation"
        assert observations[seat]['seat'] == seat
        assert len(observations[seat]['hole_cards']) == HOLE_CARDS_PER_PLAYER
        assert observations[seat]['pot_total'] == 4 * BOMB_POT_ANTE  # Only 4 players

    # Seats 4-5 should be None
    assert observations[4] is None, "Seat 4 should be None (empty)"
    assert observations[5] is None, "Seat 5 should be None (empty)"


def test_button_rotation():
    """
    Test that button position rotates correctly across hands:
    - Hand 0: button at seat 0
    - Hand 1: button at seat 1
    - Hand 5: button at seat 5
    - Hand 6: button wraps to seat 0
    """
    env = PokerEnv(num_players=6, num_hands=7)

    for hand_num in range(7):
        expected_button = hand_num % NUM_SEATS

        # Simple rigged deck
        rigged_deck = list(range(15 + NUM_SEATS * HOLE_CARDS_PER_PLAYER))
        observations, info = env.reset(options={"cards": rigged_deck})

        # All active players should see correct button position
        for seat in range(NUM_SEATS):
            assert observations[seat]['button_position'] == expected_button, \
                f"Hand {hand_num}: expected button at {expected_button}, got {observations[seat]['button_position']}"
            assert observations[seat]['hand_number'] == hand_num


# ============================================================================
# CARD SELECTION PHASE TESTS
# ============================================================================

def test_card_selection_valid():
    """
    Test that players can select 2 cards from their 5 hole cards after seeing community cards
    """
    # TODO: Implement after gym_env.py card selection is ready
    pass


def test_card_selection_invalid_count():
    """
    Test that selecting wrong number of cards (not 2) results in fold/penalty
    """
    # TODO: Implement after gym_env.py card selection is ready
    pass


def test_card_selection_invalid_card():
    """
    Test that selecting a card not in hole cards results in fold/penalty
    """
    # TODO: Implement after gym_env.py card selection is ready
    pass


# ============================================================================
# BETTING ROUND TESTS
# ============================================================================

def test_betting_order_from_button():
    """
    Test that betting starts left of button and proceeds clockwise
    """
    # TODO: Implement after gym_env.py betting is ready
    pass


def test_bet_cap_enforcement():
    """
    Test that raises above BET_CAP are rejected or clamped
    """
    # TODO: Implement after gym_env.py betting is ready
    pass


# ============================================================================
# SHOWDOWN TESTS
# ============================================================================

def test_showdown_single_winner_all_boards():
    """
    Test: One player has best hand on all 3 boards, wins entire pot
    """
    # TODO: Implement after gym_env.py showdown is ready
    pass


def test_showdown_different_winners_per_board():
    """
    Test: Different players win each board, pot split 3 ways
    """
    # TODO: Implement after gym_env.py showdown is ready
    pass


def test_showdown_tie_on_one_board():
    """
    Test: Two players tie on one board, split that board's pot
    """
    # TODO: Implement after gym_env.py showdown is ready
    pass


def test_all_fold_except_one():
    """
    Test: When all players fold except one, that player wins without showdown
    """
    # TODO: Implement after gym_env.py betting is ready
    pass


if __name__ == "__main__":
    # Run basic tests
    test_reset_6_players_full_table()
    test_reset_4_players_with_empty_seats()
    test_button_rotation()
    print("All reset() tests passed!")
