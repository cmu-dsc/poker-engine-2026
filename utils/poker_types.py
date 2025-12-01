"""
Types for JSON like objects

This includes observations, action types, and payouts for 6-player Bomb Pot PLO

"""
from typing import TypedDict, List, Tuple

class Observation(TypedDict):
    """Observation for a single player in 6-player Bomb Pot Omaha"""
    street: int                      # Current street (1=flop, 2=turn, 3=river)
    acting_agent: int                # Which player acts next (0-5)
    seat: int                        # This player's seat number (0-5)
    my_cards: Tuple[int, ...]        # Player's hole cards (5 initially, 2 after fixing)
    community_cards: Tuple[int, ...] # 5 community cards (all visible from flop in bomb pot)
    bets: Tuple[int, ...]            # Bet ledger of all players (length 6)
    min_raise: int                   # Minimum raise amount
    max_raise: int                   # Maximum raise amount
    valid_actions: List[bool]        # Which actions are valid [FOLD, RAISE, CHECK, CALL, SELECT_CARDS] 
    players_active: Tuple[bool, ...] # Which players are still active/not folded (length 6)

