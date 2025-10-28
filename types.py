from dataclasses import dataclass
from typing import (
    TypedDict,
    List
)

@dataclass 
class Observation(TypedDict):
    street: int # is there a turn and river?? -dylan
    acting_agent: int # index of who is currently at play
    my_cards: List[int]
    community_cards: List[int] # flattened array of 3 boards
    my_idx: int # 
    bets: List[int] # list of players bets
    min_raise: int
    max_raise: int
    valid_actions: List[int]
    time_used: float
    time_left: float
    opp_last_action: str # do we need this?? -dylan


@dataclass
class Info(TypedDict):
    ...
    