"""
Test infrastructure for tournament server.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.match import PokerMatch, AgentFailureTracker, AgentAPIClient
from server.tournament_server import TournamentServer
from utils.errors import DisconnectionError, InvalidActionError


@pytest.mark.asyncio
async def test_disconnection_auto_folds():
    """Test that disconnected bot is auto-folded."""
    # Create match with 3 bots
    match = PokerMatch(
        agent_urls=[
            'http://localhost:8001',
            'http://localhost:8002',
            'http://localhost:8003'
        ]
    )
    
    # Mock client 1 to raise DisconnectionError
    match.clients[1].get_action = AsyncMock(side_effect=DisconnectionError("Connection refused"))
    
    # Start hand
    obs, info = match.env.reset(seed=42)
    
    # If player 1 is acting and disconnected, should auto-fold
    if match.env.acting_agent == 1:
        obs, _ = match.env._get_single_player_obs(1)
        valid_actions = match.env._get_valid_actions(1)
        action = await match._get_action_with_recovery(1, obs, valid_actions)
        assert action == (0, 0, 0, 0)  # FOLD
        assert match.failure_tracker.failure_counts[1] == 1


@pytest.mark.asyncio
async def test_invalid_action_auto_folds():
    """Test that invalid action is treated as fold."""
    match = PokerMatch(
        agent_urls=[
            'http://localhost:8001',
            'http://localhost:8002'
        ]
    )
    
    # Mock client to return invalid action
    match.clients[0].get_action = AsyncMock(return_value=(99, 0, 0, 0))  # Invalid action type
    
    obs, info = match.env.reset(seed=42)
    
    if match.env.acting_agent == 0:
        obs, _ = match.env._get_single_player_obs(0)
        valid_actions = match.env._get_valid_actions(0)
        action = await match._get_action_with_recovery(0, obs, valid_actions)
        assert action == (0, 0, 0, 0)  # Should auto-fold
        assert match.failure_tracker.failure_counts[0] == 1


@pytest.mark.asyncio
async def test_three_failures_disqualification():
    """Test that 3 failures lead to disqualification."""
    match = PokerMatch(
        agent_urls=[
            'http://localhost:8001',
            'http://localhost:8002'
        ]
    )
    
    # Simulate 3 failures
    for i in range(3):
        match.failure_tracker.record_failure(0, f"Error {i}")
    
    assert match.failure_tracker.should_disqualify(0)
    assert 0 in match.failure_tracker.disqualified


@pytest.mark.asyncio
async def test_variable_player_support():
    """Test matches with different player counts."""
    for num_players in [2, 3, 4, 5, 6]:
        match = PokerMatch(
            agent_urls=[f'http://localhost:800{i}' for i in range(num_players)]
        )
        assert match.num_players == num_players
        assert len(match.clients) == num_players
        assert match.env.NUM_PLAYERS == num_players


@pytest.mark.asyncio
async def test_tournament_server_creation():
    """Test tournament server match creation."""
    server = TournamentServer()
    
    server.register_bot('http://localhost:8001')
    server.register_bot('http://localhost:8002')
    server.register_bot('http://localhost:8003')
    
    match = await server.create_match([
        'http://localhost:8001',
        'http://localhost:8002'
    ])
    
    assert match.match_id in server.active_matches
    assert match.num_players == 2


@pytest.mark.asyncio
async def test_action_validation():
    """Test action validation logic."""
    match = PokerMatch(
        agent_urls=[
            'http://localhost:8001',
            'http://localhost:8002'
        ]
    )
    
    obs, info = match.env.reset(seed=42)
    
    # Test invalid action type
    valid_actions = [1, 0, 1, 1, 0]  # FOLD, RAISE disabled, CHECK, CALL, SELECT_CARDS disabled
    assert not match._validate_action((1, 0, 0, 0), valid_actions)  # RAISE not valid
    
    # Test valid action
    assert match._validate_action((0, 0, 0, 0), valid_actions)  # FOLD is valid


def test_failure_tracker():
    """Test failure tracking logic."""
    tracker = AgentFailureTracker(num_players=3)
    
    # Record failures
    tracker.record_failure(0, "Error 1")
    tracker.record_failure(0, "Error 2")
    assert tracker.failure_counts[0] == 2
    assert not tracker.should_disqualify(0)
    
    tracker.record_failure(0, "Error 3")
    assert tracker.should_disqualify(0)
    assert 0 in tracker.disqualified
