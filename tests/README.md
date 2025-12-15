# Test Suite for 6-Player Bomb Pot PLO Engine

Comprehensive test suite for `engine/gym_env_v2.py` - the 6-player Bomb Pot Pot Limit Omaha poker engine.

## Test Structure

### Test Files

1. **`test_basic_functionality.py`** - Basic game mechanics
   - Environment initialization and reset
   - Card selection phase
   - Basic game flow and progression
   - Action rotation
   - Winner determination

2. **`test_plo_evaluation.py`** - PLO hand evaluation
   - PLO evaluation using exactly 2 hole + 3 board cards
   - Winner determination
   - Pot calculation and reward distribution
   - Tie handling

3. **`test_betting_logic.py`** - Betting actions and rounds
   - FOLD, RAISE, CHECK, CALL actions
   - Betting round completion
   - Street progression (flop → turn → river)
   - Multiple raises in same round
   - Valid/invalid action detection

4. **`test_edge_cases.py`** - Edge cases and unusual scenarios
   - Invalid actions (treated as fold)
   - All-in situations
   - Max bet scenarios
   - Deterministic behavior with seeds
   - Game state consistency
   - Community card visibility progression
   - Zero-sum reward verification

5. **`test_observations.py`** - Observation space correctness
   - Observation structure and fields
   - Card encoding (internal -1/0-51 → observation 0/1-52)
   - Betting information accuracy
   - Valid actions correctness
   - Players active status
   - Street and position information
   - Observation consistency across players

### Fixtures (conftest.py)

- **`env`** - Fresh PokerEnv instance
- **`env_with_seed`** - Deterministic environment (seed=42)
- **`rigged_deck`** - Factory to create deterministic card decks
- **`complete_selection_phase`** - Helper to complete card selection
- **`action_helper`** - Helper class for creating action tuples

## Running Tests

### Run All Tests
```bash
cd /Users/dylanca/Desktop/Poker\ ai/poker-engine-2026
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_basic_functionality.py -v
pytest tests/test_plo_evaluation.py -v
pytest tests/test_betting_logic.py -v
pytest tests/test_edge_cases.py -v
pytest tests/test_observations.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_basic_functionality.py::TestEnvironmentInitialization -v
```

### Run Specific Test
```bash
pytest tests/test_basic_functionality.py::TestEnvironmentInitialization::test_env_creation -v
```

### Run with Coverage
```bash
pytest tests/ --cov=engine.gym_env_v2 --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -vv -s
```

## Test Coverage

The test suite covers:

### Core Functionality
- ✅ Environment creation and initialization
- ✅ Reset with deterministic seeds
- ✅ Card selection phase (all 6 players)
- ✅ Betting rounds and actions
- ✅ Street progression (flop → turn → river)
- ✅ Winner determination
- ✅ Reward calculation

### PLO Rules
- ✅ Hand evaluation (exactly 2 hole + 3 board)
- ✅ All C(5,3) = 10 board combinations tried
- ✅ Best hand selection
- ✅ Tie handling and pot splitting

### Betting Logic
- ✅ FOLD action (marks player inactive)
- ✅ RAISE action (increases bet, updates min_raise)
- ✅ CHECK action (valid with equal bets)
- ✅ CALL action (matches max bet)
- ✅ Invalid action detection
- ✅ Valid actions calculation
- ✅ Betting round completion
- ✅ Multiple raises in same round

### Edge Cases
- ✅ Invalid actions treated as fold
- ✅ All-in situations
- ✅ Max bet limits
- ✅ Winner by elimination (all fold except one)
- ✅ Deterministic behavior with same seed
- ✅ Game state consistency
- ✅ Progressive community card visibility
- ✅ Zero-sum rewards

### Observations
- ✅ All required fields present
- ✅ Correct observation structure
- ✅ Card encoding correctness
- ✅ Betting information accuracy
- ✅ Valid actions correctness
- ✅ Players active status
- ✅ Consistency across player views

## Test Statistics

- **Total Test Files**: 5
- **Total Test Classes**: 28
- **Total Test Functions**: 100+
- **Coverage**: High coverage of core game logic

## Adding New Tests

To add new tests:

1. Create test file in `tests/` directory
2. Import fixtures from `conftest.py`
3. Use descriptive class and function names
4. Follow existing patterns for consistency

Example:
```python
"""
Description of test module
"""

import pytest


class TestFeatureName:
    """Test specific feature"""

    def test_specific_behavior(self, env, action_helper):
        """Test that specific behavior works correctly"""
        obs, info = env.reset(seed=42)

        # Test logic here
        assert some_condition
```

## Common Testing Patterns

### Testing Card Selection
```python
def test_card_selection(self, env, complete_selection_phase):
    obs, info = env.reset(seed=42)
    obs = complete_selection_phase(env)
    # Now in betting phase
```

### Testing Actions
```python
def test_fold_action(self, env, complete_selection_phase, action_helper):
    obs, info = env.reset(seed=42)
    complete_selection_phase(env)

    action = action_helper.fold()
    obs, rewards, terminated, truncated, info = env.step(action)
```

### Testing Game Progression
```python
def test_game_progression(self, env, complete_selection_phase, action_helper):
    obs, info = env.reset(seed=42)
    complete_selection_phase(env)

    terminated = False
    while not terminated:
        action = action_helper.check()
        obs, rewards, terminated, truncated, info = env.step(action)
```

## Debugging Tests

### Run with Print Statements
```bash
pytest tests/ -s
```

### Run with PDB Debugger
```bash
pytest tests/test_basic_functionality.py --pdb
```

### Show Local Variables on Failure
```bash
pytest tests/ -l
```

## Dependencies

- pytest
- numpy
- gym
- treys
- engine.gym_env_v2

## Notes

- Tests use deterministic seeds for reproducibility
- Each test is independent (fresh environment)
- Tests cover both success and failure cases
- Edge cases extensively tested
- Observations validated for correctness
