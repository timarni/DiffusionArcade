from constants import *

def epsilon_by_episode(ep):
    """Give back epsilon corresponding to each episode"""
    frac = max(0, (EPSILON_DECAY_LEN - ep) / EPSILON_DECAY_LEN)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * frac

### Helper functions
def discretise(state: dict) -> tuple:
    """Convert the raw dict into a small, hashable tuple.
    Because storing continous values in a Q-table needs a lot of memory"""
    py, pvy = state["player_y"], state["player_velocity"]
    cy = state["cpu_y"]
    bx, by = state["ball_x"], state["ball_y"]
    bvx, bvy = state["ball_velocity_x"], state["ball_velocity_y"]

    # Simple hash: put each value in a rough bucket
    return (
        int(py // (48 / STATE_BINS[0])),
        int(pvy // (15 / STATE_BINS[1])),
        int(cy // (48 / STATE_BINS[2])),
        int(bx // (64 / STATE_BINS[3])),
        int(by // (48 / STATE_BINS[6])),
        int((bvx > 0) - (bvx < 0)), # -1, 0, or +1
        int((bvy > 0) - (bvy < 0)),
    )