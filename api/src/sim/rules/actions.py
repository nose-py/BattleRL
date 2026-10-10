DIRECTIONS = {
    1: (0, -1),
    2: (1, 0),
    3: (0, 1),
    4: (-1, 0),
}

SHOOT_DIRECTIONS = {
    action + 4: direction
    for action, direction in DIRECTIONS.items()
}

ACTION_NAMES = {
    0: "wait",
    1: "move_up",
    2: "move_right",
    3: "move_down",
    4: "move_left",
    5: "shoot_up",
    6: "shoot_right",
    7: "shoot_down",
    8: "shoot_left",
}