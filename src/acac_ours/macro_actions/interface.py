PRIMITIVE_ACTIONS = ("up", "down", "right", "left", "stay", "interact")

# Each station option specifies its name, standing tile, station, and interaction on arrival.
NAV_TASKS = (
    ("get_main", (4, 1), "M", True),
    ("get_side", (5, 1), "L", True),
    ("get_plate", (1, 5), "D", True),
    ("go_cooker_left", (2, 2), "G", False),
    ("go_cooker_right", (4, 2), "G", False),
    ("go_cutting", (5, 2), "K", False),
    ("use_assembly_left", (2, 3), "A", True),
    ("use_assembly_right", (4, 3), "A", True),
    ("use_buffer_left", (2, 4), "B", True),
    ("use_buffer_right", (4, 4), "B", True),
    ("deliver", (1, 3), "S", True),
)
MACRO_NAMES = (*PRIMITIVE_ACTIONS, *(task[0] for task in NAV_TASKS), "chop_until_complete")


def select_actions(current_actions, macro_done, proposed_actions):
    """Keep an active option until its chef is ready for a new decision."""
    actions = list(current_actions)
    for chef, ready in enumerate(macro_done):
        if ready:
            selected = int(proposed_actions[chef])
            if not 0 <= selected < len(MACRO_NAMES):
                raise ValueError("Missing macro action for an available chef")
            actions[chef] = selected
    return actions


def boundary_info(current_actions, finished, episode_done):
    """An option ends on completion or invalidation, or when the episode ends."""
    return {
        "cur_mac": list(current_actions),
        "mac_done": [int(option_finished or episode_done) for option_finished in finished],
    }
