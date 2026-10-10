
def empty_rewards(participants):
    return {agent_id: 0.0 for agent_id in participants}

def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])