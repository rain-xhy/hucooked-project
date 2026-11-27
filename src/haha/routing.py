"""Shortest-path action support for goal-conditioned workers."""

from collections import deque


def advancing_moves(floor, blocked, start, goal, directions):
    if goal in blocked:
        return [False] * len(directions)
    distances = {goal: 0}
    queue = deque([goal])
    while queue and start not in distances:
        x, y = queue.popleft()
        for dx, dy in directions:
            neighbor = x + dx, y + dy
            if neighbor in floor and neighbor not in blocked and neighbor not in distances:
                distances[neighbor] = distances[(x, y)] + 1
                queue.append(neighbor)
    distance = distances.get(start)
    return [distance is not None and distances.get((start[0] + dx, start[1] + dy)) == distance - 1
            for dx, dy in directions]
