class Node_BFS:
    def __init__(self, curr_state):
        self.curr_state = curr_state
        self.heuristic_value = 0
        self.path = []

    def __lt__(self, other):
        return self.heuristic_value < other.heuristic_value

    def __eq__(self, other):
        return self.curr_state == other.curr_state

    def __gt__(self, other):
        return self.heuristic_value > other.heuristic_value


class Node_AStar:
    def __init__(self, curr_state):
        self.curr_state = curr_state
        self.distance = 0
        self.heuristic_value = 0
        self.path = []

    def __lt__(self, other):
        return self.distance + self.heuristic_value < other.distance + other.heuristic_value

    def __eq__(self, other):
        return self.curr_state == other.curr_state

    def __gt__(self, other):
        return self.distance + self.heuristic_value > other.distance + other.heuristic_value