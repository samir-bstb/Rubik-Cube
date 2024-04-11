import copy
from queue import Queue, PriorityQueue
from Nodes import Node_BFS, Node_AStar
from RubikCube import Cube


class Solver:
    def __init__(self, cube_array):
        self.cube = Cube()
        self.cube.cube_array = cube_array
        self.cube.up_mtx = self.cube.cube_array[0]
        self.cube.front_mtx = self.cube.cube_array[1]
        self.cube.low_mtx = self.cube.cube_array[2]
        self.cube.left_mtx = self.cube.cube_array[3]
        self.cube.right_mtx = self.cube.cube_array[4]
        self.cube.back_mtx = self.cube.cube_array[5]
        self.solved_cube = [
            [
                [1, 1, 1],  # Orange
                [1, 1, 1],
                [1, 1, 1]
            ],

            [
                [0, 0, 0],  # white
                [0, 0, 0],
                [0, 0, 0]
            ],

            [
                [2, 2, 2],  # Red
                [2, 2, 2],
                [2, 2, 2]
            ],
            [
                [3, 3, 3],  # Green
                [3, 3, 3],
                [3, 3, 3]
            ],
            [
                [4, 4, 4],  # Blue
                [4, 4, 4],
                [4, 4, 4]
            ],
            [
                [5, 5, 5],  # Yellow
                [5, 5, 5],
                [5, 5, 5]
            ]
        ]

        self.solved_cube2 = [571885666967682, 285942833483841, 1143771333935364, 2287542667870728, 4575085335741456,
                             9150170671482912]
        self.moves = ['F', 'R', 'U', 'B', 'L', 'D', 'G', 'S', 'W', 'V', 'I', 'O']
        self.Q = Queue()
        self.visited = []  # for breadth first search
        self.visited = set()  # for A* and BFS

    def bfs(self):
        solution = self.__bfs(self.cube.cube_array)
        if solution is None:
            print("No solution found")
        else:
            print(solution)

    def __bfs(self, start):
        self.visited = []
        cubie = self.cube.encode(start)
        self.Q.put((cubie, []))  # Store the sequence of moves with the state
        self.visited.append(cubie)

        print("bfs start", cubie)
        while not self.Q.empty():
            cubie, path = self.Q.get()
            if cubie == self.solved_cube2:
                print("Solved")
                return len(path), path

            for move in self.moves:
                curr_state = self.cube.decode(cubie)
                next_state = copy.deepcopy(curr_state)
                next_state = self.apply_move(move, next_state)
                next_state = self.cube.encode(next_state)
                if next_state not in self.visited:
                    self.Q.put((next_state, path + [move]))
                    self.visited.append(next_state)

    def apply_move(self, move, curr_state):
        self.cube.cube_array = curr_state
        self.cube.up_mtx = curr_state[0]
        self.cube.front_mtx = curr_state[1]
        self.cube.low_mtx = curr_state[2]
        self.cube.left_mtx = curr_state[3]
        self.cube.right_mtx = curr_state[4]
        self.cube.back_mtx = curr_state[5]
        self.cube.identify_move(move)
        return self.cube.cube_array

    def apply_move2(self, move, curr_state):
        self.cube.cube_array = copy.deepcopy(curr_state)
        self.cube.up_mtx = self.cube.cube_array[0]
        self.cube.front_mtx = self.cube.cube_array[1]
        self.cube.low_mtx = self.cube.cube_array[2]
        self.cube.left_mtx = self.cube.cube_array[3]
        self.cube.right_mtx = self.cube.cube_array[4]
        self.cube.back_mtx = self.cube.cube_array[5]
        self.cube.identify_move(move)
        return self.cube.cube_array

    def Best_First_Search(self, heuristic):
        self.visited = set()

        pq = PriorityQueue()
        source = Node_BFS(self.cube.cube_array)
        target = Node_BFS(self.solved_cube)
        source.heuristic_value = heuristic(source, target)
        pq.put(source)

        while not pq.empty():
            current_node = pq.get()
            if current_node.curr_state == target.curr_state:
                print("Solved!")
                return len(current_node.path), current_node.path

            curr_state_str = str(current_node.curr_state)
            for move in self.moves:
                new_state = self.apply_move2(move, current_node.curr_state)
                new_state_str = str(new_state)
                if new_state_str not in self.visited:
                    new_node = Node_BFS(new_state)
                    new_node.heuristic_value = heuristic(new_node, target)
                    new_node.path = current_node.path + [move]
                    pq.put(new_node)
                    new_node = str(new_node)
                    self.visited.add(new_node)

        return None

    def a_star(self, heuristic):
        self.visited = set()

        pq = PriorityQueue()
        source = Node_AStar(self.cube.cube_array)
        target = Node_AStar(self.solved_cube)
        source.heuristic_value = heuristic(source, target)
        source.distance = 0
        pq.put(source)

        while not pq.empty():
            current_node = pq.get()
            if current_node.curr_state == target.curr_state:
                return current_node.distance, current_node.path

            curr_state_str = str(current_node.curr_state)
            if curr_state_str not in self.visited:
                self.visited.add(curr_state_str)
                for move in self.moves:
                    new_state = self.apply_move2(move, current_node.curr_state)
                    new_node = Node_AStar(new_state)
                    new_node.distance = current_node.distance + 1
                    new_node.heuristic_value = heuristic(new_node, target)
                    new_node.path = current_node.path + [move]
                    pq.put(new_node)
        return None

    def ida_star(self, heuristic):
        source = Node_AStar(self.cube.cube_array)
        target = Node_AStar(self.solved_cube)
        source.heuristic_value = heuristic(source, target)
        threshold = source.heuristic_value

        while threshold != float('inf'):
            min_cost, path = self.dls(source, target, threshold, heuristic)
            if path is not None:
                return len(path), path
            threshold = min_cost
        return None

    def dls(self, source, target, threshold, heuristic):
        stack = [(source, 0)]
        min_cost = float('inf')

        while stack:
            node, cost = stack.pop()
            total_cost = cost + heuristic(node, target)

            if total_cost > threshold:
                min_cost = min(min_cost, total_cost)
            elif node.curr_state == target.curr_state:
                return cost, node.path
            else:
                for move in self.moves:
                    new_state = self.apply_move2(move, node.curr_state)
                    new_node = Node_AStar(new_state)
                    new_node.distance = node.distance + 1
                    new_node.heuristic_value = heuristic(new_node, target)
                    new_node.path = node.path + [move]
                    stack.append((new_node, cost + 1))
        return min_cost, None
