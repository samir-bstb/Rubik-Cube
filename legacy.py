# Esta es el primer archivo del proyecto donde se hicieron todos los primeros commits
# El proyecto finalizado y funcionando se encuentra en Rubik-Cube/Proyecto/
# Este código es solamente para evidencia, no es el proyecto finalizado

import numpy as np
import random
import math
import copy
from queue import Queue
from queue import PriorityQueue
import time


class Cube:
    def __init__(self):
        # First we initialize the cube with the initial state
        self.front_mtx = [
            [0, 0, 0],  # white
            [0, 0, 0],
            [0, 0, 0]
        ]

        self.up_mtx = [
            [1, 1, 1],  # Orange
            [1, 1, 1],
            [1, 1, 1]
        ]

        self.low_mtx = [
            [2, 2, 2],  # Red
            [2, 2, 2],
            [2, 2, 2]
        ]

        self.left_mtx = [
            [3, 3, 3],  # Green
            [3, 3, 3],
            [3, 3, 3]
        ]

        self.right_mtx = [
            [4, 4, 4],  # Blue
            [4, 4, 4],
            [4, 4, 4]
        ]

        self.back_mtx = [
            [5, 5, 5],  # Yellow
            [5, 5, 5],
            [5, 5, 5]
        ]

        self.cube_array = [self.up_mtx, self.front_mtx, self.low_mtx, self.left_mtx, self.right_mtx, self.back_mtx]

    def rotate_clockwise(self, matrix):
        return [list(reversed(col)) for col in zip(*matrix)]

    def rotate_counterclockwise(self, matrix):
        transposed = [list(col) for col in zip(*matrix)]
        return transposed[::-1]

    def print_mtx(self, mtx):
        for i in mtx:
            for j in i:
                print(f"[{j}]", end=" ")
            print()
        print()

    def print_cube(self):
        self.print_mtx(self.up_mtx)
        self.print_mtx(self.front_mtx)
        self.print_mtx(self.low_mtx)
        self.print_mtx(self.left_mtx)
        self.print_mtx(self.right_mtx)
        self.print_mtx(self.back_mtx)

    def adjust(self):
        self.cube_array[0] = self.up_mtx
        self.cube_array[1] = self.front_mtx
        self.cube_array[2] = self.low_mtx
        self.cube_array[3] = self.left_mtx
        self.cube_array[4] = self.right_mtx
        self.cube_array[5] = self.back_mtx

    def print_arr(self):
        print(self.cube_array)

    # All 12 moves are defined in the following functions
    def move_F(self):
        front_mtx_copy = [row[:] for row in self.front_mtx]
        self.left_mtx = self.rotate_clockwise(self.left_mtx)
        for i in range(3):
            self.front_mtx[i][0] = self.up_mtx[i][0]
            self.up_mtx[i][0] = self.back_mtx[i][0]
            self.back_mtx[i][0] = self.low_mtx[i][0]
            self.low_mtx[i][0] = front_mtx_copy[i][0]
        self.adjust()

    def move_R(self):
        right_mtx_copy = [row[:] for row in self.right_mtx]
        back_mtx_copy = [row[:] for row in self.back_mtx]
        self.low_mtx = self.rotate_clockwise(self.low_mtx)
        for i in range(3):
            self.right_mtx[2][i] = self.front_mtx[2][i]
            self.front_mtx[2][i] = self.left_mtx[2][i]
            self.left_mtx[2][i] = back_mtx_copy[0][2 - i]
            self.back_mtx[0][i] = right_mtx_copy[2][2 - i]
        self.adjust()

    def move_U(self):
        left_mtx_copy = [row[:] for row in self.left_mtx]
        up_mtx_copy = [row[:] for row in self.up_mtx]
        right_mtx_copy = [row[:] for row in self.right_mtx]
        self.front_mtx = self.rotate_clockwise(self.front_mtx)
        for i in range(3):
            self.up_mtx[2][i] = left_mtx_copy[2 - i][2]
            self.right_mtx[i][0] = up_mtx_copy[2][i]
            self.left_mtx[i][2] = self.low_mtx[0][i]
            self.low_mtx[0][i] = right_mtx_copy[2 - i][0]
        self.adjust()

    def move_B(self):
        front_mtx_copy = [row[:] for row in self.front_mtx]
        self.right_mtx = self.rotate_clockwise(self.right_mtx)
        for i in range(3):
            self.front_mtx[i][2] = self.low_mtx[i][2]
            self.low_mtx[i][2] = self.back_mtx[i][2]
            self.back_mtx[i][2] = self.up_mtx[i][2]
            self.up_mtx[i][2] = front_mtx_copy[i][2]
        self.adjust()

    def move_L(self):
        back_mtx_copy = [row[:] for row in self.back_mtx]
        left_mtx_copy = [row[:] for row in self.left_mtx]
        self.up_mtx = self.rotate_clockwise(self.up_mtx)
        for i in range(3):
            self.left_mtx[0][i] = self.front_mtx[0][i]
            self.front_mtx[0][i] = self.right_mtx[0][i]
            self.right_mtx[0][i] = back_mtx_copy[2][2 - i]
            self.back_mtx[2][i] = left_mtx_copy[0][2 - i]
        self.adjust()

    def move_D(self):
        low_mtx_copy = [row[:] for row in self.low_mtx]
        up_mtx_copy = [row[:] for row in self.up_mtx]
        self.back_mtx = self.rotate_clockwise(self.back_mtx)
        for i in range(3):
            self.up_mtx[0][i] = self.right_mtx[i][2]
            self.right_mtx[i][2] = low_mtx_copy[2][2 - i]
            self.low_mtx[2][i] = self.left_mtx[i][0]
            self.left_mtx[i][0] = up_mtx_copy[0][2 - i]
        self.adjust()

    def move_G(self):  # F'
        front_mtx_copy = [row[:] for row in self.front_mtx]
        self.left_mtx = self.rotate_counterclockwise(self.left_mtx)
        for i in range(3):
            self.front_mtx[i][0] = self.low_mtx[i][0]
            self.low_mtx[i][0] = self.back_mtx[i][0]
            self.back_mtx[i][0] = self.up_mtx[i][0]
            self.up_mtx[i][0] = front_mtx_copy[i][0]
        self.adjust()

    def move_S(self):  # R'
        back_mtx_copy = [row[:] for row in self.back_mtx]
        left_mtx_copy = [row[:] for row in self.left_mtx]
        self.low_mtx = self.rotate_counterclockwise(self.low_mtx)
        for i in range(3):
            self.left_mtx[2][i] = self.front_mtx[2][i]
            self.front_mtx[2][i] = self.right_mtx[2][i]
            self.right_mtx[2][i] = back_mtx_copy[0][2 - i]
            self.back_mtx[0][i] = left_mtx_copy[2][2 - i]
        self.adjust()

    def move_W(self):  # U'
        up_mtx_copy = [row[:] for row in self.up_mtx]
        low_mtx_copy = [row[:] for row in self.low_mtx]
        self.front_mtx = self.rotate_counterclockwise(self.front_mtx)
        for i in range(3):
            self.up_mtx[2][i] = self.right_mtx[i][0]
            self.right_mtx[i][0] = low_mtx_copy[0][2 - i]
            self.low_mtx[0][i] = self.left_mtx[i][2]
            self.left_mtx[i][2] = up_mtx_copy[2][2 - i]
        self.adjust()

    def move_V(self):  # B'
        front_mtx_copy = [row[:] for row in self.front_mtx]
        self.right_mtx = self.rotate_counterclockwise(self.right_mtx)
        for i in range(3):
            self.front_mtx[i][2] = self.up_mtx[i][2]
            self.up_mtx[i][2] = self.back_mtx[i][2]
            self.back_mtx[i][2] = self.low_mtx[i][2]
            self.low_mtx[i][2] = front_mtx_copy[i][2]
        self.adjust()

    def move_I(self):  # L'
        back_mtx_copy = [row[:] for row in self.back_mtx]
        right_mtx_copy = [row[:] for row in self.right_mtx]
        self.up_mtx = self.rotate_counterclockwise(self.up_mtx)
        for i in range(3):
            self.right_mtx[0][i] = self.front_mtx[0][i]
            self.front_mtx[0][i] = self.left_mtx[0][i]
            self.left_mtx[0][i] = back_mtx_copy[2][2 - i]
            self.back_mtx[2][i] = right_mtx_copy[0][2 - i]
        self.adjust()

    def move_O(self):  # D'
        left_mtx_copy = [row[:] for row in self.left_mtx]
        right_mtx_copy = [row[:] for row in self.right_mtx]
        self.back_mtx = self.rotate_counterclockwise(self.back_mtx)
        for i in range(3):
            self.right_mtx[i][2] = self.up_mtx[0][i]
            self.left_mtx[i][0] = self.low_mtx[2][i]
            self.up_mtx[0][i] = left_mtx_copy[2 - i][0]
            self.low_mtx[2][i] = right_mtx_copy[2 - i][2]
        self.adjust()

    # changes the style of the cube depending on the moves sent by the user
    def shuffle(self, arr):
        for move in arr:
            self.identify_move(move)

    def identify_move(self, move):
        if move == 'F':
            self.move_F()
        elif move == 'R':
            self.move_R()
        elif move == 'U':
            self.move_U()
        elif move == 'B':
            self.move_B()
        elif move == 'L':
            self.move_L()
        elif move == 'D':
            self.move_D()
        elif move == 'G':  # F'
            self.move_G()
        elif move == 'S':  # R'
            self.move_S()
        elif move == 'W':  # U'
            self.move_W()
        elif move == 'V':  # B'
            self.move_V()
        elif move == 'I':  # L'
            self.move_I()
        elif move == 'O':  # D'
            self.move_O()
        else:
            print(f"Invalid move: {move}")

    # changes the style of the cube randomly
    def auto_shuffle(self, limit):
        moves_list = [self.move_F, self.move_R, self.move_U, self.move_B, self.move_L, self.move_D,
                      self.move_G, self.move_S, self.move_W, self.move_V, self.move_I, self.move_O]
        calls = 0
        while calls < limit:
            move = random.choice(moves_list)
            print("move : ", move)
            move()
            calls += 1

    def encode_cube(self):
        cubo = self.cube_array
        return self.encode(cubo)

    def encode(self, arr):
        int_arr = []
        cont = 0
        x = np.int64(0)

        for c in arr:
            for i in range(3):
                for j in range(3):
                    x = x | (1 << cont * 6 + c[i][j])
                    cont += 1
            int_arr.append(x)
            cont = 0
            x = np.int64(0)

        return int_arr

    def decode(self, arr):
        cubo = []
        for element in arr:
            decoded_mtx = [[0] * 3 for _ in range(3)]
            mask = 2 ** 6 - 1
            for i in range(3):
                for j in range(3):
                    color = int(math.log2(element & mask))
                    decoded_mtx[i][j] = color
                    element >>= 6
            cubo.append(decoded_mtx)
        return cubo


class Node:
    def __init__(self, curr_state):
        self.curr_state = curr_state
        self.path = []


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
        self.Q.put(cubie)
        self.visited.append(cubie)

        print("bfs start", cubie)
        while not self.Q.empty():
            cubie = self.Q.get()
            if cubie == self.solved_cube2:
                print("Solved")
                return cubie

            for move in self.moves:
                curr_state = self.cube.decode(cubie)
                next_state = copy.deepcopy(curr_state)
                next_state = self.apply_move(move, next_state)
                next_state = self.cube.encode(next_state)
                if next_state not in self.visited:
                    self.Q.put(next_state)
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
                return current_node.path

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


class Heuristics:
    @staticmethod
    def misplaced_pieces_heuristic(node, target):
        misplaced_pieces = 0
        for i in range(6):
            for j in range(3):

                for k in range(3):
                    if node.curr_state[i][j][k] != target.curr_state[i][j][k]:
                        misplaced_pieces += 1
        return misplaced_pieces

    @staticmethod
    def stickers_out_pos(source, target):
        total_distance = 0
        for source_face, target_face in zip(source.curr_state, target.curr_state):
            face_distance = sum(1 for s, t in zip(source_face, target_face) if s != t)
            total_distance += face_distance
        return total_distance

    @staticmethod
    def manhattan_distance_heuristic(node, target):
        distance = 0
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    target_position = Heuristics.find_position(target.curr_state, node.curr_state[i][j][k])
                    distance += abs(i - target_position[0]) + abs(j - target_position[1]) + abs(k - target_position[2])
        return distance

    @staticmethod
    def find_position(state, piece):
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    if state[i][j][k] == piece:
                        return (i, j, k)


def menu():
    c = Cube()
    s = Solver(c.cube_array)
    while True:
        print("1. Shuffle manual")
        print("2. Shuffle aleatorio")
        print("3. Resolver el cubo")
        print("4. Salir")
        choice = input("Elige una opción: ")

        if choice == '1':
            moves = input("Introduce los movimientos que quieres hacer (separados por espacios): ").split()
            c.shuffle(moves)
            c.print_cube()
        elif choice == '2':
            moves = int(input("Introduce cuantos movimientos quieres hacer: "))
            c.auto_shuffle(moves)
            c.print_cube()
        elif choice == '3':
            print("1. BFS")
            print("2. A*")
            print("3. Best First Search")
            print("4. IDA*")
            algorithm = input("Elige un algoritmo: ")
            if algorithm in ['2', '3', '4']:
                print("1. Misplaced pieces heuristic")
                print("2. Stickers out of position")
                print("3. Manhattan distance heuristic")  # Nueva opción para la heurística de Manhattan
                heuristic_choice = input("Elige una heurística: ")
                if heuristic_choice == '1':
                    heuristic = Heuristics.misplaced_pieces_heuristic
                elif heuristic_choice == '2':
                    heuristic = Heuristics.stickers_out_pos
                elif heuristic_choice == '3':  # Manejar la nueva opción
                    heuristic = Heuristics.manhattan_distance_heuristic
            start_time = time.time()
            if algorithm == '1':
                s.bfs()
            elif algorithm == '2':
                print(s.a_star(heuristic))
            elif algorithm == '3':
                print(s.Best_First_Search(heuristic))
            elif algorithm == '4':
                print(s.ida_star(heuristic))
            end_time = time.time()
            print("Tiempo de ejecución: ", end_time - start_time, "segundos")
        elif choice == '4':
            break
        else:
            print("Opción no válida. Por favor, elige una opción del 1 al 4.")


menu()
