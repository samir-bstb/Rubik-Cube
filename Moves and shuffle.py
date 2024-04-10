import numpy as np
import random
import math
import copy
from queue import Queue
from queue import PriorityQueue

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
        self.cube.up_mtx =  self.cube.cube_array[0]
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
        
        self.moves = ['F', 'R', 'U', 'B', 'L', 'D', 'G', 'S', 'W', 'V', 'I', 'O']
        self.Q = Queue()
        self.visited = [] #for breadth first search
        self.visited = set() #for A* and BFS 

    def bfs(self):
        solution = self.__bfs(self.cube.cube_array)
        if solution is None:
          print("No solution found")
        else:
          print(solution)
          
    def __bfs(self, start):
      self.visited = []
      self.Q.put(start)
      self.visited.append(start)

      print("bfs start", start)
      while not self.Q.empty():
          curr_state = self.Q.get()
          if curr_state == self.solved_cube:
              print("Solved")
              return curr_state

          for move in self.moves:
              next_state = copy.deepcopy(curr_state)
              next_state = self.apply_move(move, next_state)
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

    #A star implementation
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

    #Best First Search
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

    #Heuristics
    '''
    def manhattan_distance(node, target):#no es optima
        state = node.curr_state
        distance = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 1 and j == 1 and k == 1:
                        continue  # Ignorar el centro, ya que no es una esquina
                    if state[i][j][k] != target.curr_state[i][j][k]:
                        # Buscar la posición correcta de esta esquina
                        for x in range(3):
                            for y in range(3):
                                for z in range(3):
                                    if state[i][j][k] == target.curr_state[x][y][z]:
                                        # Calcular la distancia Manhattan para esta esquina
                                        distance += abs(i - x) + abs(j - y) + abs(k - z)
                                        break  # Una vez encontrada la posición correcta, salir del bucle
        return distance'''

    def misplaced_pieces_heuristic(node, target):
        misplaced_pieces = 0
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    if node.curr_state[i][j][k] != target.curr_state[i][j][k]:
                        misplaced_pieces += 1
        return misplaced_pieces

    
c = Cube()
#arr = ['I', 'D', 'F']
#c.shuffle(arr)
c.auto_shuffle(5)
c.print_cube()
print()
c.print_arr()

#result = c.encode_cube() 
#print(result)
#s = c.decode(result)
#print(s)

s = Solver(c.cube_array)
#s.bfs()
#print(s.a_star(Solver.manhattan_heuristic))
#print(s.a_star(Solver.misplaced_pieces_heuristic))
#print(s.Best_First_Search(Solver.manhattan_heuristic))
print(s.Best_First_Search(Solver.misplaced_pieces_heuristic))
