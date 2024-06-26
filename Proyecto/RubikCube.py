import numpy as np
import random
import math


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