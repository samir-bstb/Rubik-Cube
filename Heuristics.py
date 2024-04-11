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