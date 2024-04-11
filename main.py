import time
from Heuristics import Heuristics
from RubikCube import Cube
from RubikSolver import Solver


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
            break
        elif choice == '4':
            break
        else:
            print("Opción no válida. Por favor, elige una opción del 1 al 4.")


menu()
