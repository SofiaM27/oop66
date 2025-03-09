import numpy as np
import os


class Matrix2D:
    def __init__(self, a11, a12, a21, a22):
        self.matrix = np.array([[a11, a12], [a21, a22]])

    def determinant(self):
        return np.linalg.det(self.matrix)

    def is_singular(self):
        return np.isclose(self.determinant(), 0)

    def __str__(self):
        return f"[{self.matrix[0, 0]} {self.matrix[0, 1]}]\n[{self.matrix[1, 0]} {self.matrix[1, 1]}]"

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            matrices = []
            for line in lines:
                values = list(map(float, line.split()))
                if len(values) == 4:
                    matrices.append(Matrix2D(*values))
            return matrices


class Vector2D:
    def __init__(self, x, y):
        self.vector = np.array([x, y])

    def __str__(self):
        return f"[{self.vector[0]}, {self.vector[1]}]"

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            vectors = []
            for line in lines:
                values = list(map(float, line.split()))
                if len(values) == 2:
                    vectors.append(Vector2D(*values))
            return vectors


class Solver:
    @staticmethod
    def solve(matrix, vector):
        if matrix.is_singular():
            return "Система не має розв'язку (вироджена матриця)."
        det_A = matrix.determinant()
        A1 = Matrix2D(vector.vector[0], matrix.matrix[0, 1], vector.vector[1], matrix.matrix[1, 1])
        A2 = Matrix2D(matrix.matrix[0, 0], vector.vector[0], matrix.matrix[1, 0], vector.vector[1])
        x1 = A1.determinant() / det_A
        x2 = A2.determinant() / det_A
        return Vector2D(x1, x2)


if __name__ == "__main__":
    input_matrix_file = "matrix_coefficients.txt"
    input_vector_file = "rhs_values.txt"
    output_file = "solutions.txt"

    if not os.path.exists(input_matrix_file) or not os.path.exists(input_vector_file):
        print("Файл(и) не знайдено!")
    else:
        matrices = Matrix2D.from_file(input_matrix_file)
        vectors = Vector2D.from_file(input_vector_file)

        with open(output_file, "w") as out:
            for i in range(min(len(matrices), len(vectors))):
                solution = Solver.solve(matrices[i], vectors[i])
                result = f"Система {i + 1}: {solution}\n"
                print(result)
                out.write(result)