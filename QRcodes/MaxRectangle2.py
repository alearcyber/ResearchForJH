# Python3 code for Maximum size
# square sub-matrix with all 1s

def findMaxSubSquare(M):
    R = len(M)  # no. of rows in M[][]
    C = len(M[0])  # no. of columns in M[][]

    S = []
    for i in range(R):
        temp = []
        for j in range(C):
            if i == 0 or j == 0:
                temp += M[i][j],
            else:
                temp += 0,
        S += temp,
    # here we have set the first row and first column of S same as input matrix, other entries are set to 0

    # Update other entries
    for i in range(1, R):
        for j in range(1, C):
            if (M[i][j] == 1):
                S[i][j] = min(S[i][j - 1], S[i - 1][j],
                              S[i - 1][j - 1]) + 1
            else:
                S[i][j] = 0

    # Find the maximum entry and
    # indices of maximum entry in S[][]
    max_of_s = S[0][0]
    max_i = 0
    max_j = 0
    for i in range(R):
        for j in range(C):
            if (max_of_s < S[i][j]):
                max_of_s = S[i][j]
                max_i = i
                max_j = j

    points_in_rectangle = []
    for y in range(max_i, max_i - max_of_s, -1):
        for x in range(max_j, max_j - max_of_s, -1):
            location = (x, y)
            points_in_rectangle.append(location)

    return points_in_rectangle


def verboseFind(matrix):
    """ does some fancy printing """
    results = findMaxSubSquare(matrix)
    top_left = results[-1]
    bottom_right = results[0]

    width = len(matrix[0])
    height = len(matrix)

    print('--Your input Matrix--')
    for row in matrix:
        for bit in row:
            print(bit, end='  ')
        print()

    print('\n--Largest Area Sub-matrix--')
    for y in range(height):
        for x in range(width):
            print(1 if (x, y) in results else 0, end='  ')
        print()

    print(f'Top Left Corner =', top_left)
    print(f'Bottom Right Corner =', bottom_right)
    print(f'Area =', len(results))







if __name__ == '__main__':
    """ entry point """

    # input matrix
    M1 = [[0, 1, 1, 0, 1],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]]

    M2 = [[0, 1, 1, 0, 1],
         [1, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]]

    M3 = [[0, 1, 1, 0, 1],
         [1, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]]

    verboseFind(M1)
    verboseFind(M2)
    verboseFind(M3)
