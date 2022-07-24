
import numpy as np


inputs = np.array([[3, 3, 3, 6],
                   [7, 4, 3, 2],
                   [7, 2, 8, 5],
                   [1, 7, 3, 1]])

kernel = np.array([[1, 2, 4], 
                   [1, 1, 3], 
                   [1, 2, 4]])


def convolution_matrix(m, k):
    mr, mc = len(m), len(m[0]) # matrix rows, cols
    kr, kc = len(k), len(k[0]) # kernel rows, cols

    padding = 0
    stride = 1

    fr = int(((mr - kr + 2.0 * padding) / stride) + 1);
    fc = int(((mc - kc + 2.0 * padding) / stride) + 1);

    cmat = np.zeros((fr*fc, mr*mc))


    d1 = 0
    d2 = 0
    for i in range(len(cmat)):
        if i%fc == 0 and i > 0:
            d1 += 1
            d2 = 0

        for r in range(kr):
            for c in range(kc):
                cmat[i][d1*mc + d2 + r*mc + c] = k[r][c]
        d2 += 1

    print(cmat)
    

#convolution_matrix(inputs, kernel)


dims = (4, 4)

def tupleman(dim = (0,0)):
    print(dim[0])

tupleman()
tupleman(dims)