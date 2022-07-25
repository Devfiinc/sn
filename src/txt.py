
import numpy as np


inputs = np.array([[3, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [7, 2, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

kernel = np.array([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


input_shape = (28, 28)

def convolution_matrix(m, k):
    mr, mc = m[0], m[1] # matrix rows, cols
    kr, kc = len(k), len(k[0]) # kernel rows, cols

    padding = 0
    stride = 1

    fr = int(((mr - kr + 2.0 * padding) / stride) + 1);
    fc = int(((mc - kc + 2.0 * padding) / stride) + 1);


    print(mr, mc)
    print(kr, kc)
    print(fr, fc)

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

    #print(cmat)
    return cmat
    

cmat = convolution_matrix(input_shape, kernel)
print("cmat", cmat.shape)


inp1 = inputs.flatten()
inp = np.zeros((inp1.shape[0], 1))
print("inp", inp.shape)

for i in range(len(inp)):
    inp[i] = inp1[i]

#print(inp)

out = np.dot(cmat.T, inp)


print(out.shape)

out1 = out.reshape(input_shape)

print(out1.shape)

print(out1)
#print(out)


#dims = (4, 4)
#
#def tupleman(dim = (0,0)):
#    print(dim[0])
#
#tupleman()
#tupleman(dims)