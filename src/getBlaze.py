import numpy as np

def getBlaze(img, T, ori):
    if(ori == 'up'):
        x_axis = 1
        y_axis = 0
    elif(ori == 'down'):
        x_axis = -1
        y_axis = 0
    elif(ori == 'left'):
        x_axis = 0
        y_axis = 1
    elif(ori == 'right'):
        x_axis = 0
        y_axis = -1
    elif(ori == 'leftup'):
        x_axis = 1
        y_axis = 1
    elif(ori == 'leftdown'):
        x_axis = -1
        y_axis = 1
    elif(ori == 'rightup'):
        x_axis = 1
        y_axis = -1
    elif(ori == 'rightdown'):
        x_axis = -1
        y_axis = -1

    fi = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Equ = (-abs(x_axis)) * i + (-abs(y_axis)) * j;
            fi[i, j] = (np.mod(Equ, T))

    if x_axis == 1:
        fi = np.flip(fi, 0)

    if y_axis == 1:
        fi = np.flip(fi, 1)

    fi = fi * 2 * np.pi / T
    fi = np.mod(fi, 2 * np.pi)


    return fi

    

