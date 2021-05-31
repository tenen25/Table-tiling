import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
from numpy import linalg as LA
import cmath

gap=4

a = 0   #initialize letter values for plotting in matplot
b = 64
c = 128
d = 192

beta = -128    #assigned for color plotting
alpha = 128
gamma = 32

tile1 = np.array([[a,c], [c,a]]) #first possible starting tile
tile2 = np.array([[b,d], [d,b]]) #second posiible starting tile
tile3 = np.array( [ [a] ] )
tile4 = np.array( [[b]] )
tile5 = np.array( [[c]] )
tile6 = np.array( [[d]] )

def Suffix(num):
    if num<0 or not isinstance(num,int):
        suffix = "!NO!"
    elif num>=10 and num<21:
        suffix="th"
    else:
        while(num>=10):
            num=int(num/10)
        if num == 1: suffix = "st"
        elif num == 2: suffix = "nd"
        elif num == 3: suffix = "rd"
        else: suffix = "th"
    return suffix

def print_mat(mat, size):               # prints 2d array in a more pleasing manner
    for i in range(size):
        print(mat[i])

def save_arr(num_arr, str):
    file_str = str=".txt"
    a_file = open(file_str, "w")
    for row in num_arr:
        np.savetxt(file_str, row)
    a_file.close()

def BuiltTile(num, type):       #Generates the num-th iteration of the table tiling patch according to initial patch
    if type == 1:
        table = tile1
        n = 2
    elif type == 2:
        table = tile2
        n = 2
    else:
        n = 1
        if type == 3: table = tile3
        elif type == 4: table = tile4
        elif type == 5: table = tile5
        elif type == 6: table = tile6
    wid = 2**num
    newTable = [['e' for x in range(wid)] for y in range(wid)]
    a_count = 0
    b_count = 0
    c_count = 0
    d_count = 0
    while n < wid:
        for k in range(n):
            for l in range(n):
                if table[k][l] == a:
                    newTable[2*k][2*l] = b
                    newTable[2*k][2*l +1] = a
                    newTable[2*k + 1][2*l +1] = a
                    newTable[2*k + 1][2*l] = d
                    if n*2 == wid :
                      a_count = a_count + 2
                      b_count = b_count + 1
                      d_count = d_count + 1
                elif table[k][l] == b:
                    newTable[2*k][2*l] = a
                    newTable[2*k][2*l +1] = c
                    newTable[2*k + 1][2*l +1] = b
                    newTable[2*k + 1][2*l] = b
                    if n*2 == wid :
                      a_count = a_count + 1
                      b_count = b_count + 2
                      c_count = c_count + 1
                elif table[k][l] == c:
                    newTable[2*k][2*l] = c
                    newTable[2*k][2*l +1] = b
                    newTable[2*k + 1][2*l +1] = d
                    newTable[2*k + 1][2*l] = c
                    if n*2 == wid :
                      b_count = b_count + 1
                      c_count = c_count + 2
                      d_count = d_count + 1
                elif table[k][l] == d:
                    newTable[2*k][2*l] = d
                    newTable[2*k][2*l +1] = d
                    newTable[2*k + 1][2*l +1] = c
                    newTable[2*k + 1][2*l] = a
                    if n*2 == wid :
                      a_count = a_count + 1
                      c_count = c_count + 1
                      d_count = d_count + 2
        table = newTable.copy()
        n = n*2
    str(itera) + Suffix(itera) + " iterated tile corresponding to Tile type " + str(tileType)
    print("The staistics for the "+str(num) +Suffix(num) +" iterated tile corresponding to tile"\
     +str(type)  +" is as follows:")
    print("Number of 'a' tiles is " + str(a_count))
    print("Number of 'b' tiles is " + str(b_count))
    print("Number of 'c' tiles is " + str(c_count))
    print("Number of 'd' tiles is " + str(d_count))
    return newTable

def gen_tile(num):
    if num == 1:
        start = 3
    else:
        start =1
    for j in range(start,7):
        lett="No"
        if j>2:
            if (j-2)==1: lett = "a"
            elif (j-2)==2: lett = "b"
            elif (j-2)==3: lett = "c"
            else: lett = "d"
            file_name = str(num)+Suffix(num)+" iteration,"\
            +lett+" starting tile"
        else:
            lett = str(j)+Suffix(j)
            file_name = str(num) + Suffix(num) + " iteration," \
                        + lett + " starting tile"
        tile = BuiltTile(num, j)
        np.savetxt(file_name+".txt", tile, delimiter=",")

def print_tile(num):
    if num == 1:
        start = 3
    else:
        start =1
    fig, axs = plt.subplots(3, 2)
    fig.suptitle(str(num) +Suffix(num)+" iterated tiles")
    for j in range(start,7):
        lett="No"
        if j>2:
            if (j-2)==1: lett = "a"
            elif (j-2)==2: lett = "b"
            elif (j-2)==3: lett = "c"
            else: lett = "d"
            read_name = str(num)+Suffix(num)+" iteration,"\
            +lett+" starting tile"
        else:
            lett = str(j)+Suffix(j)
            read_name = str(num) + Suffix(num) + " iteration," \
                        + lett + " starting tile"
        tile = np.loadtxt(read_name+".txt", delimiter=",")
        axs[ (j-j%2)/2 ,j%2].plt.imshow(tile, vmin=0, vmax=255)
    plt.show()




def phase(x, y):                #Converts a float array of length 2 to complex array of length 2
    xPhas = complex(0, x )
    yPhas = complex(0, y )
    return  [xPhas, yPhas]

def Op_Mat(num, patch, phase):     #Generates the Schroedinger operator into a square matrix
    wid= 2**(num)           #wid denotes width of periodic patch
    size= wid**2            #size generates the dimension of the operator matrix
    NewMat = [[0 for j in range(size)] for i in range(size)]
    for i in range(wid):
        for j in range(wid):
            NewMat[i * wid + j][((i - 1) % wid) * wid + j] = 1 if ((i - 1) >= 0) else\
                cmath.exp( phase[1]    )

            NewMat[i * wid + j][((i + 1) % wid) * wid + j] = 1 if ((i + 1) < wid) else \
                cmath.exp(- phase[1])

            NewMat[i * wid + j][i * wid + (j + 1) % wid] = 1 if (j + 1 < wid) else \
                cmath.exp( phase[0])

            NewMat[i * wid + j][i * wid + (j - 1) % wid] = 1 if (j - 1 >= 0) else \
                cmath.exp(- phase[0])
            NewMat[i * wid + j][i* wid + j] = SubDiag(patch[i][j])
    return NewMat

def Op_Mat_NoPhase(num, patch):     #Generates the Schroedinger operator into a square matrix
    wid= 2**(num)           #wid denotes width of periodic patch
    size= wid**2            #size generates the dimension of the operator matrix
    NewMat = [[0 for j in range(size)] for i in range(size)]
    for i in range(wid):
        for j in range(wid):
            if i != wid-1:
                NewMat[i*wid+j][(i+1)*wid+j] = 1
            if j != wid-1:
                NewMat[i*wid+j][i*wid+j+1] = 1
            if j != 0:
                NewMat[i*wid+j][i*wid+j-1] = 1
            if i != 0:
                NewMat[i*wid+j][(i-1)*wid+j] = 1
            NewMat[i * wid + j][i* wid + j] = SubDiag(patch[i][j])
    return NewMat

def Op_Mat_Phase(num, phase):     #Generates the Schroedinger operator into a square matrix
    wid= 2**(num)           #wid denotes width of periodic patch
    size= wid**2            #size generates the dimension of the operator matrix
    NewMat = [[0 for j in range(size)] for i in range(size)]
    for j in range(wid):
        NewMat[0*wid+j][wid*(wid-1)+j] = cmath.exp( phase[1]    )
        NewMat[(wid-1)*wid+j][0*wid+j] =  cmath.exp(- phase[1])
    for i in range(wid):
        NewMat[i*wid+0][i*wid+ wid-1] = cmath.exp(- phase[0])
        NewMat[i*wid+ wid-1][i*wid+0] = cmath.exp( phase[0])
    return NewMat

def SubDiag(lett):              #assign diagonal values to OpMat based on the tiling
    return{
        a : val1,
        b : val2,
        c : val3,
        d : val4
    }.get(lett, 0)

def tile_print( tileType, plotNum ):        #print the tile matrix in
    for j in range(plotNum+1):
        plt.subplot(3, 3, j)
        M = BuiltTile(j+1, tileType)
        plt.imshow(M, vmin=0, vmax=255)
    plt.show()

def sample_numth_eig(itera, res, num, tileType): # iter is the iteration number, res is the sampling resolution, num is the eigenvlaue number
    x = np.linspace(-cmath.pi, cmath.pi, res+1 )
    y = np.linspace(-cmath.pi, cmath.pi, res+1 )
    size = 2**(2*itera)
    if num>size or num<0:
        print('num value is incorrect')
        return
    if num > 0:
        eig_mat = [[0 for a in range(res+1)] for b in range(res+1)  ]
    elif num == 0:
        eig_mat = [[0.0 for a in range(size)  ] for s in range(2)]
        #[ [[0 for a in range(res+1)] for b in range(res+1)  ] for j in range(size) ]
    Tile = BuiltTile(itera,tileType)
    for k in range(res+1):
        for l in range(res+1):
            mat = Op_Mat(itera, Tile, phase(x[k],y[l]))
            vect=LA.eigvalsh( mat )
            if num != 0:
                eig_mat[k][l] = vect[num - 1]
            elif num == 0:
                for  j in range(size):
                    if k==0 and l==0:
                        eig_mat[0][j] = vect[j]
                        eig_mat[1][j] = vect[j]
                    else:
                        eig_mat[0][j] = max(vect[j] , eig_mat[0][j] )
                        eig_mat[1][j] = min(vect[j], eig_mat[1][j])
    return eig_mat

def sample_numth_eig_new(itera, res, num, tileType): # iter is the iteration number,\
    # res is the sampling resolution, num is the eigenvlaue number
    x = np.linspace(-cmath.pi, cmath.pi, res+1 )
    y = np.linspace(-cmath.pi, cmath.pi, res+1 )
    size = 2**(2*itera)
    if num>size or num<0:
        print('num value is incorrect')
        return
    if num > 0:
        eig_mat = [[0 for a in range(res+1)] for b in range(res+1)  ]
    elif num == 0:
        eig_mat = [[0.0 for a in range(size)  ] for s in range(2)]
        #[ [[0 for a in range(res+1)] for b in range(res+1)  ] for j in range(size) ]
    Tile = BuiltTile(itera,tileType)
    mat1 = np.array(Op_Mat_NoPhase(itera, Tile))
    for k in range(res+1):
        for l in range(res+1):
            mat2 = np.array( Op_Mat_Phase(itera,phase(x[k],y[l])) )
            mat = np.add(mat1, mat2)
            vect=LA.eigvalsh( mat )
            if num != 0:
                eig_mat[k][l] = vect[num - 1]
            elif num == 0:
                for  j in range(size):
                    if k==0 and l==0:
                        eig_mat[0][j] = vect[j]
                        eig_mat[1][j] = vect[j]
                    else:
                        eig_mat[0][j] = max(vect[j] , eig_mat[0][j] )
                        eig_mat[1][j] = min(vect[j], eig_mat[1][j])
                print("("+str(x[k])+","+str(y[l])+")"+" computed")
    return eig_mat

def min_sample(itera, num, tileType):       #smapling algorithm for minimum of points
    size = 2 ** (2 * itera)
    Tile = BuiltTile(itera, tileType)
    mat_0 = Op_Mat_NoPhase(itera, Tile)
    mat_mid = np.add(mat_0, Op_Mat_Phase(itera,  phase(0,0) ) )
    mat_horiz = np.add(mat_0, Op_Mat_Phase(itera,  phase( cmath.pi ,0) ) )
    mat_vert = np.add(mat_0, Op_Mat_Phase(itera, phase(0,cmath.pi)))
    mat_corner = np.add(mat_0, Op_Mat_Phase(itera, phase(cmath.pi  , cmath.pi)))
    vect_mid = LA.eigvalsh( mat_mid )
    vect_horiz = LA.eigvalsh(mat_horiz)
    vect_vert = LA.eigvalsh(mat_vert)
    vect_corner = LA.eigvalsh(mat_corner)
    if num > 0 and num< size+1 :
        eig_mat = [[0.0 for j in range(3)] for i in range(3) ]
        for i in range(3):
            for j in range(3):
                if i==1 and j==1:
                    eig_mat[i][j] = vect_mid[num-1]
                elif (i==1 and j!=1):
                    eig_mat[i][j] = vect_vert[num-1]
                elif (i!=1 and  j==1):
                    eig_mat[i][j] = vect_horiz[num-1]
                else:
                    eig_mat[i][j] = vect_corner[num-1]
        return eig_mat
    elif num==0:
        eig_mat = [[0.0 for j in range(size)] for i in range(2)]    #generate matrix for spectral bands
        for eig_num in range(size):
            eig_mat[0][eig_num]=min( vect_mid[eig_num], vect_horiz[eig_num], vect_vert[eig_num],vect_corner[eig_num] )
            eig_mat[1][eig_num] = max(vect_mid[eig_num], vect_horiz[eig_num], vect_vert[eig_num], vect_corner[eig_num])
        return eig_mat

def plot_mat(mat, res, num):        #Generates a wire frame plot for a 2d array
    x = np.linspace(-cmath.pi, cmath.pi, res + 1)
    y = np.linspace(-cmath.pi, cmath.pi, res + 1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if num<0:
        colors = iter(cm.rainbow(np.linspace(-2, 2, -num )))
    X, Y = np.meshgrid(x, y)
    if num < 0:
        for j in range(-num):
            Z= np.array(mat[j])
            ax.plot_wireframe(X, Y, mat[j],  label=str(j)+'-th eigenvalue.', color=next(colors))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend(fontsize='small', loc=1)
            plt.title("Eigenvalues")
            plt.show()
    else:
        ax.plot_wireframe(X, Y, mat)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #plt.legend([str(num) + '-th eigenvalue.'],  loc="upper left")
        ind_max =  np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        ind_min =  np.unravel_index(np.argmin(mat, axis=None), mat.shape)
        plt.legend( [str( round( mat[ind_min],3)  )+ " to " + str( round(mat[ind_max],3) )] )
        if num == 1:
            suffix = "st"
        elif num == 2:
            suffix = "nd"
        elif num == 3:
            suffix = "rd"
        else:
            suffix = "th"
        plt.title(str(num) +"-"+suffix+ " eigenvalue")
        print("Maximal value is obtained in "+ index_to_loc( ind_max, x, y ) +" and is " + str(round(mat[ind_max],4))+"." )
        print("Minimal value is obtained in " + index_to_loc( ind_min, x, y ) + " and is " + str(round(mat[ind_min],4))+"." )
        plt.show()

def index_to_loc( ind, x, y):               # translated index in array to location on the x-y plane and returns string
    loca = np.array([0.0,0.0])
    loca[0] = x[ind[0]]
    loca[1] = y[ind[1]]
    stri = "("+ str(  round(loca[0] ,4) )+","+str( round(loca[1] ,4))+")"
    return stri

def print_band(size, mat, num):                  #print graph of spectral bands given matrix of maxima and minima
    x = [[0.0 for j in range(2)] for i in range(size)]
    y = [0.0 for i in range(size)]
    if num == 0:
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [i + 1 for j in range(3)]
            plt.plot(x[i], y[i])
        plt.show()
    elif num > 0 and num<=size:
        left = False
        right = False
        left_val = num
        right_val = num
        while not left:
            if left_val == 1:
                #left = True
                break
            left = ( mat[1][left_val-1] > mat[0][left_val-2])
            if left:
                break
            if not left:
                left_val = left_val-1
        while not right:
            if right_val == size:
                #right = True
                break
            right = ( mat[0][right_val-1] < mat[1][right_val])
            if right:
                break
            if not right:
                right_val = right_val+1
        for i in range(left_val-1, right_val):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [i + 1 for j in range(3)]
            plt.plot(x[i], y[i])
        plt.show()

def print_tot_band_min(init_itera, fin_itera):
    plt.xlabel('X-axis')
    plt.ylabel('Iteration number')
    for l in range(init_itera, fin_itera):
        size = 2 ** (2 * l)
        mat = min_sample(l,0)
        x = [[0.0 for j in range(2)] for i in range(size)]
        y = [0.0 for i in range(size)]
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [l for j in range(3)]
            plt.plot(x[i], y[i], color='green')
    plt.show()

def print_tot_band(init_itera, fin_itera, res, tileType):
    plt.xlabel('Spectrum values')
    plt.ylabel('Iteration number')
    plt.title("Tile "+str(tileType-2)+" spectrum, "+ str(init_itera)+"-"+str(fin_itera-1)+" iterations")
    for l in range(init_itera, fin_itera):
        size = 2 ** (2 * l)
        mat = sample_numth_eig_new(l, res, 0, tileType)
        x = [[0.0 for j in range(2)] for i in range(size)]
        y = [0.0 for i in range(size)]
        for i in range(size):
            x[i] = np.linspace(mat[1][i], mat[0][i], 3)
            y[i] = [l for j in range(3)]
            plt.plot(x[i], y[i], color='green')
    plt.show()

def SubColor(num):              #assign colors to tile types based on the number
    return{
        1 : 'blue',
        2 : 'green',
        3 : 'purple',
        4 : 'black'
    }.get(num, 0)

def check_input(itera, res, eig_num, tileType, type):      #checks stages of computation
    size = 2 ** (2 * itera)
    if type=="tile":                        #Check tiling algorithm
        M = BuiltTile(itera, tileType)
        plt.imshow(M, vmin=0, vmax=255)
        plt.title(str(itera)+Suffix(itera)+" iterated tile corresponding to Tile type "+str(tileType))
        plt.show()
    if type == "Tile Print":
        tile_print( tileType, itera )
    if type == "GEN-TILE":                  #Generates different tiles and saves as txt files
        for j in range(itera+1, itera +5):
            gen_tile(j)
    if type=="matrix":                      #Check matrix algorithm
        Tile = BuiltTile(itera, tileType)
        M = Op_Mat(itera, Tile, phase(cmath.pi, -cmath.pi))
        Z= np.array(M, dtype=np.complex)
        P= Z.real
        Im= Z.imag
        print_mat(Z, size)
        plt.imshow(P)
        plt.imshow(Im)
        plt.show()
    if type=="eig":                         #Check eig_numth eigenvalue sampling algorithm
        M = sample_numth_eig_new(itera, res, eig_num, tileType)
        print_mat(M, res + 1)
        print("Minimum is")
        print(np.min(M))
        print("Maximum is")
        print(np.max(M))
    if type=="generate eigen matrix":       #Check eigenvalue sampling algorithm
        M = sample_numth_eig(itera, res, 0)
        print_mat(M, res + 1)
    if type=="EIG PLOT" and eig_num!=0:                #Check eigenvalue plotting algorithm for a single eigenvalue
        M = sample_numth_eig_new(itera, res, eig_num, tileType)
        Z= np.array(M)
        plot_mat(Z, res, eig_num)
    if type == "EIG PLOT" and eig_num == 0:
        size = 2**(2*itera)
        M = sample_numth_eig(itera, res, eig_num, tileType)
        Z = np.array(M)
        plot_mat(Z, res, -size)
    if type == "SPEC BAND":
        size = 2 ** (2 * itera)
        M = sample_numth_eig_new(itera, res, 0, tileType)
        Z = np.array(M)
        print_band(size, Z, eig_num)
    if type == "SPEC TOTAL":
        print_tot_band(itera, itera+4, res, tileType)
    if type == "MIN SPEC TOTAL":
        print_tot_band_min(itera, itera + 6)
    if type == "SPEC TOTAL2":
        for s in range(1, 5):
            colour = SubColor(s)
            for l in range(itera, itera+gap+1):
                size = 2 ** (2 * l)
                mat = sample_numth_eig_new(l, res, 0, s+2)
                x = [[0.0 for j in range(2)] for i in range(size)]
                y = [0.0 for i in range(size)]
                for i in range(size):
                    x[i] = np.linspace(mat[1][i], mat[0][i], 3)
                    y[i] = [l+ s/10 for j in range(3)]
                    plt.plot(x[i], y[i], color= colour)
        title_name="Spectrum of different tile iterations, "+str(itera)+"-"+str(itera+gap)
        plt.title(title_name)
        plt.show()
    if type == "SPEC-BANDS-GEN":
        size = 2 ** (2 * itera)
        for s in range(1, 5):
            mat = sample_numth_eig_new(itera, res, 0, s+2)
            file_name = str(itera)+Suffix(itera)+" spec bands, "\
            + str(s+2)+" tile, diag "+str( round(val1,3) )+" "+str( round(val2,3) )+" " \
            +str( round(val3,3) ) + " " + str( round(val4,3) )
            np.savetxt(file_name+".txt", mat, delimiter=',')
    if type == "SPEC-TOT-PLOT":
        fig = plt.figure()
        ax = fig.add_subplot()
        for s in range(1, 5):
            colour = SubColor(s)
            for l in range(itera, itera+gap+1):
                size = 2 ** (2 * l)
                read_name = str(l)+Suffix(l)+" spec bands, "\
                + str(s+2)+" tile, diag "+str( round(val1,3) )+" "+str( round(val2,3) )+" " \
                +str( round(val3,3) ) + " " + str( round(val4,3) )
                mat = np.loadtxt(read_name+".txt", delimiter=',')
                x = [[0.0 for j in range(2)] for i in range(size)]
                y = [0.0 for i in range(size)]
                for i in range(size):
                    x[i] = np.linspace(mat[1][i], mat[0][i], 3)
                    y[i] = [l+ s/10 for j in range(3)]
                    plt.plot(x[i], y[i], color= colour)
        title_name="Spectrum of different tile iterations, "+str(itera)+"-"+str(itera+gap)\
        +", diag "+str( round(val1,3) )+":"+str( round(val2,3) )+":"\
        + str( round(val3,3) ) + ":" + str( round(val4,3) )
        ax.set_xlabel('Spectrum values')
        ax.set_ylabel('Iteration number')
        plt.title(title_name)
        plt.show()
    if type == "SPEC-STR-MONO":
        true_val = True
        size_1 = 2**(2*itera)
        size_2 = 2**(2*(itera+2))
        file_name = str(itera) + Suffix(itera) + " spec bands, " \
                    + str(s + 2) + " tile, diag " + str(val1) + " " + str(val2) + " " \
                    + str(val3) + " " + str(val4)
        mat_1 = np.loadtxt(read_name+".txt", delimiter=",")
        while(true_val == True):
            for j in range(size_1):
                start = mat_1[1][j]
                end = mat_1[0][j]






    print("End of program test-run.")

#MAIN FUNCTION
#print("Possible check types:")
#print(" *Tile plotting - 'tile' \n *Eigenvalue plotting - 'EIG PLOT' \n *Spectral bands sketch - 'SPEC BAND'")
#type = input("Enter type of check:")              #Type of check in check_input function
type="SELF"
if type!="tile" and type!="SELF":
    print("Please assign potential values to letters")
    val1 = int(input("Assign potential value to letter 'a':"))
    val2 = int(input("Assign potential value to letter 'b':"))
    val3 = int(input("Assign potential value to letter 'c':"))
    val4 = int(input("Assign potential value to letter 'd':"))
    tileType = int(input("Assign tile type:"))
    #counter example obtained by:
    #val1 = 1#0     #assign potential value to letter 'a'
    #val2 = -3#0     #assign potential value to letter 'b'
    #val3 = 13#0     #assign potential value to letter 'c'
    #val4 = 4#0     #assign potential value to letter 'd'
if type=="SELF":
     val1 = 0#0     #assign potential value to letter 'a'
     val2 = 6*math.sqrt(2)#9     #assign potential value to letter 'b'
     val3 = 10*math.sqrt(3)#19     #assign potential value to letter 'c'
     val4 = 9*math.pi#29     #assign potential value to letter 'd'
     itera = 6
     res = 20
     eig_num = 15
     tileType = 3
     check_input(itera, res, eig_num, tileType, "SPEC-BANDS-GEN")
     #print_tile(2)
else:
    itera = int(input("Enter iteration number:"))     #iteration number of substitution, should be larger than 2
    res = 0
    eig_num = 0
    if type=="EIG PLOT" or type=="SPEC BAND":
        res = int(input("Enter sampling resolution scale number:"))
    if type=="EIG PLOT" or type=="SPEC BAND":
        eig_num = int(input("Enter eigenvalue number:"))
    print("\n")
    #counter example obtained by:
    #itera = 3
    #res = 100           #sampling resolution number
    #eig_num = 9        #the eigenvalue to be plotted
    #type="EIG PLOT"     #Type of check in check_input function
    check_input(itera, res, eig_num, tileType, type)
