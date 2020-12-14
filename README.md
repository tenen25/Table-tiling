# Table-tiling
import numpy as np

tile1 = (['a', 'c'],
           ['c', 'a'])
           
tile2 = (['b', 'd'],
          ['d', 'b'])           


def sub( arr, n ):
    if n==1:
        if arr==['a']:
            mat=['b','a']['d','a']
        if arr==['b']:
            mat=['a','c']['b','b']
        if arr==['c']:
            mat=['c','b']['c','d']
        if arr==['d']:
            mat=['d','d']['a','c']
        return mat
    else:
        for s in range (n):
            for k in range(n):
                for i in range (2):
                    for j in range(2):
                        mat[2s+i][2k+j]= sub(mat[s][k], 1)[i][j]
        return mat
        
def iter( tile , n ):
    for i in range (n):
        if i==0:
            mat=tile
        else:
            mat=sub( mat, i+1 )
    print(mat)    
    
    
