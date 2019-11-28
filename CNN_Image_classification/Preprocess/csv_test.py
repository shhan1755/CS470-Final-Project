import sys
import pandas as pd

path = sys.argv[1]



f = open(path, 'r')

lines = f.readlines()

matrix = lines[1].split(',')[1]
matrix_splitted = matrix.split(' ')
print(len(matrix_splitted))
csv = pd.read_csv(path)
print(csv.shape)
print(csv['pixels'])
print(csv['emotion'])
print(csv['Usage'])
print(pd.__version__)