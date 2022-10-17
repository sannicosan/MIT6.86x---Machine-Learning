import project1 as p1
import utils
import numpy as np

# x = np.array([[1,2,3],[4,5,6],[7,8,9]])
# y = x
# for xi,yi in zip(x,y):
#     xi += 1
#     yi = 2
#
# print(x)
# print(y)

# z = np.zeros(x.shape[0])
# print(z)


stop_words_list = []
with open('stopwords.txt') as file:
    for line in file:
         stop_words_list.append(line.split()[0])

print(stop_words_list)