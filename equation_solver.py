import numpy as np
from fractions import Fraction as frac
import math


def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact


def U(k, h):
#     print("calling U({}, {})".format(k, h))
    answer = -1
    if (k, h) in solved_dict:
        return solved_dict[(k, h)]
    if h == 0:
        answer = 1 / (2.718 * factorial(k - 1))
    else:
        # part 1
        a12 = U(k + 2, h -1)
        a1 = epsilon * (k + 2) * (k + 1) * a12

        #part 2
        a2 = 0
        for r in range(0, k + 1):
            for s in range(0, h):
                a21 = k - r + 1
                a22 = U(r, h - 1 - s)
                a23 = U(k - r + 1, s)
                a2 = a2 + a21 * a22 * a23

        answer = (a1 - a2) / h
        #         answer = frac(a1, a3)

    if (k, h) not in solved_dict:
        solved_dict[(k, h)] = answer
#         print("Saved U({}, {})".format(k, h))

    # print(m, h, answer)
    return answer


def superscript(n):
    return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c) - ord('0')] for c in str(n)])


def make_pretty(character, digit):
    if digit == 0:
        return ''
    if digit == 1:
        return character
    return character + superscript(digit)


def print_equation(solved_dict, max_k, max_h):
    value_equation = ''
    for k in range(max_k + 1):
        for h in range(max_h + 1):
            if (k, h) not in solved_dict:
                print('values are not present in the dictionary, please solve the equation first!')
                return
            if solved_dict[(k, h)] != 0:
                temp_equation = ''
                if solved_dict[(k, h)] != 1:
                    temp_equation += str(solved_dict[(k, h)])
                remaining = ''
                remaining += make_pretty('x', k)
                remaining += make_pretty('y', h)

                if remaining != '' and temp_equation != '':
                    temp_equation = temp_equation + ' * ' + remaining
                elif temp_equation == '' and remaining != '':
                    temp_equation = remaining
                #                 print(temp_equation, remaining)

                if temp_equation != '':
                    if value_equation == '':
                        value_equation += temp_equation
                    else:
                        value_equation += ' + ' + temp_equation

    print(value_equation)
    # return value_equation


def get_value_equation(solved_dict, max_k, max_h, x, y):
    value_equation = 0
    for k in range(0, max_k + 1):
        for h in range(max_h + 1):
            if (k, h) not in solved_dict:
                print('values are not present in the dictionary, please solve the equation first!')
                return
            curr_value = solved_dict[(k, h)]
            curr_value *= math.pow(x, k)
            curr_value *= math.pow(y, h)
            value_equation += curr_value

    #     print(value_equation)
    return value_equation


####### Calculate U(k, h)
# Initialize the values
epsilon = 1
K = 6
H = 6
# M = K + 2
e = 2.71
solved_dict = {}

# Call function to calculate all values
U(K, H)

####### Print all U(k, h)
# print values as you need
for i in range(0, K):
    for j in range(0, H):
        print('U(' + str(i) + ', ' + str(j) + ') = ' + str(solved_dict[(i, j)]))


####### print equation
max_X = 1
max_Y = 1
max_k = max_h = 5

print_equation(solved_dict, max_k, max_h)

####### Calculate F(x, y)
answer_xy = {}
x_range = np.arange(0.0, max_X + 0.05, 0.2)
y_range = np.arange(0.0, max_Y + 0.05, 0.2)
X = []
Y = []
Z = []
for x in x_range:
    for y in y_range:
        answer_xy[(x, y)] = get_value_equation(solved_dict, max_k, max_h, x, y)
        X.append(x)
        Y.append(y)
        Z.append(answer_xy[(x, y)])

####### plot 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# max_K = max_H = 10
# max_X = max_Y = 10
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, cmap='Blues');
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

####### Write output to a file
f = open('output_values_x10_y10_k_10_h10_step0.1.txt', 'w')
f.write('f(x, y) = ')
for i in range(len(X)):
    f.write('f({0}, {1}) = {2}'.format(round(X[i], 1), round(Y[i], 1), round(Z[i], 2)) + '\n')
f.close()

import pandas as pd

df = pd.DataFrame(data={'x': X, 'y': Y, 'U(x, y)': Z})
# df.to_excel('doutput_uxy.xlsx')
# df.to_csv('doutput_uxy.csv')
df.tail(2)

########## fix y and plot U(x, y)
y = 0.2
tmp = df[df['y'] == y]
plt.scatter(tmp['x'], tmp['U(x, y)'])
plt.xlabel("x")
plt.ylabel("U(x, " + str(y) + ")")
plt.show()

########## fix x and plot U(x, y)
x = 0.2
tmp = df[df['x'] == x]
plt.scatter(tmp['y'], tmp['U(x, y)'])
plt.xlabel("y")
plt.ylabel("U(" + str(x) + ", y)")
plt.show()
