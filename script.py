
from numpy import *

def find_error(m, c, points):
    total_error = 0
    for i in range(1, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m*x + c))**2

    return total_error/float(len(points))

def step_grad(m_curr, c_curr, points, learning_rate):
    c_grad = 0
    m_grad = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        c_grad = -(2/N)*(y - (m_curr*x + c_curr))
        m_grad = -(2/N)*x*(y - (m_curr*x + c_curr))
    return [(c_curr - (learning_rate*c_grad)), (m_curr - (learning_rate*m_grad))]

def gradient(points, init_c, init_m, num_iter, learning_rate):
    m = init_m
    c = init_c
    for i in range(num_iter):
        c, m = step_grad(m, c, array(points), learning_rate)
    return [c, m]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    initial_m = 0.0
    initial_c = 0.0
    num_iterations = 1000
    learning_rate = 0.0001

    print("Starting gradient descent at m = {0}, c = {1}, error = {2}".format(initial_m, initial_c, find_error(initial_m, initial_c, points)))
    print ("Running...")
    [c, m] = gradient(points, initial_c, initial_m, num_iterations, learning_rate)
    print("Starting gradient descent at m = {0}, c = {1}, error = {2}".format(m, c, find_error(m, c, points)))

if __name__ == '__main__':
    run()
