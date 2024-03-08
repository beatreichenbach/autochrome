# https://github.com/Hiroaki-K4/func_opt/blob/main/levenberg_marquardt.py

import random

import numpy as np


def calculate_f(x, y, z, u):
    return (u[0] * x**3 + u[1] * y**2 + u[2] * x * y + 27) - z


def calculate_cost(x, y, z, u):
    cost_sum = 0
    for i in range(len(x)):
        cost_sum += calculate_f(x[i], y[i], z[i], u) ** 2

    return cost_sum / 2


def create_dataset(x, y, z, u):
    z_list = []
    for i in range(len(x)):
        noise = random.random() * 0.01
        z = calculate_f(x[i], y[i], 0, u) + noise
        z_list.append(z)

    return z_list


def df_du_0(x):
    return x**3


def df_du_1(y):
    return y**2


def df_du_2(x, y):
    return x * y


def calculate_nabla_u(x, y):
    return np.array([[df_du_0(x)], [df_du_1(y)], [df_du_2(x, y)]])


def levenberg_marquardt(x, y, z, u_0):
    c = 0.0001
    u = u_0
    J = calculate_cost(x, y, z, u_0)
    while True:
        grad_J_sum = np.zeros([3, 1])
        H_sum = np.zeros([3, 3])
        new_J = J
        while J <= new_J:
            for i in range(len(x)):
                nabla_u = calculate_nabla_u(x[i], y[i])
                grad_J_sum += calculate_f(x[i], y[i], z[i], u) * nabla_u
                H_sum += np.dot(nabla_u, nabla_u.T)

            mix_H = H_sum + c * np.diag(np.diag(H_sum))
            u_move = np.dot(np.linalg.inv(mix_H), grad_J_sum)
            new_u = u - u_move
            new_J = calculate_cost(x, y, z, new_u)

            if new_J > J:
                c = 10 * c
            else:
                c = c / 10
                J = new_J
                u = new_u
                break

        if np.linalg.norm(u_move) < 1e-8:
            break

    return u


def main():
    true_u = np.array([3, 2, -9])
    u_0 = np.array([1, 1, 1]).reshape(3, 1)
    x = [i for i in range(10)]
    y = [i for i in range(-5, 5)]
    z = create_dataset(x, y, 0, true_u)
    predict_u = levenberg_marquardt(x, y, z, u_0)
    print("True U: ", true_u)
    print("Predicted U: ", predict_u[:, 0])


if __name__ == "__main__":
    main()
