from __future__ import print_function
import csv
import pandas
import numpy as np
import sys


# Hipoteza: funkcja liniowa(h) jednej zmiennej
def h(theta, x):
    return theta[0] + theta[1] * x

#Funkcja kosztu
def J(h, theta, x, y):
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i])**2 for i in range(m)))

def gradient_descent(h, cost_fun, theta, x, y, alpha, eps):
    current_cost = cost_fun(h, theta, x, y)
    log = [[current_cost, theta]]  # log przechowuje wartości kosztu i parametrów
    m = len(y)
    while True:
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta  # jednoczesna aktualizacja - używamy zmiennej tymaczasowej
        try:
            current_cost, prev_cost = cost_fun(h, theta, x, y), current_cost
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log

def normalize(X):
    X_norm = X
    m = np.mean(X)
    s = np.std(X)
    X_norm = (X_norm - m) / s
    return X_norm

#Wczytywanie danych:
#

with open('train/train.tsv', 'r', encoding="utf-8") as trainset_data:
    reader = csv.DictReader(trainset_data, delimiter='\t')
    X_train_pow = list()
    y_train_cena = list()
    for row in reader:
        X_train_pow.append(float(row['Powierzchnia w m2']))
        y_train_cena.append(float(row['cena']))

test_in = pandas.read_csv('test-A/in.tsv', sep='\t', header=None)[0]

X_train_pow = normalize(X_train_pow)
test_in = normalize(test_in)

best_theta, log = gradient_descent(h, J, [0.0, 0.0], X_train_pow, y_train_cena, alpha=0.05, eps=0.0001)

test_out = h(best_theta, test_in)

with open('test-A/out.tsv', 'w') as output_file:
    for out in test_out:
        print('%.0f' % out, file = output_file)
