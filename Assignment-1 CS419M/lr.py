import numpy as np
import pandas as pd
import matplotlib as plt
import math
import csv

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

# """ 
# You are allowed to change the names of function arguments as per your convenience, 
# but it should be meaningful.

# E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
# but abc, a, b, etc. are not.

# """

def get_features(file_path):
    # # Given a file path , return feature matrix and target labels 
    

    data = pd.read_csv(file_path, sep = ',' )
    phi = data.values[0:, 0:6]
    y = data.values[0:, 6:7]
    new_phi = np.ones((len(phi),11))
    new_phi[0:, 7:11] = phi[0:, 1:5]                            # ignoring the number of passengers

    for i in range(len(phi)):
        #first column has all ones in new_phi
        new_phi[i][1] = (phi[i][0].split()[0]).split('-')[0]    # year
        new_phi[i][2] = (phi[i][0].split()[0]).split('-')[1]    # month
        new_phi[i][3] = (phi[i][0].split()[0]).split('-')[2]    # date
        new_phi[i][4] = (phi[i][0].split()[1]).split(':')[0]    # hour
        new_phi[i][5] = (phi[i][0].split()[1]).split(':')[1]    # minute
        new_phi[i][6] = (phi[i][0].split()[1]).split(':')[2]    # second

    for i in range(1,11):
        std = (np.std(new_phi[:,i])) 
        mean = np.mean(new_phi[:,i]) * np.ones((1,len(phi)))         # feature scaling by ((x- mean) / std_deviationn)

        new_phi[:,i] = ( new_phi[:,i] - mean ) / std
        
    return new_phi, y

def get_features_basis1(file_path):
    # Given a file path , return feature matrix and target labels 
    
    phi, y = get_features(file_path)

    for i in range(len(phi.T)):
        phi[:, i] =  np.multiply(phi[:,i],np.sin(phi[:,i]))

    return phi, y

def get_features_basis2(file_path):
    # Given a file path , return feature matrix and target labels 
    
    phi, y = get_features(file_path)

    for i in range(len(phi.T) - 1):
        phi[:, i] = phi[:, i] + phi[:, i + 1]
    phi[:,len(phi.T) - 1] = phi[:, len(phi.T) - 1] + phi[:, 0] 

    return phi, y

def compute_RMSE(phi, w , y) :
    # Root Mean Squared Error

    sq_err_sum_mean = np.sum([x ** 2 for x in (y - phi.dot(w))]) / len(phi)
    error = math.sqrt(sq_err_sum_mean)
    
    return error


def generate_output(phi_test, w):
    # writes a file (output.csv) containing target variables in required format for Kaggle Submission.
    with open('Output.csv', 'w') as output:
        (csv.writer(output)).writerow(('Id', 'fare'))
        for i in range(len(phi_test)):
            (csv.writer(output)).writerow((i, float(phi[i, :].dot(w))))
        output.close()
	
def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return ((np.linalg.inv((phi.T).dot(phi))).dot(phi.T)).dot(y)
	
def gradient_descent(phi, y) :
    # Mean Squared Error 

    w = np.zeros((len(phi.T),1))
    learning_rate = 0.0000001
    eps = 0.001 

    L2norm_of_gradient = 1
    while L2norm_of_gradient >= eps:
        gradient = (-2 * (phi.T).dot((y - phi.dot(w))))
        w = w - learning_rate * gradient
        L2norm_of_gradient = math.sqrt(np.sum(gradient.T.dot(gradient)))      

    return w

def sgd(phi, y):

    w = np.zeros((len(phi.T),1))
    learning_rate = 0.0000001
    eps = 0.1 

    L2norm_of_gradient = 1
    while L2norm_of_gradient >= eps:
        for i in range(len(phi)):
            gradient = (- 2 *  (phi[i]).reshape((-1,1)) * (y[i] - phi[i].dot(w)))
            w = w - learning_rate * gradient
            L2norm_of_gradient = math.sqrt(np.sum(gradient.T.dot(gradient)))
        if L2norm_of_gradient <= eps:
            break      

    return w

def pnorm(phi, y, p) :
    # Mean Squared Error
    w = np.zeros((len(phi.T),1))
    learning_rate = 0.000001
    eps = 0.001 

    lmbda = 1

    L2norm_of_gradient = 1
    while L2norm_of_gradient >= eps:
        grad_w_pnorm = np.sum([ p*(x**(p-1)) for x in w ])
        gradient_with_regularization = (-2 * (phi.T).dot((y - phi.dot(w))) + lmbda * grad_w_pnorm)
        w = w - learning_rate * gradient_with_regularization
        L2norm_of_gradient = math.sqrt(np.sum(gradient_with_regularization.T.dot(gradient_with_regularization)))      

    return w

	
def main():
# """ 
# The following steps will be run in sequence by the autograder.
# """
        ######## Task 1 #########
        phase = "train"
        phi, y = get_features('train.csv')
        w1 = closed_soln(phi, y)
        w2 = gradient_descent(phi, y)
        phase = "eval"
        phi_dev, y_dev = get_features('dev.csv')
        r1 = compute_RMSE(phi_dev, w1, y_dev)
        r2 = compute_RMSE(phi_dev, w2, y_dev)
        print('1a: ')
        print(abs(r1-r2))
        w3 = sgd(phi, y)
        r3 = compute_RMSE(phi_dev, w3, y_dev)
        print('1c: ')
        print(abs(r2-r3))

        ######## Task 2 #########
        w_p2 = pnorm(phi, y, 2)  
        w_p4 = pnorm(phi, y, 4)  
        r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
        r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
        print('2: pnorm2')
        print(r_p2)
        print('2: pnorm4')
        print(r_p4)

        ######## Task 3 #########
        phase = "train"
        phi1, y = get_features_basis1('train.csv')
        phi2, y = get_features_basis2('train.csv')
        phase = "eval"
        phi1_dev, y_dev = get_features_basis1('dev.csv')
        phi2_dev, y_dev = get_features_basis2('dev.csv')
        w_basis1 = pnorm(phi1, y, 2)  
        w_basis2 = pnorm(phi2, y, 2)  
        rmse_basis1 = compute_RMSE(phi1_dev, w_basis1, y_dev)
        rmse_basis2 = compute_RMSE(phi2_dev, w_basis2, y_dev)
        print('Task 3: basis1')
        print(rmse_basis1)
        print('Task 3: basis2')
        print(rmse_basis2)
        
main()