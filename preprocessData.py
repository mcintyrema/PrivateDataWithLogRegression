import pandas as pd
import numpy as np

def preprocess_data():
    user_data = pd.read_csv("../data/adult.csv/adult.csv", header=None) #[15001 x 10]
    headings = user_data.iloc[0].tolist()
    user_data.columns = headings

    #extract patient dataset feature columns without headings
    X = user_data.iloc[1:, :-1].values  # features (excluding income)
    y = user_data.iloc[1:, -1].values  # labels (>50k or <=50K)
   
    # append array of ones to temp with length = to amt of x rows and 1 col
    oneVector = np.ones((X.shape[0], 1))
    # combine ones vector with score vectors to make feature matrix
    X = np.hstack((oneVector, X)) #(32561, 15)

    # min-max Normalization
    # X[:, 1:] = (X[:, 1:] - np.min(X[:, 1:], axis=0)) / (np.max(X[:, 1:], axis=0) - np.min(X[:, 1:], axis=0))

    y = np.array(y) # (32561,)

    # randomly grabbing 90% of training set
    size90 = int(0.9 * X.shape[0])
    #getting random training sample for y by selecting indices at random w/o replacement
    y_train_index = np.random.choice(np.arange(y.size), size90, replace=False)
    y_train = [y[i] for i in y_train_index]
    # #getting test sample for y by getting indices not chosen above
    y_test_index = np.setdiff1d(np.arange(y.size), y_train_index)
    y_test = [y[i] for i in y_test_index]
    # #getting random sample from x of training examples
    x_train_index = np.random.choice(X.shape[0], size90, replace=False)
    x_train_index = x_train_index.astype(int) # convert to int
    x_train = X[x_train_index] 
    #getting other 10%
    x_test_index = np.setdiff1d(np.arange(X.shape[0]), x_train_index)
    x_test = X[x_test_index]
    
    #convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return [x_train, y_train, x_test, y_test]
    

def main():
    preprocess_data()

if __name__ == '__main__':
    main()