from pathlib import Path
import pandas as pd
import numpy as np

def load_data_csv(path:str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    path = Path(path)
    return pd.read_csv(path)
    
    
# SoftmaxRegression using Batch GD from scrath
class SoftmaxRegressionBatchGD:
    def __init__(self,learning_rate=0.01,n_iteration=1000, early_stop = 100):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.early_stop = early_stop
        self.param_matrix = 0        
        
    
    def fit(self,X,y,X_val,y_val):
        """
        This function will initialize the parameters of the parameters matrix 
        It will call iterativly the train function which will perform Batch Gradient Descent
        Early stopping is also implemented
        """
        
        # 1. Create the parameter space
        num_classes = len(np.unique(y,axis=0))
        num_parameters = X.shape[1]
        
        # Normal distribution of parameters
        self.param_matrix = np.random.randn(num_parameters,num_classes) * 0.1
        print(f"Random parameters matrix:\n{self.param_matrix}")
       
        train_losses = []
        val_losses = []
    
        i=0
        
        best_parameters = None
        
        min_val_loss = float('inf')
        count_no_min_update = 0
        # 2. Use training algorithm
        while i < self.n_iteration and count_no_min_update < self.early_stop :
            train_loss, val_loss = self.train_BatchGD(X,y,X_val,y_val)
            print(f"Epoch {i}: , train_loss={train_loss} , val_loss={val_loss}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < min_val_loss:
                best_parameters = self.param_matrix.copy()
                min_val_loss = val_loss
                count_no_min_update = 0  # Count reset
            else:
                count_no_min_update += 1
            
            i+=1
            
        # Restore best parameters
        if best_parameters is not None:
            self.param_matrix = best_parameters
        
        print(f"Training stopped at epoch {i}, val_loss = {min_val_loss}")
        
        return train_losses, val_losses
            
    def train_BatchGD(self,X,y,X_val,y_val):
        """This function will compute the scores of the instances and then the estimated probabilities.
           It will next evaluate the prediction.
           Then it will compute the gradient of the loss function and update the weights.
        

        Args:
            X : input features
            y : target labels

        Returns:
            float, float: train and validation loss
        """
        
        # 1. Compute the scores
        
        # Matrix of score for each class for each instance ([[score_class_0,...,score_class_2],...])
        scores_train = X @ self.param_matrix
        scores_val = np.dot(X_val,self.param_matrix)
        
        # 2. Compute the estimated probabilities
        proba_matrix_train = self._softmax(scores_train)
        proba_matrix_val = self._softmax(scores_val)
        
        # 3. Evaluate train and val losses
        epsilon = 10e-15 # Avoid log(0)
        cross_entropy_loss_train = (- np.sum(y * np.log(proba_matrix_train + epsilon),axis=1)).mean()
        cross_entropy_loss_val = (- np.sum(y_val * np.log(proba_matrix_val + epsilon),axis=1)).mean()
                
        
        # 4. One round of batchGD
        gradient = (X.T @ (proba_matrix_train - y))/X.shape[0]

        self.param_matrix -= self.learning_rate*gradient
        
        return (cross_entropy_loss_train,cross_entropy_loss_val)
    
    def _softmax(self,scores):
        """For a n,m matrix computes the softmax score of each instance and each probability

        Args:
            scores (_type_): Matrix of scores (n,m)
        """
        
        # 1. exp_k(x)
        exp_scores = np.exp(scores)
          
        # 2. sum of exp
        sum_exp = np.sum(exp_scores,axis=1,keepdims=True)

        # 3. Probabilities
        probabilities = exp_scores / sum_exp
        
        return probabilities
    
    def predict(self,X):
        """For a given test set, the function will compute the scores and return a list of classes

        Args:
            X (_type_): Matrix(n,m)
        """
    
        scores = X @ self.param_matrix

        y_pred = np.argmax(scores,axis=1)
        
        return y_pred
    