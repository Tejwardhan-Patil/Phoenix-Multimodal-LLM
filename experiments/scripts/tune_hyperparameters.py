import numpy as np
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer
import yaml
import logging
import os
import time

# Function to set up logging
def setup_logging(log_dir='logs', log_file='hyperparameter_tuning.log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging initialized")

# Load hyperparameter space configuration file
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error reading YAML config file: {e}")
        raise

# Define model hyperparameter search spaces
def get_hyperparameter_space():
    param_space = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(10, 50),
        'min_samples_split': Real(0.01, 0.1, prior='log-uniform'),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.1, 1.0)
    }
    logging.info("Defined hyperparameter search space")
    return param_space

# Function for Grid Search Hyperparameter Tuning
def grid_search(model, param_grid, X_train, y_train):
    logging.info("Starting Grid Search...")
    start_time = time.time()
    
    grid_search_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search_cv.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Grid Search completed in {elapsed_time:.2f} seconds")
    
    return grid_search_cv.best_params_

# Function for Bayesian Optimization Hyperparameter Tuning
def bayesian_optimization(model, param_space, X_train, y_train):
    logging.info("Starting Bayesian Optimization...")
    start_time = time.time()
    
    bayes_search_cv = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=32, cv=5, scoring='accuracy')
    bayes_search_cv.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Bayesian Optimization completed in {elapsed_time:.2f} seconds")
    
    return bayes_search_cv.best_params_

# Function to evaluate the model performance
def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    logging.info(f"Model accuracy on test data: {accuracy:.4f}")
    return accuracy

# Function to save the best parameters to a file
def save_best_params(best_params, output_path='best_params.yaml'):
    try:
        with open(output_path, 'w') as file:
            yaml.dump(best_params, file)
        logging.info(f"Saved best hyperparameters to {output_path}")
    except Exception as e:
        logging.error(f"Error saving best hyperparameters: {e}")

# Main function to perform hyperparameter tuning
def tune_hyperparameters(X_train, y_train, X_test=None, y_test=None, method='bayesian', config_path='configs/hyperparams.yaml', output_path='best_params.yaml'):
    setup_logging()
    
    logging.info("Hyperparameter tuning started")
    
    # Load config for hyperparameter space
    config = load_config(config_path)
    
    # Define model 
    model = RandomForestClassifier()

    # Hyperparameter tuning method selection
    if method == 'grid':
        logging.info("Selected method: Grid Search")
        best_params = grid_search(model, config['param_grid'], X_train, y_train)
    elif method == 'bayesian':
        logging.info("Selected method: Bayesian Optimization")
        best_params = bayesian_optimization(model, get_hyperparameter_space(), X_train, y_train)
    else:
        logging.error("Invalid hyperparameter tuning method. Choose 'grid' or 'bayesian'.")
        raise ValueError("Invalid hyperparameter tuning method. Choose 'grid' or 'bayesian'.")

    logging.info(f"Best hyperparameters found: {best_params}")
    
    # Evaluate the model
    if X_test is not None and y_test is not None:
        logging.info("Evaluating model performance on test data...")
        evaluate_model(model, X_test, y_test)
    
    # Save the best parameters
    save_best_params(best_params, output_path)

    logging.info("Hyperparameter tuning finished")
    return best_params

# Utility function to split data into training and testing sets
def split_data(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    logging.info(f"Splitting data into training and testing sets with test size {test_size}")
    return train_test_split(X, y, test_size=test_size)

# Function to preprocess input data
def preprocess_data(X):
    logging.info(f"Preprocessing data with {X.shape[0]} samples and {X.shape[1]} features")
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Generate synthetic dataset for hyperparameter tuning
def generate_synthetic_data(n_samples=1000, n_features=20):
    from sklearn.datasets import make_classification
    logging.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
    X, y = make_classification(n_samples=n_samples, n_features=n_features)
    return X, y

# Main function for running hyperparameter tuning
if __name__ == "__main__":
    logging.info("Running hyperparameter tuning script...")

    # Generate synthetic dataset
    X, y = generate_synthetic_data()
    
    # Preprocess data
    X = preprocess_data(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Perform hyperparameter tuning with Bayesian optimization
    best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, method='bayesian')

    logging.info(f"Optimized Hyperparameters: {best_params}")
    print(f"Optimized Hyperparameters: {best_params}")