import logging
from logging.handlers import TimedRotatingFileHandler
import os
import time
import traceback

class Logger:
    def __init__(self, log_dir='logs', log_file='app.log', level=logging.INFO, env='development'):
        """
        Initializes the logger with different configurations based on the environment (e.g., development, production).
        """
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Choose log format based on environment
        if env == 'development':
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:  # production environment format
            log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Set up the file handler with rotation every day and a backup count of 30 days
        log_path = os.path.join(log_dir, log_file)
        file_handler = TimedRotatingFileHandler(log_path, when="midnight", backupCount=30)
        file_handler.setFormatter(log_format)

        # Set up the console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)

        # Get the logger instance and configure it
        self.logger = logging.getLogger('MultimodalLogger')
        self.logger.setLevel(level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Additional attributes for tracking performance
        self.performance_logs = []

    def log_info(self, message):
        """
        Logs an informational message.
        """
        self.logger.info(message)

    def log_warning(self, message):
        """
        Logs a warning message.
        """
        self.logger.warning(message)

    def log_error(self, message):
        """
        Logs an error message along with exception details.
        """
        self.logger.error(message)
        self.logger.error(traceback.format_exc())

    def log_critical(self, message):
        """
        Logs a critical error that needs immediate attention.
        """
        self.logger.critical(message)

    def log_multimodal_event(self, modality, message):
        """
        Custom log function to track specific multimodal events.
        """
        self.logger.info(f"Modality: {modality} - Event: {message}")

    def log_performance(self, task_name, start_time, end_time):
        """
        Logs the performance of a task, including the time taken.
        """
        time_taken = end_time - start_time
        self.performance_logs.append({
            'task': task_name,
            'time_taken': time_taken
        })
        self.logger.info(f"Task: {task_name} took {time_taken:.4f} seconds.")

    def log_exception(self, exception_message):
        """
        Logs exceptions in a structured format.
        """
        self.logger.error(f"Exception occurred: {exception_message}")
        self.logger.error(traceback.format_exc())

    def log_batch_processing(self, batch_size, success_count, failure_count):
        """
        Logs batch processing details, including the number of successful and failed operations.
        """
        self.logger.info(f"Batch size: {batch_size}, Success: {success_count}, Failures: {failure_count}")

    def log_multimodal_request(self, request_id, modalities, status, error=None):
        """
        Logs a multimodal request with its status and potential errors.
        """
        if error:
            self.logger.error(f"Request ID: {request_id}, Modalities: {modalities}, Status: {status}, Error: {error}")
        else:
            self.logger.info(f"Request ID: {request_id}, Modalities: {modalities}, Status: {status}")

    def monitor_model_performance(self, model_name, metrics):
        """
        Logs model performance metrics like accuracy, F1-score, etc.
        """
        for metric, value in metrics.items():
            self.logger.info(f"Model: {model_name} - {metric}: {value}")

    def log_experiment(self, experiment_name, parameters, results):
        """
        Logs details about an experiment, including its parameters and results.
        """
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Parameters: {parameters}")
        self.logger.info(f"Results: {results}")

    def track_model_latency(self, model_name, latency):
        """
        Logs the inference latency of a model.
        """
        self.logger.info(f"Model: {model_name}, Latency: {latency:.4f} seconds")

    def log_system_resources(self, cpu_usage, memory_usage):
        """
        Logs system resource usage (CPU and memory) during inference or training.
        """
        self.logger.info(f"CPU Usage: {cpu_usage:.2f}%, Memory Usage: {memory_usage:.2f}%")

    def log_training_epoch(self, epoch, loss, accuracy):
        """
        Logs the details of a training epoch, including the loss and accuracy.
        """
        self.logger.info(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def log_hyperparameters(self, hyperparams):
        """
        Logs the hyperparameters used in model training or experiments.
        """
        self.logger.info(f"Hyperparameters: {hyperparams}")

    def log_user_interaction(self, user_id, action, timestamp):
        """
        Logs user interactions with the system.
        """
        self.logger.info(f"User ID: {user_id}, Action: {action}, Timestamp: {timestamp}")

    def finalize_performance_logs(self):
        """
        Finalizes and logs the collected performance metrics at the end of a session.
        """
        total_time = sum([log['time_taken'] for log in self.performance_logs])
        self.logger.info(f"Total time for all tasks: {total_time:.4f} seconds.")
        for log in self.performance_logs:
            self.logger.info(f"Task: {log['task']}, Time Taken: {log['time_taken']:.4f} seconds")

    def clear_logs(self):
        """
        Clears the performance logs.
        """
        self.performance_logs = []

# Utility functions for performance monitoring
def track_time(func):
    """
    Decorator for tracking the time taken by a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.log_performance(func.__name__, start_time, end_time)
        return result
    return wrapper

# Usage
if __name__ == '__main__':
    logger = Logger(env='production')
    
    # Log basic information
    logger.log_info("Multimodal system initialized.")
    logger.log_warning("Low memory warning.")
    
    # Multimodal request
    logger.log_multimodal_request(request_id='12345', modalities=['text', 'image'], status='success')
    
    # Log performance of tasks
    start = time.time()
    # Task processing...
    time.sleep(1.5)  # 1.5-second task
    end = time.time()
    logger.log_performance('Text-Image Fusion', start, end)

    # Log batch processing
    logger.log_batch_processing(batch_size=100, success_count=95, failure_count=5)

    # Log an experiment with results
    experiment_params = {'learning_rate': 0.001, 'batch_size': 32}
    experiment_results = {'accuracy': 0.89, 'f1_score': 0.87}
    logger.log_experiment('Text-Image Model Training', experiment_params, experiment_results)

    # Log model performance
    model_metrics = {'accuracy': 0.92, 'f1_score': 0.91}
    logger.monitor_model_performance('Text-Image Model', model_metrics)

    # Log training epoch
    logger.log_training_epoch(epoch=10, loss=0.015, accuracy=0.982)

    # Log hyperparameters
    hyperparams = {'optimizer': 'Adam', 'learning_rate': 0.001, 'dropout': 0.5}
    logger.log_hyperparameters(hyperparams)

    # Some operation that may raise an error
    try:
        risky_operation_result = 1 / 0
    except Exception as e:
        logger.log_exception("An error occurred during risky operation.")

    # Log system resource usage
    logger.log_system_resources(cpu_usage=75.5, memory_usage=68.2)

    # Log and finalize performance metrics
    logger.finalize_performance_logs()
    logger.clear_logs()