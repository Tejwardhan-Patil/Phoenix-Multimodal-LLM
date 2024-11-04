import time
import logging
from collections import defaultdict
import json
import smtplib
from email.mime.text import MIMEText

# Setup logging
logging.basicConfig(filename='monitor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceMonitor:
    def __init__(self, alert_thresholds=None, email_alerts=False):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.alert_thresholds = alert_thresholds or {}
        self.email_alerts = email_alerts
        self.alert_recipients = []
        
    def set_alert_recipients(self, recipients):
        """Set the list of email recipients for alerts."""
        self.alert_recipients = recipients

    def send_alert(self, metric_name, value):
        """Send an email alert if a metric breaches its threshold."""
        if self.email_alerts and self.alert_recipients:
            subject = f"Alert: {metric_name} exceeded threshold"
            body = f"The metric {metric_name} has exceeded its threshold with a value of {value}."
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'monitor@website.com'
            msg['To'] = ', '.join(self.alert_recipients)
            
            try:
                with smtplib.SMTP('smtp.website.com') as server:
                    server.sendmail('monitor@website.com', self.alert_recipients, msg.as_string())
                logging.info(f"Alert sent for {metric_name} with value {value}")
            except Exception as e:
                logging.error(f"Failed to send alert: {e}")

    def log_metric(self, metric_name, value):
        """Logs the value of a given metric and checks for alert thresholds."""
        self.metrics[metric_name].append(value)
        logging.info(f"Metric logged - {metric_name}: {value}")
        
        # Check if the metric exceeds the threshold
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            if value > threshold:
                logging.warning(f"{metric_name} exceeded threshold with value {value}")
                self.send_alert(metric_name, value)

    def calculate_average(self, metric_name):
        """Calculates the average of the values for a given metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        else:
            logging.warning(f"No data for metric: {metric_name}")
            return None

    def log_performance(self):
        """Logs the current performance statistics for all tracked metrics."""
        for metric, values in self.metrics.items():
            avg_value = self.calculate_average(metric)
            logging.info(f"Average {metric}: {avg_value}")

    def track_model_metrics(self, accuracy, f1_score, multimodal_coherence, loss, precision, recall):
        """Logs performance metrics for accuracy, F1-score, multimodal coherence, loss, precision, and recall."""
        self.log_metric('accuracy', accuracy)
        self.log_metric('f1_score', f1_score)
        self.log_metric('multimodal_coherence', multimodal_coherence)
        self.log_metric('loss', loss)
        self.log_metric('precision', precision)
        self.log_metric('recall', recall)

    def log_runtime(self):
        """Logs the total runtime of the monitoring session."""
        end_time = time.time()
        total_runtime = end_time - self.start_time
        logging.info(f"Total runtime: {total_runtime} seconds")

    def generate_report(self, report_file='performance_report.json'):
        """Generates a report of all tracked metrics and their averages in JSON format."""
        report_data = {}
        for metric, values in self.metrics.items():
            avg_value = self.calculate_average(metric)
            report_data[metric] = {
                'values': values,
                'average': avg_value
            }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4)
        logging.info(f"Performance report generated: {report_file}")

    def reset_metrics(self):
        """Resets all metrics."""
        self.metrics = defaultdict(list)
        logging.info("All metrics have been reset.")

    def log_alert_thresholds(self):
        """Logs the current alert thresholds for all metrics."""
        for metric, threshold in self.alert_thresholds.items():
            logging.info(f"Alert threshold for {metric}: {threshold}")

    def print_summary(self):
        """Prints a summary of the monitoring session to the console."""
        print("=== Performance Summary ===")
        for metric, values in self.metrics.items():
            avg_value = self.calculate_average(metric)
            print(f"{metric} - Average: {avg_value}, Count: {len(values)}")
        print("===========================")

# Usage
if __name__ == "__main__":
    monitor = PerformanceMonitor(alert_thresholds={'accuracy': 0.95, 'loss': 0.05}, email_alerts=True)
    
    # Set email alert recipients
    monitor.set_alert_recipients(['admin@website.com'])

    # Metrics tracking
    monitor.track_model_metrics(
        accuracy=0.85, f1_score=0.80, multimodal_coherence=0.90,
        loss=0.10, precision=0.88, recall=0.85
    )
    monitor.track_model_metrics(
        accuracy=0.97, f1_score=0.85, multimodal_coherence=0.93,
        loss=0.03, precision=0.90, recall=0.87
    )
    
    # Log performance, runtime, and generate report
    monitor.log_performance()
    monitor.log_runtime()
    monitor.generate_report()

    # Reset metrics for new session
    monitor.reset_metrics()
    
    # Log alert thresholds and print summary
    monitor.log_alert_thresholds()
    monitor.print_summary()