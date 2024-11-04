import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging

# Setup logging
logging.basicConfig(filename='reports/report_generation.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path configurations
results_dir = 'experiments/results/'
output_dir = 'reports/'
os.makedirs(output_dir, exist_ok=True)

# Constants
MODALITIES = ['text', 'image', 'audio']
METRICS = ['accuracy', 'f1_score', 'precision', 'recall']

# Load evaluation results
def load_results(modality):
    result_file = os.path.join(results_dir, f'{modality}_evaluation.json')
    if not os.path.exists(result_file):
        logging.error(f"Result file not found for {modality}")
        return None
    
    with open(result_file, 'r') as file:
        logging.info(f"Loaded results for {modality}")
        return json.load(file)

# Generate report for a given modality
def generate_report(modality, results):
    if results is None:
        logging.warning(f"No results for {modality}; skipping report generation.")
        return
    
    report_path = os.path.join(output_dir, f'{modality}_report.txt')
    
    with open(report_path, 'w') as report_file:
        logging.info(f"Generating report for {modality}")
        report_file.write(f"Report for {modality.capitalize()} Modality\n")
        report_file.write("=" * 50 + "\n")
        for metric in METRICS:
            report_file.write(f"{metric.capitalize()}: {results[metric]:.4f}\n")
        
        report_file.write("\nDetailed Metrics:\n")
        for metric in results['detailed_metrics']:
            report_file.write(f"{metric}: {results['detailed_metrics'][metric]:.4f}\n")
        
        report_file.write("\nConfusion Matrix:\n")
        confusion_matrix = np.array(results['confusion_matrix'])
        for row in confusion_matrix:
            report_file.write("\t".join(map(str, row)) + "\n")
        report_file.write("=" * 50 + "\n")
    
    logging.info(f"Report generated for {modality} at {report_path}")

# Generate summary report across modalities
def generate_summary_report(modalities):
    summary_data = []

    for modality in modalities:
        results = load_results(modality)
        if results:
            summary_data.append({
                'modality': modality,
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'precision': results['precision'],
                'recall': results['recall']
            })

    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'summary_report.csv')
    df.to_csv(summary_path, index=False)
    
    logging.info("Summary report generated successfully at summary_report.csv")

# Generate plots for visualizing performance
def generate_plots(modality, results):
    if results is None:
        logging.warning(f"No results for {modality}; skipping plot generation.")
        return
    
    plt.figure(figsize=(10, 6))
    labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [results['accuracy'], results['f1_score'], results['precision'], results['recall']]
    
    plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
    plt.title(f'{modality.capitalize()} Performance Metrics')
    plt.ylabel('Scores')
    plt.ylim([0, 1])
    
    plot_path = os.path.join(output_dir, f'{modality}_performance.png')
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"Performance plot saved for {modality} at {plot_path}")

# Generate confusion matrix heatmap
def generate_confusion_matrix_heatmap(modality, results):
    if results is None:
        logging.warning(f"No results for {modality}; skipping confusion matrix heatmap generation.")
        return

    confusion_matrix = np.array(results['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title(f'{modality.capitalize()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.xticks(np.arange(len(confusion_matrix)), np.arange(len(confusion_matrix)))
    plt.yticks(np.arange(len(confusion_matrix)), np.arange(len(confusion_matrix)))

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

    heatmap_path = os.path.join(output_dir, f'{modality}_confusion_matrix.png')
    plt.savefig(heatmap_path)
    plt.close()

    logging.info(f"Confusion matrix heatmap saved for {modality} at {heatmap_path}")

# Generate detailed performance insights
def generate_performance_insights(results):
    if results is None:
        return {}
    
    insights = {}
    accuracy = results['accuracy']
    f1_score = results['f1_score']

    if accuracy > 0.9:
        insights['performance'] = 'Excellent'
    elif 0.75 <= accuracy <= 0.9:
        insights['performance'] = 'Good'
    else:
        insights['performance'] = 'Needs Improvement'

    if f1_score > 0.8:
        insights['balance'] = 'Balanced'
    else:
        insights['balance'] = 'Imbalance Detected'
    
    return insights

# Generate insights report
def generate_insights_report(modality, results):
    if results is None:
        logging.warning(f"No results for {modality}; skipping insights report generation.")
        return
    
    insights = generate_performance_insights(results)
    report_path = os.path.join(output_dir, f'{modality}_insights.txt')

    with open(report_path, 'w') as report_file:
        report_file.write(f"Insights for {modality.capitalize()} Modality\n")
        report_file.write("=" * 50 + "\n")
        for key, value in insights.items():
            report_file.write(f"{key.capitalize()}: {value}\n")
        report_file.write("=" * 50 + "\n")
    
    logging.info(f"Insights report generated for {modality} at {report_path}")

# Main process to run all reporting tasks
def main():
    logging.info("Starting report generation process")
    
    for modality in MODALITIES:
        results = load_results(modality)
        generate_report(modality, results)
        generate_plots(modality, results)
        generate_confusion_matrix_heatmap(modality, results)
        generate_insights_report(modality, results)
    
    generate_summary_report(MODALITIES)
    logging.info("All reports and summaries generated successfully")

if __name__ == "__main__":
    main()