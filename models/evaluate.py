import os
import torch
from torch.utils.data import DataLoader
from models import text, image, audio, multimodal
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix
from data.data_loader import load_dataset
import yaml
import json
import logging
from datetime import datetime

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = self.load_models()
        self.dataloaders = self.load_datasets()
        self.setup_logging()

    def setup_logging(self):
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.info('Evaluation started.')

    def load_models(self):
        models = {}
        logging.info('Loading models...')
        if 'text' in self.config['modalities']:
            models['text'] = text.GPTModel().to(self.device)
            logging.info('Text model loaded.')
        if 'image' in self.config['modalities']:
            models['image'] = image.CNNModel().to(self.device)
            logging.info('Image model loaded.')
        if 'audio' in self.config['modalities']:
            models['audio'] = audio.RNNModel().to(self.device)
            logging.info('Audio model loaded.')
        if 'multimodal' in self.config['modalities']:
            models['multimodal'] = multimodal.FusionModel().to(self.device)
            logging.info('Multimodal model loaded.')
        
        for modality, model in models.items():
            checkpoint = torch.load(os.path.join(self.config['pretrained_dir'], f'{modality}_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logging.info(f'{modality} model checkpoint loaded.')

        return models

    def load_datasets(self):
        dataloaders = {}
        logging.info('Loading datasets...')
        for modality in self.config['modalities']:
            dataset = load_dataset(modality, self.config['data_path'])
            dataloaders[modality] = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
            logging.info(f'{modality} dataset loaded.')
        return dataloaders

    def evaluate(self):
        logging.info('Evaluation started...')
        results = {}
        for modality, dataloader in self.dataloaders.items():
            logging.info(f'Evaluating {modality} modality...')
            metrics = self.evaluate_modality(modality, dataloader)
            results[modality] = metrics
            self.save_metrics(modality, metrics)
        logging.info('Evaluation completed.')
        return results

    def evaluate_modality(self, modality, dataloader):
        model = self.models[modality]
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds)
        logging.info(f'Metrics for {modality}: {metrics}')
        self.visualize_results(all_labels, all_preds, modality)
        return metrics

    def visualize_results(self, labels, preds, modality):
        logging.info(f'Visualizing results for {modality}...')
        plot_confusion_matrix(labels, preds, title=f'{modality} Confusion Matrix')
        logging.info(f'{modality} Confusion Matrix saved.')

    def save_metrics(self, modality, metrics):
        metrics_dir = self.config.get('metrics_dir', 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f'{modality}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f'{modality} metrics saved to {metrics_file}.')

    def summarize_results(self, results):
        summary_file = os.path.join(self.config['metrics_dir'], 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f'Summary of evaluation results saved to {summary_file}.')

    def log_hyperparameters(self):
        logging.info('Logging hyperparameters...')
        params_file = os.path.join(self.config['log_dir'], 'hyperparameters.yaml')
        with open(params_file, 'w') as f:
            yaml.dump(self.config, f)
        logging.info('Hyperparameters logged.')

if __name__ == '__main__':
    config_path = 'configs/evaluate_config.yaml'
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize evaluator
    evaluator = Evaluator(config)

    # Log hyperparameters
    evaluator.log_hyperparameters()

    # Perform evaluation
    evaluation_results = evaluator.evaluate()

    # Summarize results
    evaluator.summarize_results(evaluation_results)

    # Print final metrics
    for modality, metrics in evaluation_results.items():
        print(f"Metrics for {modality} modality: {metrics}")
        logging.info(f"Final metrics for {modality}: {metrics}")