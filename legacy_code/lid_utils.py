import os
import pathlib
import logging
import json
import random
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd

random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# POS tag constants
LID_TAGS = {
    'h': 0,
    'e': 1,
    'u': 2,
}

def save_json(dataset, path):
    """Save dataset to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        logger.info(f"Dataset saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")

def load_json(path):
    """Load dataset from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return None

# def mix_data(args, test=False, sample_size=250):
#     """Load and process COMI-LINGUA dataset."""
#     try:
#         # Load COMI-LINGUA dataset
#         dataset = load_dataset("LingoIITGN/COMI-LINGUA", 'POS')
#         split = 'test' if test else 'train'
#         data = dataset[split]
        
#         processed_data = []
#         for item in data:
#             # Process annotations
#             annotations = eval(item['Annotated by: Annotator 2'])
            
#             # Create comma-separated string of word-tag pairs
#             word_tag_pairs = [f"{ann['word']} {ann['entity']}" for ann in annotations]
#             labels = ", ".join(word_tag_pairs)
            
#             sample = {
#                 'input': item['Sentences'],
#                 'labels': labels  # String format: "word1 TAG1, word2 TAG2, word3 TAG3"
#             }
            
#             processed_data.append(sample)
        
#         # Subsample if needed
#         if test:
#             random.shuffle(processed_data)
#             processed_data = processed_data[:sample_size]
        
#         return processed_data
    
#     except Exception as e:
#         logger.error(f"Error processing dataset: {e}")
#         return []


def mix_data(args, test=False, sample_size=250):
    """Load and process data from CSV files."""
    try:
        # Correct file names and paths (use LID/ subdirectory)
        file_name = 'LID_Test.csv' if test else 'LID_with_consolidated.csv'
        file_path = os.path.join(args.data_dir, file_name)
        data = pd.read_csv(file_path)
        
        processed_data = []
        for _, item in data.iterrows():
            try:
                # Process annotations
                annotations = eval(item['consolidated_LID_tags'])
                
                # Create space-separated string of word-tag pairs
                word_tag_pairs = [f"{ann['key']} {ann['value']}" for ann in annotations]
                labels = " ".join(word_tag_pairs)  # Using space as separator
                
                sample = {
                    'input': item['Sentences'],
                    'labels': labels  # String format: "word1 TAG1 word2 TAG2 word3 TAG3"
                }
                
                processed_data.append(sample)
            except Exception as e:
                logger.warning(f"Skipping malformed entry: {e}")
                continue
        
        # Subsample if needed for test set
        if test and sample_size > 0:
            random.shuffle(processed_data)
            processed_data = processed_data[:sample_size]
        
        logger.info(f"Processed {len(processed_data)} examples from {file_name}")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return []

def calculate_metrics(predictions, gold_labels):
    """Calculate LID tagging metrics."""
    accuracy = accuracy_score(gold_labels, predictions)
    p, r, f1, _ = precision_recall_fscore_support(
        gold_labels, 
        predictions,
        average='weighted',
        zero_division=0
    )
    return {
        'accuracy': float(accuracy),
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1)
    }

def save_ner_metrics(predictions, path):
    """Save LID tagging evaluation metrics."""
    try:
        # Extract predictions and gold labels
        all_preds = []
        all_labels = []
        for item in predictions:
            all_preds.extend(item['predicted_tags'])
            all_labels.extend(item['labels'])
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_labels)
        
        # Generate detailed report
        report = classification_report(
            all_labels, 
            all_preds,
            target_names=list(LID_TAGS.keys()),
            output_dict=True
        )
        
        # Save results
        results = {
            'overall_metrics': metrics,
            'per_class_metrics': report
        }
        save_json(results, path)
        
        # Log summary
        logger.info(f"Overall F1: {metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
