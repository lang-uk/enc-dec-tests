"""
Fine-tuning and evaluation scripts for NLLB model on English-Ukrainian translation.
Supports bidirectional training and evaluation using SacreBLEU metrics.

This script provides functionality for:
- Loading and preprocessing parallel corpora from jsonlines files
- Fine-tuning NLLB model for machine translation
- Evaluating translation quality using SacreBLEU
"""

import json
import argparse
from typing import  Optional, Dict
from smart_open import open
import torch
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import pandas as pd
import sacrebleu
from datasets import Dataset as HFDataset
import wandb

# Constants
# MODEL_NAME = "facebook/nllb-200-3.3B"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

def prepare_dataset(examples: Dict,
                   tokenizer: NllbTokenizer,
                   max_length: int,
                   source_lang: str,
                   target_lang: str) -> Dict:
    """
    Prepare dataset for training by tokenizing inputs and targets.
    
    Args:
        examples: Dictionary containing source and target texts
        tokenizer: NLLB tokenizer instance
        max_length: Maximum sequence length for tokenization
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        Dictionary containing tokenized inputs and labels
    """
    # Tokenize sources
    tokenizer.src_lang = source_lang
    model_inputs = tokenizer(
        examples['source'],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # Tokenize targets
    tokenizer.src_lang = target_lang
    labels = tokenizer(
        examples['target'],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(
    data_path: str,
    output_dir: str,
    source_lang: str = "eng_Latn",
    target_lang: str = "ukr_Cyrl",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    warmup_steps: int = 500,
    wandb_project: Optional[str] = None
) -> None:
    """
    Fine-tune NLLB model on the provided parallel corpus.
    
    Args:
        data_path: Path to jsonlines file containing parallel data
        output_dir: Directory to save the fine-tuned model
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for optimization
        max_length: Maximum sequence length
        warmup_steps: Number of warmup steps for learning rate scheduler
        wandb_project: Weights & Biases project name (optional)
    """
    # Initialize wandb if project name is provided
    if wandb_project:
        wandb.init(
            project=wandb_project,
            config={
                "model": MODEL_NAME,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_length": max_length,
                "warmup_steps": warmup_steps
            }
        )
    
    # Load model and tokenizer
    tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Load and preprocess the data
    raw_data = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            raw_data.append({
                'source': data['src'],
                'target': data['mt']
            })
    
    # Convert to HuggingFace dataset format
    dataset = HFDataset.from_list(raw_data)
    
    # Add tokenization parameters to dataset
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': max_length,
            'source_lang': source_lang,
            'target_lang': target_lang
        },
        batched=True
    )
    
    # Calculate total number of training steps
    num_training_steps = (len(dataset) // batch_size) * num_epochs
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb" if wandb_project else "none",
        logging_steps=100,
        warmup_steps=warmup_steps,
        remove_unused_columns=False,
        prediction_loss_only=True
    )
    
    # Initialize trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    if wandb_project:
        wandb.finish()

def evaluate_model(
    model_path: str,
    eval_file: str,
    source_lang: str = "eng_Latn",
    target_lang: str = "ukr_Cyrl",
    batch_size: int = 8,
    max_length: int = 128
) -> float:
    """
    Evaluate the fine-tuned model using SacreBLEU.
    
    Args:
        model_path: Path to the fine-tuned model
        eval_file: Path to CSV file with src,ref,hyp columns
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Batch size for inference
        max_length: Maximum sequence length
    
    Returns:
        BLEU score
    """
    # Load model and tokenizer
    tokenizer = NllbTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load evaluation data
    eval_data = pd.read_csv(eval_file)
    
    hypotheses = []
    references = []
    
    # Generate translations
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data.iloc[i:i + batch_size]
        
        tokenizer.src_lang = source_lang
        inputs = tokenizer(
            batch['src'].tolist(),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                max_length=max_length,
                num_beams=5
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hypotheses.extend(decoded)
        references.extend(batch['ref'].tolist())
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate NLLB model")
    
    # Common arguments
    parser.add_argument("--source-lang", default="eng_Latn", help="Source language code")
    parser.add_argument("--target-lang", default="ukr_Cyrl", help="Target language code")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-path", required=True, help="Path to training data")
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    train_parser.add_argument("--wandb-project", help="Weights & Biases project name")
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model-path", required=True, help="Path to trained model")
    eval_parser.add_argument("--eval-file", required=True, help="Path to evaluation CSV file")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "train":
        train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            warmup_steps=args.warmup_steps,
            wandb_project=args.wandb_project
        )
    elif args.mode == "eval":
        bleu_score = evaluate_model(
            model_path=args.model_path,
            eval_file=args.eval_file,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        print(f"BLEU Score: {bleu_score:.2f}")