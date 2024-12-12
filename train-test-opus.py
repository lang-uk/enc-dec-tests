"""
Fine-tuning and evaluation scripts for OpusMT model on English-Ukrainian translation.
Supports bidirectional training and evaluation using SacreBLEU metrics.

This script provides functionality for:
- Loading and preprocessing parallel corpora from jsonlines files
- Fine-tuning OpusMT model for machine translation
- Evaluating translation quality using SacreBLEU and FLORES
"""

import json
import argparse
from typing import Optional, Dict, List
from smart_open import open
import torch
from transformers import (
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import pandas as pd
import sacrebleu
from datasets import Dataset as HFDataset, load_dataset
import wandb
from tqdm.auto import tqdm

# Constants
MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-zle"
WANDB_PROJECT = "opus-finetune"


def prepare_dataset(
    examples: Dict,
    tokenizer: MarianTokenizer,
    max_length: int,
    source_lang: str,
    target_lang: str,
) -> Dict:
    """
    Prepare dataset for training by tokenizing inputs and targets.

    Args:
        examples: Dictionary containing source and target texts
        tokenizer: OpusMT tokenizer instance
        max_length: Maximum sequence length for tokenization
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Dictionary containing tokenized inputs and labels
    """
    # Prepend target language token to source sentences
    prefixed_sources = [f">>{target_lang}<< {src}" for src in examples["source"]]

    # Tokenize sources
    model_inputs = tokenizer(
        prefixed_sources,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # Tokenize targets
    labels = tokenizer(
        examples["target"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(
    data_path: str,
    output_dir: str,
    source_lang: str = "eng",
    target_lang: str = "ukr",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    wandb_project: Optional[str] = WANDB_PROJECT,
) -> None:
    """
    Fine-tune OpusMT model on the provided parallel corpus.

    Args:
        data_path: Path to jsonlines file containing parallel data
        output_dir: Directory to save the fine-tuned model
        source_lang: Source language code (e.g., 'eng')
        target_lang: Target language code (e.g., 'ukr')
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for optimization
        max_length: Maximum sequence length
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
            },
        )

    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load and preprocess the data
    raw_data = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            raw_data.append({"source": data["src"], "target": data["mt"]})

    # Convert to HuggingFace dataset format
    dataset = HFDataset.from_list(raw_data)

    # Add tokenization parameters to dataset
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
        batched=True,
    )

    # Calculate total number of training steps
    num_training_steps = len(dataset) // batch_size
    num_warmup_steps = int(num_training_steps * 0.05)
    print(f"Total training steps per epoch: {num_training_steps}")
    print(f"Total warmup steps: {num_warmup_steps}")
    print(f"Total training steps: {num_training_steps * num_epochs}")

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
        warmup_steps=num_warmup_steps,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    # Initialize trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model()

    if wandb_project:
        wandb.finish()


def evaluate_model(
    model_path: str,
    source_lang: str = "eng",
    target_lang: str = "ukr",
    batch_size: int = 8,
    max_length: int = 256,
    num_beams: int = 10,
    decode_subset: str = "dev",
    output_file: str = "beam_outputs.jsonl",
) -> float:
    """
    Evaluate the fine-tuned model using SacreBLEU on FLORES dataset.

    Args:
        model_path: Path to the fine-tuned model
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        num_beams: Number of beams for beam search
        decode_subset: Subset of FLORES to use ('dev' or 'devtest')
        output_file: Path to save beam outputs

    Returns:
        BLEU score
    """
    print(f"Loading model from {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")

    # Convert language codes for FLORES
    flores_lang_map = {"eng": "eng_Latn", "ukr": "ukr_Cyrl"}
    flores_source = flores_lang_map[source_lang]
    flores_target = flores_lang_map[target_lang]

    print(f"Loading FLORES dataset ({decode_subset} split)")
    dataset = load_dataset(
        "facebook/flores", f"{flores_source}-{flores_target}", trust_remote_code=True
    )[decode_subset]

    # Select required columns
    columns = ["id", f"sentence_{flores_source}", f"sentence_{flores_target}"]
    dataset = dataset.select_columns(columns)

    hypotheses = []
    references = []
    beam_outputs = []

    # Calculate number of batches for progress bar
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # Generate translations
    for i in tqdm(
        range(0, len(dataset), batch_size), total=num_batches, desc="Evaluating"
    ):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))

        # Prepend target language token to source sentences
        sources = [
            f">>{target_lang}<< {src}" for src in batch[f"sentence_{flores_source}"]
        ]

        inputs = tokenizer(
            sources,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Get sequences and scores
        sequences = outputs.sequences.reshape(len(batch), num_beams, -1)
        scores = torch.exp(outputs.sequences_scores).reshape(len(batch), num_beams)

        # Process each example in the batch
        for idx in range(len(batch)):
            example_id = batch["id"][idx]
            reference = batch[f"sentence_{flores_target}"][idx]

            # Decode all beams for this example
            beam_translations = []
            for beam_idx in range(num_beams):
                translation = tokenizer.decode(
                    sequences[idx, beam_idx], skip_special_tokens=True
                )
                score = scores[idx, beam_idx].item()
                beam_translations.append({"translation": translation, "score": score})

            # Store beam outputs
            beam_outputs.append(
                {
                    "id": example_id,
                    "source": batch[f"sentence_{flores_source}"][idx],
                    "reference": reference,
                    "beams": beam_translations,
                }
            )

            # Use top beam for BLEU calculation
            hypotheses.append(beam_translations[0]["translation"])
            references.append(reference)

    print(f"Saving outputs to {output_file}")
    with open(output_file, "w") as f:
        for output in beam_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate OpusMT model")

    # Common arguments
    parser.add_argument("--source-lang", default="eng", help="Source language code")
    parser.add_argument("--target-lang", default="ukr", help="Target language code")
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data-path", required=True, help="Path to training data"
    )
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    train_parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    train_parser.add_argument(
        "--wandb-project", default=WANDB_PROJECT, help="Weights & Biases project name"
    )

    # Evaluation arguments
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path", required=True, help="Path to trained model"
    )
    eval_parser.add_argument(
        "--decode-subset",
        default="dev",
        choices=["dev", "devtest"],
        help="FLORES subset to use for evaluation",
    )
    eval_parser.add_argument(
        "--num-beams", type=int, default=10, help="Number of beams for beam search"
    )
    eval_parser.add_argument(
        "--output-file", default="beam_outputs.jsonl", help="Path to save beam outputs"
    )

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
            wandb_project=args.wandb_project,
        )
    elif args.mode == "eval":
        bleu_score = evaluate_model(
            model_path=args.model_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            decode_subset=args.decode_subset,
            output_file=args.output_file,
        )
        print(f"BLEU Score: {bleu_score:.2f}")
