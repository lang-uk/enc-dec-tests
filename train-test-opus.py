"""
Fine-tuning and evaluation scripts for OpusMT model on English-Ukrainian translation.
Supports bidirectional training and evaluation using SacreBLEU metrics.

This script provides functionality for:
- Loading and preprocessing parallel corpora from jsonlines files
- Fine-tuning OpusMT model for machine translation
- Evaluating translation quality using SacreBLEU and FLORES
"""

import json
import random
import argparse
from typing import Optional, Dict, List
from pathlib import Path

from pyaml import yaml
from smart_open import open
from tqdm.auto import tqdm

import torch
from transformers import (
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import sacrebleu
from datasets import Dataset as HFDataset, load_dataset
import wandb

# Constants
MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-zle"
WANDB_PROJECT = "opus-finetune"


# def get_inverse_square_root_schedule_with_warmup(
#     optimizer, num_warmup_steps: int, decay_start: int, last_epoch: int = -1
# ):
#     """Create a schedule with inverse square root learning rate decay after warmup.

#     Args:
#         optimizer: The optimizer for which to schedule the learning rate
#         num_warmup_steps: The number of steps for linear warmup
#         decay_start: The step to start the inverse square root decay
#         last_epoch: The index of the last epoch

#     Returns:
#         A learning rate scheduler
#     """

#     def lr_lambda(current_step: int):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(0.0, math.sqrt(decay_start / float(max(current_step, decay_start))))

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_training_config(args: argparse.Namespace, output_dir: str) -> None:
    """
    Save training configuration to YAML file.

    Args:
        args: Parsed command line arguments
        output_dir: Directory to save the training configuration
    Returns:
        None
    """

    config = vars(args)
    output_path = Path(output_dir) / "training_config.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(metrics: Dict, output_path: str) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path to save the metrics
    Returns:
        None
    """

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_and_sort_data(
    data_path: str, sort_mode: Optional[str] = None, seed: int = 42
) -> List[Dict]:
    """
    Load and optionally sort the training data based on scores.

    Args:
        data_path: Path to jsonlines file containing parallel data
        sort_mode: Sorting mode ('random', 'low-to-high', 'high-to-low', None)
        seed: Random seed for shuffling

    Returns:
        List of dictionaries containing source and target texts
    """
    raw_data = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            raw_data.append(
                {
                    "source": data["src"],
                    "target": data["mt"],
                    "score": data.get("wmt23-cometkiwi-da-xxl_score", 0.0),
                }
            )

    if sort_mode:
        rng = random.Random(seed)
        if sort_mode == "random":
            rng.shuffle(raw_data)
        elif sort_mode == "low-to-high":
            raw_data.sort(key=lambda x: x["score"])
        elif sort_mode == "high-to-low":
            raw_data.sort(key=lambda x: x["score"], reverse=True)

    return raw_data


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

    # Prepare both source and target texts using prepare_seq2seq_batch
    model_inputs = tokenizer.prepare_seq2seq_batch(
        src_texts=prefixed_sources,
        tgt_texts=examples["target"],
        max_length=max_length,
        max_target_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # Return python lists for dataset mapping
    )

    return model_inputs


def train_model(
    data_path: str,
    output_dir: str,
    source_lang: str = "eng",
    target_lang: str = "ukr",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    beta1: float = 0.9,
    beta2: float = 0.98,
    epsilon: float = 1e-9,
    max_grad_norm: float = 5.0,
    max_length: int = 256,
    wandb_project: Optional[str] = WANDB_PROJECT,
    sort_mode: Optional[str] = None,
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
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        epsilon: Adam epsilon parameter
        max_grad_norm: Maximum gradient norm for clipping
        max_length: Maximum sequence length
        wandb_project: Weights & Biases project name
        sort_mode: Data sorting mode ('random', 'low-to-high', 'high-to-low', None)
    """

    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load and preprocess the data
    raw_data = load_and_sort_data(data_path, sort_mode)

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
        num_proc=32,
    )

    # Calculate total number of training steps
    num_training_steps = len(dataset) // batch_size
    num_warmup_steps = int(num_training_steps * num_epochs * 0.02)

    print(f"Total training steps per epoch: {num_training_steps}")
    print(f"Total warmup steps: {num_warmup_steps}")
    print(f"Total training steps: {num_training_steps * num_epochs}")

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
                "warmup_steps": num_warmup_steps,
                "decay_steps": num_warmup_steps,
                "beta1": beta1,
                "beta2": beta2,
                "epsilon": epsilon,
                "max_grad_norm": max_grad_norm,
                "max_length": max_length,
            },
        )

    # Training arguments with updated optimizer settings
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        save_total_limit=num_epochs,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb" if wandb_project else "none",
        logging_steps=100,
        max_grad_norm=max_grad_norm,
        remove_unused_columns=False,
        prediction_loss_only=True,
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=num_warmup_steps,
        adam_beta1=beta1,
        adam_beta2=beta2,
        adam_epsilon=epsilon,
    )

    # # Initialize optimizer
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=learning_rate,
    #     betas=(beta1, beta2),
    #     eps=epsilon,
    # )

    # # Create custom scheduler
    # scheduler = get_inverse_square_root_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     decay_start=num_warmup_steps,
    # )

    # Initialize trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Custom Trainer class to use our scheduler
    # class CustomTrainer(Seq2SeqTrainer):
    #     def create_optimizer_and_scheduler(self, num_training_steps: int):
    #         self.optimizer = optimizer
    #         self.lr_scheduler = scheduler

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

    # Run evaluation after training
    print("\nRunning post-training evaluation...")
    metrics = evaluate_and_save_metrics(
        model_path=str(output_dir),
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=10,  # Default value for evaluation
        decode_subset="dev",
        output_dir=str(output_dir),
    )
    print(f"\nFinal BLEU score: {metrics['bleu_score']:.2f}")


def evaluate_and_save_metrics(
    model_path: str,
    source_lang: str,
    target_lang: str,
    batch_size: int,
    max_length: int,
    num_beams: int,
    decode_subset: str,
    output_dir: str,
) -> Dict:
    """
    Evaluate model and save comprehensive metrics.

    Returns:
        Dictionary containing all evaluation metrics
    """
    output_file = Path(output_dir) / f"beam_outputs_{decode_subset}.jsonl"
    metrics_file = Path(output_dir) / f"metrics_{decode_subset}.json"

    bleu_score = evaluate_model(
        model_path=model_path,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams,
        decode_subset=decode_subset,
        output_file=str(output_file),
    )

    # Get detailed metrics from the latest evaluation
    metrics = {
        "bleu_score": bleu_score,
        "bleu_details": {
            "moses": {
                "score": bleu_score,
                "tokenizer": "13a",
            },
            "spm": {
                "score": sacrebleu.corpus_bleu(
                    hypotheses,  # This needs to be passed from evaluate_model
                    [references],  # This needs to be passed from evaluate_model
                    tokenize="spm",
                ).score,
                "tokenizer": "spm",
            },
        },
        "dataset_split": decode_subset,
        "model_path": model_path,
        "parameters": {
            "beam_size": num_beams,
            "max_length": max_length,
            "batch_size": batch_size,
        },
    }

    save_metrics(metrics, metrics_file)
    return metrics


def evaluate_model(
    model_path: str,
    output_dir: Path,
    source_lang: str = "eng",
    target_lang: str = "ukr",
    batch_size: int = 8,
    max_length: int = 256,
    num_beams: int = 10,
    decode_subset: str = "dev",
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

    output_file = Path(output_dir) / f"beam_outputs_{decode_subset}.jsonl"
    metrics_file = Path(output_dir) / f"metrics_{decode_subset}.json"

    print(f"Loading model from {model_path}")

    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

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
            source = batch[f"sentence_{flores_source}"][idx]

            # Decode all beams for this example
            beam_translations = []
            for beam_idx in range(num_beams):
                # Get token IDs for this beam
                beam_tokens = sequences[idx, beam_idx].cpu().numpy()
                # Decode tokens to text
                translation = tokenizer.decode(beam_tokens, skip_special_tokens=True)
                score = scores[idx, beam_idx].item()
                beam_translations.append({"translation": translation, "score": score})

            # Store beam outputs
            beam_outputs.append(
                {
                    "id": example_id,
                    "source": source,
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
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize="13a",  # Use standard Moses tokenizer from sacrebleu
    )

    # Also calculate BLEU with different tokenizers for comparison
    bleu_spm = sacrebleu.corpus_bleu(
        hypotheses, [references], tokenize="spm"  # SentencePiece tokenizer
    )

    print(f"BLEU (Moses tokenizer): {bleu.score:.2f}")
    print(f"BLEU (SentencePiece tokenizer): {bleu_spm.score:.2f}")
    print("\nDetailed BLEU scores with Moses tokenizer:")
    print(bleu.format())

    metrics = {
        "bleu_score": bleu.score,
        "bleu_details": {
            "moses": {
                "score": bleu.score,
                "tokenizer": "13a",
            },
            "spm": {
                "score": bleu_spm.score,
                "tokenizer": "spm",
            },
        },
        "dataset_split": decode_subset,
        "model_path": model_path,
        "parameters": {
            "beam_size": num_beams,
            "max_length": max_length,
            "batch_size": batch_size,
        },
    }

    save_metrics(metrics, metrics_file)

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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

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
        "--sort-mode",
        choices=["random", "low-to-high", "high-to-low"],
        help="Data sorting mode",
    )

    # Learning rate and scheduler parameters
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Learning rate"
    )

    # Optimizer parameters
    train_parser.add_argument(
        "--beta1", type=float, default=0.9, help="Adam beta1 parameter"
    )
    train_parser.add_argument(
        "--beta2", type=float, default=0.98, help="Adam beta2 parameter"
    )
    train_parser.add_argument(
        "--epsilon", type=float, default=1e-9, help="Adam epsilon parameter"
    )
    train_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping",
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
        "--output-dir", type=Path, help="Path to save beam outputs"
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
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            wandb_project=args.wandb_project,
            sort_mode=args.sort_mode,
        )
    elif args.mode == "eval":
        bleu_score = evaluate_model(
            output_dir=args.output_dir,
            model_path=args.model_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            decode_subset=args.decode_subset,
            output_dir=args.output_dir,
        )
        print(f"BLEU Score: {bleu_score:.2f}")
