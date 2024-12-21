"""
Fine-tuning and evaluation scripts for OpusMT model on English-Ukrainian translation.
Supports bidirectional training and evaluation using SacreBLEU metrics.
"""

import json
import argparse
from typing import Optional, Dict, List
from pathlib import Path

import yaml
from smart_open import open
from tqdm.auto import tqdm
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
import torch
import os
from transformers import (
    MarianTokenizer,
    MarianMTModel,
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


def save_training_config(args: argparse.Namespace, output_dir: str) -> None:
    """Save training configuration to YAML file."""
    config = vars(args)
    output_path = Path(output_dir) / "training_config.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save evaluation metrics to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_data(data_path: str) -> List[Dict]:
    """Load the training data from jsonlines file."""
    raw_data = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            raw_data.append({"source": data["en"], "target": data["uk"]})
    return raw_data


def filter_long_sequences(
    dataset: HFDataset, tokenizer: MarianTokenizer, max_length: int
) -> HFDataset:
    """Remove examples where either source or target exceeds max_length tokens."""

    def is_valid_length(example):
        src_tokens = tokenizer(example["source"], truncation=False)["input_ids"]
        tgt_tokens = tokenizer(example["target"], truncation=False)["input_ids"]
        return len(src_tokens) <= max_length and len(tgt_tokens) <= max_length

    return dataset.filter(is_valid_length)


def prepare_dataset(
    examples: Dict,
    tokenizer: MarianTokenizer,
    max_length: int,
    source_lang: str,
    target_lang: str,
) -> Dict:
    """Prepare dataset for training by tokenizing inputs and targets."""
    prefixed_sources = [f">>{target_lang}<< {src}" for src in examples["source"]]

    model_inputs = tokenizer.prepare_seq2seq_batch(
        src_texts=prefixed_sources,
        tgt_texts=examples["target"],
        max_length=max_length,
        max_target_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )

    return model_inputs


def objective(
    trial: optuna.Trial,
    train_dataset: HFDataset,
    model: AutoModelForSeq2SeqLM,
    tokenizer: MarianTokenizer,
    args: argparse.Namespace,
) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    scheduler = trial.suggest_categorical(
        "scheduler", ["linear", "cosine", "cosine_with_restarts"]
    )

    # Calculate steps
    num_training_steps = len(train_dataset) // args.batch_size * args.num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/trial_{trial.number}",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=lr,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb" if args.wandb_project else "none",
        logging_steps=100,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        prediction_loss_only=True,
        lr_scheduler_type=scheduler,
        warmup_steps=num_warmup_steps,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.epsilon,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        tokenizer=tokenizer,
    )

    # Get GPU ID based on worker ID
    gpu_id = trial.number % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    print(f"Running trial {trial.number} on GPU {gpu_id}")

    # Train and evaluate
    trainer.train()

    # Evaluate on dev set
    trainer.save_model()

    bleu_score = evaluate_model(
        model_path=f"{args.output_dir}/trial_{trial.number}",
        output_dir=Path(args.output_dir),
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=10,
        decode_subset="dev",
    )

    # Clean up
    del trainer
    torch.cuda.empty_cache()

    return bleu_score


def train_model(
    data_path: str,
    output_dir: str,
    source_lang: str = "eng",
    target_lang: str = "ukr",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    beta1: float = 0.9,
    beta2: float = 0.998,
    epsilon: float = 1e-9,
    max_grad_norm: float = 5.0,
    max_length: int = 256,
    wandb_project: Optional[str] = WANDB_PROJECT,
    from_scratch: bool = False,
    n_trials: int = 20,
) -> None:
    """Fine-tune OpusMT model with hyperparameter optimization."""

    # Load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    # Load or initialize model
    if from_scratch:
        config = MarianMTModel.from_pretrained(MODEL_NAME).config
        model = MarianMTModel(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load and preprocess the data
    raw_data = load_data(data_path)
    dataset = HFDataset.from_list(raw_data)

    # Filter long sequences
    print(f"Dataset size before filtering: {len(dataset)}")
    dataset = filter_long_sequences(dataset, tokenizer, max_length)
    print(f"Dataset size after filtering: {len(dataset)}")

    # Add tokenization with caching
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

    if wandb_project:
        wandb.init(project=wandb_project)

    # Set up storage for parallel optimization
    storage = RDBStorage(
        "sqlite:///optuna_study.db", heartbeat_interval=60, grace_period=120
    )

    # Create Optuna study with storage
    study = optuna.create_study(
        study_name="opus_mt_optimization",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    # Create args object for the objective function
    args = argparse.Namespace(
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_grad_norm=max_grad_norm,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        wandb_project=wandb_project,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset, model, tokenizer, args),
        n_trials=n_trials,
    )

    # Train final model with best parameters
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")

    # Clean up before final training
    torch.cuda.empty_cache()

    # Train final model with best parameters
    if from_scratch:
        model = MarianMTModel(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    final_training_args = Seq2SeqTrainingArguments(
        output_dir=f"{output_dir}/final_model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=best_params["learning_rate"],
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb" if wandb_project else "none",
        logging_steps=100,
        max_grad_norm=max_grad_norm,
        remove_unused_columns=False,
        prediction_loss_only=True,
        lr_scheduler_type=best_params["scheduler"],
        warmup_steps=int(
            len(dataset) // batch_size * num_epochs * best_params["warmup_ratio"]
        ),
        adam_beta1=beta1,
        adam_beta2=beta2,
        adam_epsilon=epsilon,
    )

    final_trainer = Seq2SeqTrainer(
        model=model,
        args=final_training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        tokenizer=tokenizer,
    )

    final_trainer.train()
    final_trainer.save_model()

    if wandb_project:
        wandb.finish()

    # Clean up before evaluation
    del model
    del final_trainer
    torch.cuda.empty_cache()

    # Run evaluation
    print("\nRunning post-training evaluation...")
    evaluate_model(
        model_path=f"{output_dir}/final_model",
        output_dir=Path(output_dir),
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=10,
        decode_subset="dev",
    )


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
    """Evaluate the fine-tuned model using SacreBLEU on FLORES dataset."""
    output_file = Path(output_dir) / f"beam_outputs_{decode_subset}.jsonl"
    metrics_file = Path(output_dir) / f"metrics_{decode_subset}.json"

    print(f"Loading model from {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    flores_lang_map = {"eng": "eng_Latn", "ukr": "ukr_Cyrl"}
    flores_source = flores_lang_map[source_lang]
    flores_target = flores_lang_map[target_lang]

    dataset = load_dataset(
        "facebook/flores", f"{flores_source}-{flores_target}", trust_remote_code=True
    )[decode_subset]

    columns = ["id", f"sentence_{flores_source}", f"sentence_{flores_target}"]
    dataset = dataset.select_columns(columns)

    hypotheses = []
    references = []
    beam_outputs = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(dataset), batch_size), total=num_batches, desc="Evaluating"
    ):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
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

        sequences = outputs.sequences.reshape(len(batch), num_beams, -1)
        scores = torch.exp(outputs.sequences_scores).reshape(len(batch), num_beams)

        for idx in range(len(batch)):
            example_id = batch["id"][idx]
            reference = batch[f"sentence_{flores_target}"][idx]
            source = batch[f"sentence_{flores_source}"][idx]

            beam_translations = []
            for beam_idx in range(num_beams):
                beam_tokens = sequences[idx, beam_idx].cpu().numpy()
                translation = tokenizer.decode(beam_tokens, skip_special_tokens=True)
                score = scores[idx, beam_idx].item()
                beam_translations.append({"translation": translation, "score": score})

            beam_outputs.append(
                {
                    "id": example_id,
                    "source": source,
                    "reference": reference,
                    "beams": beam_translations,
                }
            )

            hypotheses.append(beam_translations[0]["translation"])
            references.append(reference)

    with open(output_file, "w") as f:
        for output in beam_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

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

    # Clean up
    del model
    torch.cuda.empty_cache()

    return bleu.score


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "--from-scratch", action="store_true", help="Train model from scratch"
    )
    train_parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of Optuna trials"
    )
    train_parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel jobs for Optuna"
    )
    train_parser.add_argument(
        "--study-name",
        type=str,
        default="opus_mt_optimization",
        help="Name for the Optuna study",
    )
    train_parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="Storage URL for Optuna database",
    )

    # Learning rate and optimizer parameters
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Initial learning rate"
    )
    train_parser.add_argument(
        "--beta1", type=float, default=0.9, help="Adam beta1 parameter"
    )
    train_parser.add_argument(
        "--beta2", type=float, default=0.998, help="Adam beta2 parameter"
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
        "--output-dir", type=Path, required=True, help="Path to save beam outputs"
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
            from_scratch=args.from_scratch,
            n_trials=args.n_trials,
        )
    elif args.mode == "eval":
        bleu_score = evaluate_model(
            model_path=args.model_path,
            output_dir=args.output_dir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
            decode_subset=args.decode_subset,
        )
        print(f"Final BLEU Score: {bleu_score:.2f}")
