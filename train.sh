python train-test.py \
    --source-lang eng_Latn \
    --target-lang ukr_Cyrl \
    --batch-size 8 \
    train \
    --data-path data/filtered_pairs_500_001.jsonl.bz2 \
    --output-dir exps/claude_en-uk_500k \
    --num-epochs 3 \
    --wandb-project nmt-finetune