python train-test.py \
    --source-lang eng_Latn \
    --target-lang ukr_Cyrl \
    --batch-size 16 \
    train \
    --data-path data/filtered_pairs_500_001.jsonl.bz2 \
    --output-dir exps/claude_en-uk_500k_lr-1e-4_wd-1e-3 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --weight-decay 1e-3 \
    --wandb-project nmt-finetune