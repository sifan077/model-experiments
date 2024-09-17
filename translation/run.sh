python3 train.py \
    --output_dir=./trans_marian_results/ \
    --model_type=marian \
    --model_checkpoint=Helsinki-NLP/opus-mt-zh-en \
    --train_file=D:/code/dataset/translation2019zh/translation2019zh_train.json \
    --dev_file=D:/code/dataset/translation2019zh/translation2019zh_train.json \
    --test_file=D:/code/dataset/translation2019zh/translation2019zh_valid.json \
    --max_input_length=128 \
    --max_target_length=128 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=16 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42

python train.py --output_dir=./checkpoints/ --model_type=marian --model_checkpoint=Helsinki-NLP/opus-mt-zh-en --train_file=D:/code/dataset/translation2019zh/translation2019zh_train.json --dev_file=D:/code/dataset/translation2019zh/translation2019zh_train.json --test_file=D:/code/dataset/translation2019zh/translation2019zh_valid.json --max_input_length=128 --max_target_length=128 --learning_rate=1e-5 --num_train_epochs=3   --batch_size=16  --do_train --warmup_proportion=0.   --seed=42