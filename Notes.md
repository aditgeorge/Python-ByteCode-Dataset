clean_dataset
make_dataset
finetune

tmux ls

tmux new -s trainer

Ctrl + B, then D

tmux attach -t trainer

python finetune.py 2>&1 | tee finetune_log.txt 