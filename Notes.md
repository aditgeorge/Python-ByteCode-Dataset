clean_dataset
make_dataset
finetune

tmux ls

tmux new -s trainer

Ctrl + B, then D

tmux attach -t trainer

tmux kill-session -t trainer

To finetune:
python finetune.py 2>&1 | tee "./logs/finetune_log_$(date +'%Y-%m-%d_%H-%M-%S').txt"

or 
chmod +x finetune
./finetune
