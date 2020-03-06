# request a gpu
srun -t2:30:00 --mem=16000 --gres=gpu:1 --pty /bin/bash

# request 2 cpu
srun -c2 -t2:00:00 --mem=16000 --pty /bin/bash
