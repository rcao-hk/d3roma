accelerate launch train.py \
  task=train_dreds_reprod \
  task.tag=release \
  task.eval_num_batch=10 \
  task.val_every_global_steps=5000