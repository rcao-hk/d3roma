# accelerate launch train.py \
#   task=train_dreds_reprod \
#   task.tag=release \
#   task.eval_num_batch=10 \
#   task.val_every_global_steps=5000
accelerate launch train.py \
  task=train_hammer \
  task.tag=hamer \
  task.train_batch_size=8 \
  task.eval_num_batch=8 \
  task.val_every_global_steps=1000