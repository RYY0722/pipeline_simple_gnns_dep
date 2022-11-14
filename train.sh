model=GCN  # [GAT, GCN, GPN, GraghSage]
python -u main_train.py --model $model --episodes 50 --num_repeat 2 > logs/$model.log