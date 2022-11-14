model=GraghSage  # [GAT, GCN, GPN, GraghSage]
python -u main_train.py --model $model --episodes 500 > $model.log