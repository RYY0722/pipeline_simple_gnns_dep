model=GPN  # [GAT, GCN, GPN, GraghSage]
python -u main_train.py --model $model --episodes 500 --num_repeat 5 > logs/$model.log
echo $!