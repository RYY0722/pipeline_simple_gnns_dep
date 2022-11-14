### Main structure from

[kaize0409/GPN_Graph-Few-shot: Implementation of CIKM2020 -- Graph Prototypical Networks for Few-shot Learning on Attributed Networks (github.com)](https://github.com/kaize0409/GPN_Graph-Few-shot)

### Models implementation from

[colflip/gnns_fewshot: code implementation of GNNs in few-shot learning: GCN, GAT, GraphSAGE to the node classification task of some datasets. (github.com)](https://github.com/colflip/gnns_fewshot)

Note: for this, the following dependencies are required

- Python ≥ 3.10
- PyTorch ≥ 11.3
- pyg ≥ 1.12.0

My settings:

- Python 3.9

- PyTorch    1.10.1+cpu

- for pyg, I follow the following link [关于protobuf报错：If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. - BooTurbo - 博客园 (cnblogs.com)](https://www.cnblogs.com/booturbo/p/16339195.html)

- Used the below command to solve some bugs (forgot the details.. )

  ​    pip3 install --upgrade protobuf==3.20.1

### Train a model

sh train.sh

- Model: specify in *train.sh* 
- Datasets: specify in *main_train.py*

