# Development of domestic thesis sentence semantic tagging model
## Introduction
- To automate the meaning tagging of the domestic thesis sentence by predicting the rhetorical category of a thesis sentence.
- Hierarchical embedding structure and multiple loss functions are used to represent the meaning of rhetorical categories.
  
## Dataset description
There are a total of 155,740 thesis sentences and tag pairs, and the semantic tags form a hierarchical structure with semantic structure classification/detailed semantic classification.

## Main strategy
1. Constructed text representation for thesis sentences using KorSciBert and GCN.
2. Label embedding is constructed to extract the label semantic representation.
3. Multiple loss function was constructed to reflect hierarchical properties through label semantic distance.
   - Classification loss : We predicted labels using only text representation.
   - Join embedding loss : We minimized the distance between text semantics and target label semantics within the same embedding space.
   - Matching loss : We put distance between text semantics and incorrect label semantics.
   
### Directory Structure
```
/root/workspace
├── data
│    ├── csv
│    │    ├── train.csv
│    │    ├── dec.csv
│    │    ├── test.csv
│    │    └── label_desc.csv
│    ├── hierar
│    │    ├── hierar_prob.json
│    │    ├── hierar.txt
│    │    ├── label.dict
│    │    ├── label_i2v.pickle
│    │    └── label_v2i.pickle
│    └── make_df.py 
│
├── src
│    ├── models
│    │    ├── pretrained_model
│    │    │    └── korscibert
│    │    │         ├── bert_config_kisti.json
│    │    │         ├── pytorch_model.bin
│    │    │         ├── tokenization_kisti.py
│    │    │         └── vocab_kisti.txt
│    │    │   
│    │    ├── structure_model
│    │    │    ├── graphcnn.py
│    │    │    ├── structure_encoder.py
│    │    │    └── tree.py
│    │    │    
│    │    ├── matching_network.py
│    │    ├── model.py
│    │    └── text_feature_propagation.py
│    │   
│    ├── utils
│    │    ├── configure.py
│    │    ├── evaluation_modules.py
│    │    ├── hierarchy_tree_stastistic.py
│    │    ├── train_modules.py
│    │    └── utils.py
│    │  
│    ├── config.json
│    ├── dataloader.py
│    ├── main.py
│    └── trainer.py
│
└── sen_cls.yaml
```

### How to Use

1. Create Environment & Import Library
    ```
    conda env create -f sen_cls.yaml
    conda activate sen_cls
    pip install torch==1.8.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html
    ```
2. Training
   ```
   python main.py --do_train=True --exp_num='exp'
   ```
3. Test
   ```
   python main.py --do_test=True --exp_num='exp0' 
   ```
4. Predict
   ```
   python main.py --do_predict=True --exp_num='0'  
   ```

