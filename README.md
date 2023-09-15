# 국내 논문 문장 의미 태깅 모델 개발

### 개요
- 주어진 논문 문장의 수사학적 카테고리를 예측하여 국내 논문 문장 의미 태깅 자동화를 하고자 함. 
- 수사학적 카테고리를 위해 계층적 임베딩 구조, 다중 손실 함수를 사용함.

### Training Environment
- python 3.7
- pytorch 1.8.0
- Ubuntu 18.04
- CUDA 11.1
- GeForce RTX 3090

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

### Model
- CrossValidation을 진행했으므로 총 5개의 checkpoint가 results 디렉토리에 존재
- model_0fold 의 경우, 0fold를 validation dataset으로 사용한 것임. 
- 사용코드는 src 디렉토리에 저장

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

### Arguments
- `--config_path` : 사용할 모델 parameter config
- `--exp_num` : 학습된 모델 저장 위치
- `--do_train` : 모델 학습
- `--do_test` : 모델 평가
- `--do_predict` : 데모 실행
