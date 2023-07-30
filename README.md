## Graph Convolutional Networks based Muti-Label Deep Cross-Modal Hashing


## Dependencies

- Python (>=3.8)

- PyTorch (>=1.7.1)

- Scipy (>=1.5.2)

## Datasets
You can download the features of the datasets from:
 - MIRFlickr, [BaiduPan(password: p4od)](https://pan.baidu.com/s/1w-zE69vU0F1g7Nihg05kLQ?pwd=p4od)
 - NUS-WIDE (top-21 concepts), [BaiduPan(password: yzx0)](https://pan.baidu.com/s/1d8dL17XwvUVyamEDMo3TNQ?pwd=yzx0yzx0)
 - embedding, [BaiduPan(password: 1obm)](https://pan.baidu.com/s/1KhGFYcYugiblT85_OWUmcQ?pwd=1obm)
 
## Implementation

Here we provide the implementation of our proposed models, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE, MIRFlickr;
 - `model` contains the implementation of the `gcn_net` , `img_module` and `txt_module`;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE or MS-COCO.

## Process
 - Place the datasets in `data/`
 - Set the experiment parameters in `main.py`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```

