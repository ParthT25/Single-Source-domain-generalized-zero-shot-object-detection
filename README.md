# Single-Source-domain-generalized-zero-shot-object-detection

### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and requires python >= 3.6

Install the required packages
```
pip install -r requirements.txt
```
Download [Diverse Weather](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)

Install Detectron2 from here [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Set dataset path for Detectron2 
```
DETECTRON2_DATASETS=/path/to/datasets
```
### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy
            /dusk_rainy
            ...
```

We train our models on a single A100 GPU.
```
    python train.py --config-file configs/diverse_weather.yaml
```
## After Training

Follow these steps:

1. Edit the value from 6 to 7 in the `configs/diverse_weather.yaml` file [here](https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/configs/diverse_weather.yaml#L31).

2. Add the `bus` class in the array in the `data/datasets/diverse_weather.py` file [here](https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/data/datasets/diverse_weather.py#L15).

3. Remove the commented-out code in the `modeling/meta_arch.py` file:
   - From lines [241 to 271](https://github.com/papz2000/Single-Source-domain-generalized-zero-shot-object-detection/blob/4d986b70f0c9fea48db8ae30cc107d7adc35ecd1/modeling/meta_arch.py#L238-L267]).
   - From lines [547 to 550](https://github.com/papz2000/Single-Source-domain-generalized-zero-shot-object-detection/blob/1ce7f3beff4c63b70c65048ad14f266e8c663890/modeling/meta_arch.py#L543-L547).

4. Run the code in evaluation mode to extract ROI feature maps:
   ```sh
   python train.py --eval-only --config-file configs/diverse_weather.yaml MODEL.WEIGHTS /u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/all_outs/diverse_weather/model_best.pth
   ```
Run the WC-DCGAN and generate the unseen class ROIs and fine tune clip attention pooling layer:
  ```sh
  cd WC-DCGAN
  python train.py
  python generate.py
  python retrain_clip_attn.py
  ```
Run the code in evaluation mode using the finetuned model
   ```sh
   python train.py --eval-only --config-file configs/diverse_weather.yaml MODEL.WEIGHTS /u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/Models/updated_clipattn.pth
   ```
