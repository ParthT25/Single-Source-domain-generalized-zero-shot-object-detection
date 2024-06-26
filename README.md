# Single-Source-domain-generalized-object-detection

### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and requires python >= 3.6

Install the required packages
```
pip install -r requirements.txt
```

### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy
            ...
        /comic
        /watercolor
        /VOC2007
        /VOC2012 

```
Download [Diverse Weather](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)
We train our models on a single A100 GPU.
```
    python train.py --config-file configs/diverse_weather.yaml
```

After Training follow these steps
```
    here (https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/configs/diverse_weather.yaml#L31) edit the value from 6 to 7
    here https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/data/datasets/diverse_weather.py#L15 add bus class in the array
    Remove commented out code https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/modeling/meta_arch.py#L241-L271 and https://github.com/papz2000/Single-Source-domain-generalized-object-detection/blob/5e0a712684367d0523293226f81cb159b29935bd/modeling/meta_arch.py#L241-L271
```

