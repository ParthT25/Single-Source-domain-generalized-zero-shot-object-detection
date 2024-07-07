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
## Steps

1. Run `extraction_setup.py` in Mode 1

    ### Modes

    #### Mode 1: Setup for Extraction of ROI Feature Maps
    - Use this mode to set up the environment for extracting Region of Interest (ROI) feature maps.

    #### Mode 2: Setup for Testing the Model and Disable ROI Feature Extraction
    - Use this mode to prepare the environment for testing the model and to disable ROI feature extraction.

    > **Note**: Please disable extraction (Run Mode 2) before testing the model.

2. Run the code in evaluation mode to extract ROI feature maps:
   ```sh
   python train.py --eval-only --config-file configs/diverse_weather.yaml MODEL.WEIGHTS /u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/all_outs/diverse_weather/model_best.pth

Run the WC-DCGAN and generate the unseen class ROIs and fine tune clip attention pooling layer:
  ```sh
  cd WC-DCGAN
  python train.py
  python generate.py
  python retrain_clip_attn.py
  ```

## Steps for zero shot learning

1. Run `extraction_setup.py` in Mode 2 to disable feature extraction.
2. Run `zero_setup_file.py` and write the names of new classes separated by a comma.
    
   
Run the code in evaluation mode using the finetuned model
   ```sh
   python train.py --eval-only --config-file configs/diverse_weather.yaml MODEL.WEIGHTS /u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/Models/updated_clipattn.pth
   ```
