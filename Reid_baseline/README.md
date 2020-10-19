# NAIC_ReID_Competition


## Authors

- [Bochuan Wu](https://github.com/bochuanwu/)



### Project File Structure

```
+-- NAIC_reid_2020
|   +-- reid_baseline(source code)
|   +-- model(dir to save the output)
|   +-- NAIC_Person_Reid(put dataset here)
|		+--image_B
```

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/bochuanwu/Naic_reid_2020.git`

3. Install dependencies:
   - [pytorch>=1.1.0](https://pytorch.org/)
   - python>=3.5
   - torchvision
   - [yacs](https://github.com/rbgirshick/yacs)
   - cv2
   
   I use cuda 10.1/python 3.6.10/torch 1.5.0/torchvision 0.6.0 for training and testing.
   
5.  [ResNet-ibn](https://github.com/XingangPan/IBN-Net) is applied as the backbone. Download ImageNet pretrained model  [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S)  or using the link in the source code 'pth.txt'

## RUN

1. If you want to test the Person ReID Compitition of NAIC . Put the test dataset to correct position and Use the following commands:

   ```bash
   bash run.sh
   ```

2. If  you want to use our baseline for training. 

   ```bash
   python train.py --config_file [CHOOSE WHICH config TO RUN]
   # E.g
   #python train.py --config_file configs/naic_round2_model_a.yml
   ```

