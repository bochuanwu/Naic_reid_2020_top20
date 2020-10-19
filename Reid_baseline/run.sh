echo NAIC Online Score Reproduction of DMT

echo Train

#python divided_dataset.py --data_dir_query ../NAIC_Person_Reid/reid_baseline/dataset/image_A/query --data_dir_gallery ../NAIC_Person_Reid/reid_baseline/dataset/image_A/bounding_box_test --save_dir ../NAIC_Person_Reid/reid_baseline/dataset/test/
#https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth
#nohup python train.py --config_file configs/naic_efn.yml > logs.txt 2>&1 & 
#nohup python train.py --config_file configs/naic_round2_model_a.yml > logs2.txt 2>&1 & 
nohup python train.py --config_file configs/naic_resnext_a.yml > logs.txt 2>&1 & 

#nohup python train.py --config_file configs/naic_round2_model_b.yml > logs2.txt 2>&1 & 

#nohup python train.py --config_file configs/naic_round2_model_se.yml > logs3.txt 2>&1 & 

#nohup python train_UDA.py --config_file configs/naic_round2_model_b.yml --config_file_test configs/naic_round2_model_a.yml --data_dir_query ../NAIC_Person_Reid/reid_baseline/dataset/image_A/query --data_dir_gallery ../NAIC_Person_Reid/reid_baseline/dataset/image_A/bounding_box_test > logs.txt 2>&1 & 

#nohup python train_UDA.py --config_file configs/naic_round2_model_se.yml --config_file_test configs/naic_round2_model_b.yml --data_dir_query ../NAIC_Person_Reid/reid_baseline/dataset/image_A/query --data_dir_gallery ../NAIC_Person_Reid/reid_baseline/dataset/image_A/bounding_box_test > logs2.txt 2>&1 & 

#echo Test
# here we will get Distmat Matrix after test.
#python test.py --config_file configs/naic_efn.yml 
#python test.py --config_file configs/naic_resnext_a.yml 

#python test.py --config_file configs/naic_round2_model_a.yml

#python test.py --config_file configs/naic_round2_model_b.yml

#python test.py --config_file configs/naic_efn.yml

#echo Ensemble

#python ensemble_dist.py
