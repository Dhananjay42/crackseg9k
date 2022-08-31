CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 8 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0
