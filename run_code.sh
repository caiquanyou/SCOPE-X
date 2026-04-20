#export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使程序仅识别7张卡
source /XYAIFS00/gibh_jkchen_7/HOME/miniconda3/bin/activate SCPOEX
cd /XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/SCOPE-X/HongKong/AI_code_archive/20260403_SCOPE-X_V3.5.0_parameter_test2
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 run_token1_test.py
  