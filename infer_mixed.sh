# CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --dataset 'housecat6d' --dataset_root '/data/robotarm/dataset/housecat6d' --split '/home/robotarm/object_depth_percetion/dataset/splits/housecat6d_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'HAMMER' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt'

# CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --dataset 'PhoCAL' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/PhoCAL_test.txt'

# CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --dataset 'TransCG' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/TransCG_d435_test.txt'

# CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --dataset 'XYZ-IBD' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --dataset 'YCB-V' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --dataset 'T-LESS' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt'

CUDA_VISIBLE_DEVICES=6 python infer_mixed_dataset.py --dataset 'GN-Trans' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/GN-Trans_test.txt'
