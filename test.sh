cd federated

# Breast cancer histology classification
python fed_train.py --test --test_path ../checkpoint/camelyon17/seed0/RRSFL_exp/RRSFL --data camelyon17

# Prostate MRI segmentation
python fed_train.py --test --test_path ../checkpoint/prostate/seed0/RRSFL_exp/RRSFL --data prostate

# Diabetic retinopathy grading
# python fed_train.py --test --test_path ../checkpoint/dgdr/seed0/RRSFL_exp/RRSFL --data dgdr
