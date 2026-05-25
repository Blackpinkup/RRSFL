cd federated

# Breast cancer histology classification
python fed_train.py --log --data camelyon17

# Prostate MRI segmentation
python fed_train.py --log --data prostate --batch 16

# Diabetic retinopathy grading
# python fed_train.py --log --data dgdr --batch 16
