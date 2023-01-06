for d in 8 16 32 64; do
   python experiment_multiple_reference/synthetic_fpr/fpr_ltm_mref.py classification_model/model_$d.h5 $d $1
done