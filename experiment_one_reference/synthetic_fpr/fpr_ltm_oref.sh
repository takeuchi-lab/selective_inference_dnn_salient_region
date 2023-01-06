for d in 8 16 32 64; do
   python experiment_one_reference/synthetic_fpr/fpr_ltm_oref.py classification_model/model_$d.h5 $d $1
done