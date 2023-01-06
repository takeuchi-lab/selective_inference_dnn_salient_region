for signal in 1 2 3 4; do
   python experiment_one_reference/synthetic_tpr/tpr_ltm_oref.py classification_model/model_16.h5 $signal 0 $1
done