for signal in 1 2 3 4; do
   qsub experiment_multiple_reference/synthetic_tpr/qsub_tpr_lta_mref.py classification_model/model_16.h5 $signal 0 $1
done