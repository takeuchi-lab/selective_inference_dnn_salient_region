for signal in 1 2 3 4; do
   qsub experiment_one_reference/synthetic_tpr/qsub_tpr_ltm_oref.sh classification_model/model_16.h5 $signal 0 $1
   echo experiment_one_reference/synthetic_tpr/qsub_tpr_ltm_oref.sh classification_model/model_16.h5 $signal 0 $1
done