for d in 8 16 32 64; do
   qsub experiment_multiple_reference/synthetic_fpr/qsub_fpr_ltm_mref.sh classification_model/model_$d.h5 $d $1
   echo experiment_multiple_reference/synthetic_fpr/qsub_fpr_ltm_mref.sh classification_model/model_$d.h5 $d $1
done