for d in 8 16 32 64; do
   qsub experiment_one_reference/synthetic_fpr/qsub_fpr_lta_oref.sh classification_model/model_$d.h5 $d $1
   echo experiment_one_reference/synthetic_fpr/qsub_fpr_lta_oref.sh classification_model/model_$d.h5 $d $1
done