import os
import sys
import tensorflow as tf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from tqdm import tqdm
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

import models
from si4dnn import cam_si
import data
from si4dnn.problem.selection import NoObjectsError
from si4dnn.si import util


def run(args):
    model_path, shape,method,seed= args
    np.random.seed(seed)

    # load tensorflow model and create cam
    model = tf.keras.models.load_model(model_path)
    layers = model.layers
    cam = models.CAM(layers[-1], shape)([layers[-3].output, layers[-1].output])
    model_with_cam = tf.keras.Model(inputs=model.input, outputs=cam)

    # make selective inference instance from model_with_cam
    cam_si_thr = cam_si.si4dnn_cam_si_thr(model_with_cam)

    flag = True

    test=False

    result_dic = {}

    while flag:
        try:
            # sampling test image from standard normal distribution
            X = data.generate_data_classification(shape, 1, 0, 0, c=True)

            # sampling 10 reference images from normal distribution
            ref = data.generate_data_classification(shape, 1, 0, 0, c=True)

            # compute saliency region
            output = model_with_cam.predict(X,verbose=0)[0]

            # do hypothesis test
            if method["parametric"]:
                p_value = cam_si_thr.inference(X, ref=ref)[0]
                result_dic["parametric"] = p_value
            
            if method["oc"]:
                p_value = cam_si_thr.inference(X, ref=ref,oc=True)[0]
                result_dic["oc"] = p_value
            
            if method["naive"]:
                p_value =  util.naive_p_mean(X,ref,output,1)
                result_dic["naive"] = p_value
            
            if method["bonf"]:
                p_value = util.naive_p_mean_bonf(X, ref, output,1)
                result_dic["bonf"] = p_value
            
            if method["permutation1"]:
                p_value = util.permutation_test_mean(X, ref, output, model_with_cam,test=test)
                result_dic["permutation1"] = p_value
            
            if method["permutation2"]:
                p_value =  util.permutation_test_mean_2(X, ref, output, model_with_cam,test=test)
                result_dic["permutation2"] = p_value
            
            flag=False

        except NoObjectsError:
            pass
        except util.CantCalcPvalueError:
            print("切断区間を計算できませんでした!")

    return result_dic


def do_experiment(num_iter, shape, model_path, method ,num_worker=40):

    # do experiments with  multiprocessing
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        params = map(lambda _: (model_path, shape,method,
                     np.random.randint(10000)), range(num_iter))
        results = list(tqdm(executor.map(run, params),total=num_iter))
    
    p_dic = {}

    for key in method:
        if method[key]:
            p_dic[key] = []
    
    for result in results:
        for method in result:
            p_dic[method].append(result[method])
    
    return p_dic


if __name__ == "__main__":
    # 初期化
    model_path = sys.argv[1]
    d = int(sys.argv[2])
    num_iter = int(sys.argv[3])
    is_tracking = False

    shape = (d, d, 1)
    n = d*d

    alpha = 0.05

    method = {
        "parametric":True,
        "oc":True,
        "naive":True,
        "permutation1":True,
        "permutation2":True,
        "bonf":True
    }


    # For reproducibillity
    np.random.seed(1234)

    # experiment tracking with mlflow
    if is_tracking:
        TRACKING_URI = os.environ["TRACKING_URI"]
        mlflow.set_tracking_uri(TRACKING_URI)

        ARTIFACT_LOCATION = os.environ["ARTIFACT_LOCATION"]

        EXPERIMENT_NAME = "final_fpr_ltm_oref"
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
        else:
            experiment_id = experiment.experiment_id

        mlflow.start_run(experiment_id=experiment_id)

        mlflow.log_params({
            "n": d*d,
            "num_iter": num_iter,
            "alpha":  alpha
        })

    # do experiment
    p_dic = do_experiment(num_iter,shape,model_path,method)

    # log expreiment result
    result_path = "./experiment_result/fpr_thr_ltm_oref"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    artifact_path = result_path +f"/num_iter_{num_iter}"

    # make directory if it doesn't exist
    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path)

    artifact_path += f"/n_{n}"

    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path)

    csv_path = artifact_path + "/fpr.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path,index_col=0)
    else :
        df = pd.DataFrame()

    for m in method:
        if method[m]:
            p_values = np.array(p_dic[m])
            fpr = np.mean(p_values<=alpha)

            df[m+"_fpr"] = [fpr]

            p_ks = stats.kstest(p_values, "uniform")[1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(p_values)
            ax.set_title(f"{m} P:{p_ks:.2f}")

            fig_path = artifact_path + f"/{m}_p_values.pdf"
            fig.savefig(fig_path)

            if is_tracking:
                mlflow.log_metrics({
                    f"{m}_fpr": fpr,
                    f"{m}_ks": p_ks
                })
    
    df.to_csv(csv_path)

    if is_tracking:
        mlflow.log_artifacts(artifact_path)