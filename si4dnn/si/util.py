import numpy as np
from scipy import stats
from mpmath import mp
import matplotlib.pyplot as plt


from si4dnn.problem.selection import NoObjectsError

class CantCalcPvalueError(Exception):
    pass

def calc_p_value_two_sided(z,intervals,std):
    numerator = 0
    denominator = 0

    normalized_z = z / std
    normalized_intervals = [interval/std for interval in intervals]

    for interval in normalized_intervals:
        l = interval[0]
        u = interval[1]

        denominator = denominator + mp.ncdf(u)-mp.ncdf(l)

        # -abs() part
        if u < 0:
            if l <= -abs(normalized_z)  and -abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(-abs(normalized_z)) - mp.ncdf(l)
            elif u <= -abs(normalized_z):
                numerator = numerator + mp.ncdf(u)-mp.ncdf(l)
        elif l<= 0 and 0 <= u:
            if l <= -abs(normalized_z):
                numerator = numerator + mp.ncdf(-abs(normalized_z)) - mp.ncdf(l)
            if 0 <= abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(u) - mp.ncdf(abs(normalized_z))
        elif  0 < l:
            if abs(normalized_z) <= l:
                numerator = numerator + mp.ncdf(u)-mp.ncdf(l)
            elif l <= abs(normalized_z) and abs(normalized_z) <= u:
                numerator = numerator + mp.ncdf(u)-mp.ncdf(abs(normalized_z))

    if denominator == 0:
        raise CantCalcPvalueError(normalized_z,normalized_intervals)
        
    return numerator/denominator

def _breakpoint_to_interval(breakpoint_list:np.ndarray)->list:
    """the function which convert the breakpoint_list that is [L1,U1,L2,U2,...] to the list which is [[L1,U1],[L2,U2]]

        Returns
            intervals 
    """
    n = breakpoint_list.shape[0] // 2
    intervals = []
    for i in range(n):
        lower = breakpoint_list[2*i]
        upper = breakpoint_list[2*i+1]
        intervals.append([lower.numpy(),upper.numpy()])

    return intervals

def permutation_test_abs(X,X_ref,output,model_with_cam,thr=0,multi_ref=False,test=False):
    rng = np.random.default_rng()

    shape = X.shape
    X_reshaped = np.reshape(X,-1)

    if multi_ref:
        X_ref_reshaped = np.reshape(np.sum(X_ref,axis=0),-1)
    else :
        X_ref_reshaped = np.reshape(X_ref,-1)

    model = np.reshape(output,-1) >= thr
    X_diff = X_reshaped-X_ref_reshaped
    X_abs = np.where(X_diff>=0,1,-1)
    eta = np.where(model,X_abs,0)/np.sum(model)

    if np.sum(model) == 0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)
    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)

    z_obs = eta_new @ X_new

    num_iter = 1000
    if test:
        num_iter = 5

    permute_test_statistics = np.full(num_iter,np.nan,dtype="float64")

    index = list(range(0,X_reshaped.shape[0]))

    for i in range(num_iter):
        try : 
            # permute image
            shuffled_index = rng.permuted(index)
            X_shuffled_vec = X_reshaped[shuffled_index]
            X_ref_shuffled_vec = X_ref_reshaped[shuffled_index]
            X_shuffled = np.reshape(X_shuffled_vec,shape)

            # obtain cam output and get model and compute statistics
            output  = model_with_cam.predict(X_shuffled,verbose=0)[0]
            model = np.reshape(output,-1) >= thr
            num_selected_pixels = np.sum(model)

            if num_selected_pixels == 0:
                raise NoObjectsError

            X_diff = X_shuffled_vec-X_ref_shuffled_vec
            X_abs = np.where(X_diff>=0,1,-1)
            eta = np.where(model,X_abs,0)/np.sum(model)
            eta_new = np.concatenate([eta,-eta],axis=0)
            X_new = np.concatenate([X_shuffled_vec,X_ref_shuffled_vec],axis=0)
            z = eta_new @ X_new

            # store shuffled test statistics
            permute_test_statistics[i] = z

        except NoObjectsError:
            pass

    permute_test_statistics = permute_test_statistics[~np.isnan(permute_test_statistics)]

    nominater1 = np.sum(permute_test_statistics<=-np.abs(z_obs))
    nominater2 = np.sum(permute_test_statistics>=np.abs(z_obs))
    
    print(z_obs)
    
    p_value = (nominater1+nominater2)/permute_test_statistics.shape[0]

    return p_value

def permutation_test_mean(X,X_ref,output,model_with_cam,thr=0,multi_ref=False,test=False):
    rng = np.random.default_rng()

    shape = X.shape
    X_reshaped = np.reshape(X,-1)

    if multi_ref:
        X_ref_reshaped = np.reshape(np.sum(X_ref,axis=0),-1)
        print(X_ref_reshaped.shape)
    else :
        X_ref_reshaped = np.reshape(X_ref,-1)

    model = np.reshape(output,-1) >= thr

    num_selected_pixels = np.sum(model)
    eta = model/num_selected_pixels

    if num_selected_pixels == 0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)
    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    z_obs = eta_new @ X_new

    num_iter = 1000
    if test:
        num_iter = 5
    permute_test_statistics = np.full(num_iter,np.nan,dtype="float64")

    index = list(range(0,X_reshaped.shape[0]))

    for i in range(num_iter):
        try : 

            # permute image
            index = list(range(0,X_reshaped.shape[0]))
            shuffled_index = rng.permuted(index)
            X_shuffled_vec = X_reshaped[shuffled_index]
            X_ref_shuffled_vec = X_ref_reshaped[shuffled_index]
            X_shuffled = np.reshape(X_shuffled_vec,shape)

            # obtain cam output and get model and compute statistics
            output  = model_with_cam.predict(X_shuffled,verbose=0)[0]
            eta = np.where(np.reshape(output,-1)>=thr,1,0)
            num_selected_pixels = np.sum(eta)
            if num_selected_pixels == 0:
                raise NoObjectsError
            eta = eta / np.sum(eta)
            eta_new = np.concatenate([eta,-eta],axis=0)
            X_new = np.concatenate([X_shuffled_vec,X_ref_shuffled_vec])
            z = eta_new @ X_new

            # store shuffled test statistics
            permute_test_statistics[i] = z

        except NoObjectsError:
            # print("no_object")
            pass
    
    permute_test_statistics = permute_test_statistics[~np.isnan(permute_test_statistics)]


    nominater1 = np.sum(permute_test_statistics<=-np.abs(z_obs))
    nominater2 = np.sum(permute_test_statistics>=np.abs(z_obs))
    
    p_value = (nominater1+nominater2)/permute_test_statistics.shape[0]

    return p_value

def permutation_test_abs_2(X,X_ref,output,model_with_cam,thr=0,multi_ref=False,test=False):
    rng = np.random.default_rng()

    shape = X.shape
    X_reshaped = np.reshape(X,-1)

    if multi_ref:
        X_ref_reshaped = np.reshape(np.sum(X_ref,axis=0),-1)
        X_total = np.reshape(np.concatenate([X,X_ref],axis=0),-1)
    else :
        X_ref_reshaped = np.reshape(X_ref,-1)

    model = np.reshape(output,-1) >= thr
    X_diff = X_reshaped-X_ref_reshaped
    X_abs = np.where(X_diff>=0,1,-1)
    eta = np.where(model,X_abs,0)/np.sum(model)

    if np.sum(eta) == 0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)
    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)

    z_obs = eta_new @ X_new

    num_iter = 1000
    if test:
        num_iter = 5
    permute_test_statistics = np.full(num_iter,np.nan,dtype="float64")

    for i in range(num_iter):
        try : 
            # permute image
            if multi_ref:
                X_total_shuffled = rng.permuted(X_total)
                X_shuffled_vec = X_total_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)
                X_ref_shuffled_vec = np.mean(np.reshape(X_total_shuffled[X_reshaped.shape[0]:],[-1,X_reshaped.shape[0]]),axis=0)

            else :
                X_new_shuffled = rng.permuted(X_new)
                X_shuffled_vec = X_new_shuffled[X_reshaped.shape[0]:]
                X_ref_shuffled_vec = X_new_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)

            # obtain cam output and get model and compute statistics
            output  = model_with_cam.predict(X_shuffled,verbose=0)[0]
            model = np.reshape(output,-1) >= thr
            num_selected_pixels = np.sum(model)
            if num_selected_pixels == 0:
                raise NoObjectsError
            X_diff = X_shuffled_vec-X_ref_shuffled_vec
            X_abs = np.where(X_diff>=0,1,-1)
            eta = np.where(model,X_abs,0)/np.sum(model)
            eta_new = np.concatenate([eta,-eta],axis=0)
            X_new = np.concatenate([X_shuffled_vec,X_ref_shuffled_vec],axis=0)
            z = eta_new @ X_new

            # store shuffled test statistics
            permute_test_statistics[i] = z

        except NoObjectsError:
            pass

    permute_test_statistics = permute_test_statistics[~np.isnan(permute_test_statistics)]


    nominater1 = np.sum(permute_test_statistics<=-np.abs(z_obs))
    nominater2 = np.sum(permute_test_statistics>=np.abs(z_obs))
    
    p_value = (nominater1+nominater2)/permute_test_statistics.shape[0]

    return p_value

def permutation_test_mean_2(X,X_ref,output,model_with_cam,thr=0,multi_ref=False,test=False):
    rng = np.random.default_rng()

    shape = X.shape
    X_reshaped = np.reshape(X,-1)

    if multi_ref:
        X_ref_reshaped = np.reshape(np.sum(X_ref,axis=0),-1)
        X_total = np.reshape(np.concatenate([X,X_ref],axis=0),-1)
    else :
        X_ref_reshaped = np.reshape(X_ref,-1)

    model = np.reshape(output,-1) >= thr

    num_selected_pixels = np.sum(model)
    eta = model/num_selected_pixels

    if num_selected_pixels == 0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)
    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    z_obs = eta_new @ X_new

    num_iter = 1000
    if test:
        num_iter = 5
    permute_test_statistics = np.full(num_iter,np.nan,dtype="float64")

    for i in range(num_iter):
        try : 

            if multi_ref:
                X_total_shuffled = rng.permuted(X_total)
                X_shuffled_vec = X_total_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)
                X_ref_shuffled_vec = np.mean(np.reshape(X_total_shuffled[X_reshaped.shape[0]:],[-1,X_reshaped.shape[0]]),axis=0)

            else :
                X_new_shuffled = rng.permuted(X_new)
                X_shuffled_vec = X_new_shuffled[X_reshaped.shape[0]:]
                X_ref_shuffled_vec = X_new_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)

            # obtain cam output and get model and compute statistics
            output  = model_with_cam.predict(X_shuffled,verbose=0)[0]
            eta = np.where(np.reshape(output,-1)>=thr,1,0)
            num_selected_pixels = np.sum(eta)
            if num_selected_pixels == 0:
                raise NoObjectsError
            eta = eta / np.sum(eta)
            eta_new = np.concatenate([eta,-eta],axis=0)
            X_new = np.concatenate([X_shuffled_vec,X_ref_shuffled_vec])
            z = eta_new @ X_new

            # store shuffled test statistics
            permute_test_statistics[i] = z

        except NoObjectsError:
            # print("no_object")
            pass
    
    permute_test_statistics = permute_test_statistics[~np.isnan(permute_test_statistics)]

    nominater1 = np.sum(permute_test_statistics<=-np.abs(z_obs))
    nominater2 = np.sum(permute_test_statistics>=np.abs(z_obs))
    
    p_value = (nominater1+nominater2)/permute_test_statistics.shape[0]

    return p_value

def naive_p_mean_bonf_rectangle(X,X_ref,cam_output,var,thr=0):
    _,m,_,_ = X.shape
    mp.dps = 5000

    num_rectangle = m*m*(m+1)*(m+1)/4

    _,x,y = np.where(cam_output>=thr)

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)

    model = np.zeros([m,m],dtype=bool)
    model[x_min:x_max+1,y_min:y_max+1] = 1

    num_pixels = (x_max+1-x_min)*(y_max+1-y_min)

    eta = np.reshape(model/num_pixels,-1)

    if num_pixels==0:
        raise NoObjectsError
    
    eta_new = np.concatenate([eta,-eta],axis=0)
    std = np.sqrt(eta_new @ eta_new * var)
    X_new = np.concatenate([np.reshape(X,-1),np.reshape(X_ref,-1)],axis=0)

    z = eta_new @ X_new

    std = np.sqrt(eta_new @ eta_new * var)

    pi = mp.ncdf(z/std)
    p_value = min(pi,1-pi)*2

    return min(float(p_value*num_rectangle),1)

def naive_p_abs_bonf_rectangle(X,X_ref,cam_output,var,thr=0):
    mp.dps = 5000
    _,m,_,_ = X.shape
    num_rectangle =  np.sum([[(m-i+1)*(m-j+1)*(2**(i*j)) for i in range(1,m+1)] for j in range(1,m+1)])

    _,x,y = np.where(cam_output>=thr)
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)

    model = np.zeros([m,m],dtype=bool)
    model[x_min:x_max+1,y_min:y_max+1] = True

    num_pixels = int((x_max+1-x_min)*(y_max+1-y_min))

    eta = model

    if num_pixels==0:
        raise NoObjectsError

    X_reshaped = np.reshape(X,-1)
    X_ref_reshaped = np.reshape(X_ref,-1)

    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    X_diff = X_reshaped - X_ref_reshaped
    model = np.reshape(cam_output,-1) >= thr

    X_diff_abs = np.where(X_diff >=0,1,-1)
    eta = np.where(model,X_diff_abs,0)

    eta_new = np.concatenate([eta,-eta],axis=0)

    z = eta_new @ X_new
    std = np.sqrt(eta_new @ eta_new * var)


    pi = mp.ncdf(z/std)
    p_value = min(pi,1-pi)*2
    print("num_hypo",num_rectangle)
    print(p_value)
    print("corrected p",p_value*num_rectangle)

    return min(float(p_value*num_rectangle),1)

def naive_p_mean_bonf(X,ref,cam_output,var,thr=0,multi_ref=False,verbose=False):
    _,m,_,_  = X.shape

    if multi_ref:
        num_ref = ref.shape[0]
        X_ref = np.mean(ref,axis=0)
    else :
        X_ref = ref

    num_hypothesis =  2 ** (m*m)

    model = np.where(np.reshape(cam_output,-1)>=thr,1,0)
    num_selected_pixels = np.sum(model)

    eta = model/num_selected_pixels

    if num_selected_pixels==0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)

    if multi_ref:
        std = np.sqrt((eta@eta + (eta@eta/num_ref))*var)
    else :
        std = np.sqrt((eta_new @ eta_new) * var)

    X_new = np.concatenate([np.reshape(X,-1),np.reshape(X_ref,-1)],axis=0)
    z = eta_new @ X_new

    mp.dps = 5000
    pi = mp.ncdf(z/std)
    p_value = min(pi,1-pi)*2
    temp = p_value*num_hypothesis

    if verbose:
        print("test statistic:",z/std)
        print("num_hypothesis:",num_hypothesis)
        print("p_value",p_value)
        print("correction p_value",temp)

    return float(min(temp,1))

def naive_p_abs_bonf(X,ref,cam_output,var,thr=0,multi_ref=False,verbose=False):
    from scipy import special

    if multi_ref:
        num_ref = ref.shape[0]
        X_ref = np.mean(ref,axis=0)
    else :
        X_ref = ref


    n = np.reshape(X,-1).shape[0]

    num_hypothesis =  np.sum([special.comb(n,i,exact=True)*(2**i) for i in range(1,n+1)])

    X_reshaped = np.reshape(X,-1)
    X_ref_reshaped = np.reshape(X_ref,-1)

    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    X_diff = X_reshaped - X_ref_reshaped

    model = np.reshape(cam_output,-1) >= thr

    if np.sum(model) == 0:
        raise NoObjectsError

    X_diff_abs = np.where(X_diff >=0,1,-1)

    eta = np.where(model,X_diff_abs,0)

    eta_new = np.concatenate([eta,-eta],axis=0)
    z = eta_new @ X_new

    if multi_ref:
        std = np.sqrt((eta@eta + (eta@eta/num_ref))*var)
    else :
        std = np.sqrt((eta_new @ eta_new) * var)

    mp.dps = 5000
    pi = mp.ncdf(z/std)
    p_value = min(pi,1-pi)*2

    temp = p_value*num_hypothesis

    if verbose:
        print("test statistic:",z)
        print("num_hypothesis:",num_hypothesis)
        print("p_value",p_value)
        print("correction p_value",temp)

    p_value = float(min(temp,1))

    return p_value

def naive_p(X, model,construct_eta,var,thr=0):
    eta = construct_eta(model)[0].numpy()
    X_reshaped = np.reshape(X,-1)
    z = eta @ X_reshaped
    std = np.sqrt(eta @ eta)*var
    pi = stats.norm.cdf(z, loc=0, scale=std)
    p_value = min(pi, 1 - pi) * 2
    return p_value

def naive_p_mean(X,X_ref,cam_output,var,thr=0,multi_ref=False):
    if multi_ref:
        X_ref = np.mean(X_ref,axis=0)
        num_ref = X_ref.shape[0]

    model = np.where(np.reshape(cam_output,-1)>=thr,1,0)
    num_selected_pixels = np.sum(model)
    eta = model/num_selected_pixels
    if num_selected_pixels==0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)

    if multi_ref:
        std = np.sqrt((eta @ eta + (eta @ eta) /num_ref)*var)
    else :
        std = np.sqrt(eta @ eta * var)

    X_new = np.concatenate([np.reshape(X,-1),np.reshape(X_ref,-1)],axis=0)
    z = eta_new @ X_new

    pi = stats.norm.cdf(z, loc=0, scale=std)
    p_value = min(pi, 1 - pi) * 2

    return p_value

def naive_p_abs(X,X_ref,cam_output,var,thr=0,multi_ref=False):
    if multi_ref:
        X_ref = np.mean(X_ref,axis=0)
        num_ref = X_ref.shape[0]

    X_reshaped = np.reshape(X,-1)
    X_ref_reshaped = np.reshape(X_ref,-1)

    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    X_diff = X_reshaped - X_ref_reshaped
    model = np.reshape(cam_output,-1) >= thr
    X_diff_abs = np.where(X_diff >=0,1,-1)

    eta = np.where(model,X_diff_abs,0)

    eta_new = np.concatenate([eta,-eta],axis=0)
    z = eta_new @ X_new

    if np.sum(model)==0:
        raise NoObjectsError

    if multi_ref:
        std = np.sqrt((eta @ eta + (eta @ eta) /num_ref)*var)
    else :
        std = np.sqrt(eta @ eta * var)

    pi = stats.norm.cdf(z, loc=0, scale=std)
    p_value = min(pi,1-pi)*2

    return p_value

def permutation_test_mean_3(X,X_ref,output,model_with_cam,thr=0,multi_ref=False,test=False):
    rng = np.random.default_rng()

    shape = X.shape
    X_reshaped = np.reshape(X,-1)

    if multi_ref:
        X_ref_reshaped = np.reshape(np.sum(X_ref,axis=0),-1)
        X_total = np.reshape(np.concatenate([X,X_ref],axis=0),-1)
    else :
        X_ref_reshaped = np.reshape(X_ref,-1)

    model = np.reshape(output,-1) >= thr

    num_selected_pixels = np.sum(model)
    eta = model/num_selected_pixels

    if num_selected_pixels == 0:
        raise NoObjectsError

    eta_new = np.concatenate([eta,-eta],axis=0)
    X_new = np.concatenate([X_reshaped,X_ref_reshaped],axis=0)
    z_obs = eta_new @ X_new

    num_iter = 1000
    if test:
        num_iter = 5
    permute_test_statistics = np.full(num_iter,np.nan,dtype="float64")

    i = 0
    while i < num_iter:
        try : 

            if multi_ref:
                X_total_shuffled = rng.permuted(X_total)
                X_shuffled_vec = X_total_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)
                X_ref_shuffled_vec = np.mean(np.reshape(X_total_shuffled[X_reshaped.shape[0]:],[-1,X_reshaped.shape[0]]),axis=0)

            else :
                X_new_shuffled = rng.permuted(X_new)
                X_shuffled_vec = X_new_shuffled[X_reshaped.shape[0]:]
                X_ref_shuffled_vec = X_new_shuffled[:X_reshaped.shape[0]]
                X_shuffled = np.reshape(X_shuffled_vec,shape)

            # obtain cam output and get model and compute statistics
            output  = model_with_cam.predict(X_shuffled,verbose=0)[0]
            eta = np.where(np.reshape(output,-1)>=thr,1,0)
            num_selected_pixels = np.sum(eta)
            if num_selected_pixels == 0:
                raise NoObjectsError
            eta = eta / np.sum(eta)
            eta_new = np.concatenate([eta,-eta],axis=0)
            X_new = np.concatenate([X_shuffled_vec,X_ref_shuffled_vec])
            z = eta_new @ X_new

            # store shuffled test statistics
            permute_test_statistics[i] = z
            i+=1

        except NoObjectsError:
            # print("no_object")
            pass
    
    permute_test_statistics = permute_test_statistics[~np.isnan(permute_test_statistics)]

    nominater1 = np.sum(permute_test_statistics<=-np.abs(z_obs))
    nominater2 = np.sum(permute_test_statistics>=np.abs(z_obs))
    
    p_value = (nominater1+nominater2)/permute_test_statistics.shape[0]

    return p_value