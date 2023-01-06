import tensorflow as tf
import numpy as np
from .model import fcn
from . import util
from .model.layers import truncated_interval

class NoIntervalError(Exception):
    pass

class SI4DNN:
    def __init__(self, model,construct_model,construct_eta, comparison_model,abs=False,**kwrgs):

        self.fcn = fcn.FCN(model)
        self.construct_model = construct_model
        self.construct_eta = construct_eta
        self.comparison_model = comparison_model
        self.abs = abs
    
    def construct_test_statistics(self,var):

        X_reshaped = tf.reshape(self.x, [-1])
        self.n = X_reshaped.shape[0]

        if self.ref is None:
            self.z = tf.tensordot(self.eta, X_reshaped, 1)
            self.var = tf.tensordot(self.eta, self.eta, 1)*(var)
            self.b = self.eta / self.var
            self.a = X_reshaped - tf.multiply(self.eta, tf.tensordot(X_reshaped, self.eta, 1)) / self.var
        else:
            if self.multi_ref:
                X_reshaped = tf.reshape(self.x,-1)
                ref_reshaped = tf.reshape(self.ref,-1)

                # 連結した値を求める
                X_new = tf.concat([X_reshaped,ref_reshaped],axis=0)
                self.X_new = X_new
                eta = self.eta
                self.z = tf.tensordot(eta,X_new,1)

                first = tf.ones(self.n,dtype=tf.float64)
                second = first * (1.0/self.num_ref)

                vec = tf.concat([first,second],axis=0)

                eta_num_ref  =  eta * vec

                self.var = tf.tensordot(eta,eta_num_ref,1)*var
                self.b =  eta_num_ref / self.var
                self.a = X_new - tf.multiply(eta_num_ref, tf.tensordot(eta, X_new, 1)) / self.var
            else :
                ref_reshaped = tf.reshape(self.ref,[-1]) 

                self.x_reshaped = X_reshaped
                self.x_ref_reshaped = ref_reshaped

                # 連結した値を求める
                X_new = tf.concat([X_reshaped,ref_reshaped],axis=0)
                self.X_new = X_new
                eta = self.eta
                self.z = tf.tensordot(eta,X_new,1)
                self.var = tf.tensordot(eta,eta,1)*var
                self.b =  eta / self.var
                self.a = X_new - tf.multiply(eta, tf.tensordot(eta, X_new, 1)) / self.var

    def inference(self,input,*,ref=None,var=1,parallelization=False,oc=False,eps=1e-7,verbose=False,multi_ref=False,**kargs):

        self.multi_ref = multi_ref
        self.ref = ref

        if self.multi_ref:
            self.num_ref = self.ref.shape[0]
            self.ref =np.mean(self.ref,axis=0,keepdims=True) 
        else : 
            self.ref = ref

        self.x = tf.constant(input, dtype=tf.float64)
        self.output = tf.cast(self.fcn.forward(self.x), dtype=tf.float64)
        self.model = self.output
        self.eta = self.construct_model(self.output,self.x,self.ref)
        self.construct_test_statistics(var)
        self.z_min = -tf.abs(self.z)-10*tf.sqrt(self.var)
        self.z_max = tf.abs(self.z)+10*tf.sqrt(self.var)
        self.eps = eps

        if verbose == True:
            print("test_statistics", self.z)
            print("探索範囲", self.z_min, self.z_max)

        if oc:
            intervals,output = self._compute_solution_path_oc()
            assert np.all(np.isclose(output[0].numpy()>=0.5,self.model.numpy()>=0.5,atol=1e-3,rtol=1e-3))
            number_tortal_intervals = 1
        else:
            breakpoints,number_tortal_intervals = self._compute_solution_path(
                self.z_min, self.z_max,verbose
            )
            intervals = util._breakpoint_to_interval(breakpoints)
        
        if verbose:
            tf.print("selected interval",intervals)
        
        if intervals is None:
            raise NoIntervalError(f"test statistic : {self.z}\n search interval [{self.z_min},{self.z_max}]]\n obtained intervals {intervals}")


        selective_p_value = float(util.calc_p_value_two_sided(
            self.z.numpy(), intervals, np.sqrt(self.var.numpy())
        ))

        return selective_p_value,self.z,np.sqrt(self.var),self.model,intervals,number_tortal_intervals

    def _compute_solution_path(self, z_min, z_max,verbose):
        count = 0
        detect_count = 0
        z = z_min

        intervals = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        i = 0

        while z < z_max:
            count += 1
            y = self.a + self.b * z

            n = self.n

            y_first = y[:n]
            y_second = y[n:]
            a_first = self.a[:n]
            a_second = self.a[n:]
            b_first = self.b[:n]
            b_second = self.b[n:]

            if self.abs:
                positive_index = (y_first-y_second)>=0
                tTa = tf.where(positive_index,-(a_first-a_second),(a_first-a_second))
                tTb = tf.where(positive_index,-(b_first-b_second),(b_first-b_second))
                bias = tf.zeros(positive_index.shape,dtype=tf.float64)
                l_abs,u_abs = truncated_interval(tTa,tTb,bias)

            B, H, W, C = self.x.shape
            y = tf.reshape(tf.constant(y_first, dtype=tf.float64),[B,H,W,C])
            bias = tf.zeros([B, H, W, C], dtype=tf.float64)
            a = tf.reshape(tf.constant(a_first, dtype=tf.float64), [B, H, W, C])
            b = tf.reshape(tf.constant(b_first, dtype=tf.float64), [B, H, W, C])
            
            input_si = (y, bias, a, b, z, z_max)
            l, u, output = self.fcn.forward_si(input_si)

            if self.abs:
                l = tf.maximum(l,l_abs)
                u = tf.minimum(u,u_abs)

            assert l<u

            if verbose == True:
                tf.print("探索区間",l,u)

            if self.comparison_model(self.output,output,self.x,y_first,self.ref,y_second):
                detect_count += 0
                intervals = intervals.write(i, l)
                i += 1
                intervals = intervals.write(i, u)
                i += 1
            else : 
                pass

            z = u + self.eps

        return intervals.stack(),count

    def _compute_solution_path_oc(self):

        n = self.n

        y_first = self.X_new[:n]
        y_second = self.X_new[n:]
        a_first = self.a[:n]
        a_second = self.a[n:]
        b_first = self.b[:n]
        b_second = self.b[n:]

        B, H, W, C = self.x.shape
        bias = tf.zeros([B, H, W, C], dtype=tf.float64)
        a = tf.reshape(tf.constant(a_first, dtype=tf.float64), [B, H, W, C])
        b = tf.reshape(tf.constant(b_first, dtype=tf.float64), [B, H, W, C])

        if self.abs:
            positive_index = (y_first-y_second)>=0
            tTa = tf.where(positive_index,-(a_first-a_second),(a_first-a_second))
            tTb = tf.where(positive_index,-(b_first-b_second),(b_first-b_second))
            s = tf.zeros(n,dtype=tf.float64)
            l_abs,u_abs = truncated_interval(tTa,tTb,s)

        input_si = (
            self.x,
            bias,
            a,
            b,
            tf.constant(-np.inf, dtype=tf.float64),
            tf.constant(np.inf, dtype=tf.float64),
        )

        l, u, output = self.fcn.forward_si(input_si)

        if self.abs:
            l = tf.maximum(l,l_abs)
            u = tf.minimum(u,u_abs)

        assert l<u

        return np.array([[l.numpy(), u.numpy()]]),output

    def inference_global_intersection_null(self,input,*,ref=None,var=1,parallelization=False,oc=False,eps=1e-7,verbose=False):
        shape = input.shape
        self.ref = ref
        self.x = tf.constant(input, dtype=tf.float64)
        self.output = tf.cast(self.fcn.forward(self.x), dtype=tf.float64)
        self.model,num_selected_pixels = self.construct_model(self.output)
        n = self.model.shape[0]
        index = tf.where(self.model)
        p_map = np.full(n,-1,dtype="float")

        for i in range(num_selected_pixels):
            self.eta = self.construct_eta(self.model,index[i].numpy()[0],X)
            self.construct_test_statistics(var)
            self.z_min = tf.sqrt(self.var) * -(tf.abs(self.z)+10)
            self.z_max = tf.sqrt(self.var) * (tf.abs(self.z)+10)
            self.eps = eps
            
            if verbose == True:
                print("test_statistics", self.z)
                print("探索範囲", self.z_min, self.z_max)

            B, H, W, C = self.x.shape
            x = tf.constant(self.x, dtype=tf.float64)
            bias = tf.zeros([B, H, W, C], dtype=tf.float64)
            a = tf.reshape(tf.constant(self.a, dtype=tf.float64), [B, H, W, C])
            b = tf.reshape(tf.constant(self.b, dtype=tf.float64), [B, H, W, C])

            if oc:
                intervals,output = self._compute_solution_path_oc(
                    self.fcn, self.model, x, bias, a, b, self.z_max, self.z_min
                )
                assert np.all(np.isclose(output[0].numpy()>=0.5,self.output.numpy()>=0.5,atol=1e-3,rtol=1e-3))
                number_tortal_intervals = 1
            else:
                breakpoints,number_tortal_intervals = self._compute_solution_path(
                    self.fcn, self.model, x, bias, a, b, self.z_min, self.z_max,verbose
                )
                intervals = util._breakpoint_to_interval(breakpoints)

            if verbose==True:
                print(intervals)

            selective_p_value = util.calc_p_value_two_sided(
                self.z.numpy(), intervals, np.sqrt(self.var.numpy())
            )

            p_map[index[i].numpy()] = float(selective_p_value)
        
        return np.reshape(p_map,shape)