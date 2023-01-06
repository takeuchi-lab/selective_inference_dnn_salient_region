import numpy as np
import tensorflow as tf
import time

from .layers import *

class FCN:
    def __init__(self, model):
        self.input = model.input_names[0]
        self.output = model.output_names[0]
        self.layers = []
        self.layers_dict = {}
        for layer in model.layers:
            type = layer.__class__.__name__
            name = layer.name
            if type == "Conv2D":
                self.layers.append(Conv(layer))
                self.layers_dict[name] = Conv(layer)
            elif type == "MaxPooling2D":
                self.layers.append(MaxPool(layer))
                self.layers_dict[name] = MaxPool(layer)
            elif type == "UpSampling2D":
                self.layers.append(UpSampling(layer))
                self.layers_dict[name] = UpSampling(layer)
            elif type == "Dense":
                self.layers.append(Dense(layer))
                self.layers_dict[name] = Dense(layer)
            elif type == "BatchNormalization":
                self.layers.append(BatchNorm(layer))
                self.layers_dict[name] = BatchNorm(layer)
            elif type == "Dropout":
                self.layers.append(Dropout(layer))
                self.layers_dict[name] = MaxPool(layer)
            elif type == "InputLayer":
                self.layers.append(Input(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Concatenate":
                self.layers.append(Concatenate(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Add":
                self.layers.append(Add(layer))
                self.layers_dict[name] = Add(layer)
            elif type == "TFOpLambda":
                self.layers.append(sigmoid(layer))
            elif type == "Conv2DTranspose":
                self.layers.append(ConvTranspose(layer))
                self.layers_dict[name] = ConvTranspose(layer)
            elif type == "MaxPoolingWithArgmax2D":
                self.layers.append(MaxPoolingWithArgmax2D(layer))
                self.layers_dict[name] = MaxPoolingWithArgmax2D(layer)
            elif type == "MaxUnpooling2D":
                self.layers.append(MaxUnpooling2D(layer))
                self.layers_dict[name] =MaxUnpooling2D(layer)
            elif type == "CAM":
                self.layers.append(CAM(layer))
                self.layers_dict[name]= CAM(layer)
            elif type == "GlobalAveragePooling2D":
                self.layers.append(GlobalAveragePooling2D(layer))
                self.layers_dict[name]= GlobalAveragePooling2D(layer)
            else:
                assert False, "Unsupported layer type: {}".format(type)

        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

        def get_layer_summary_with_connections(layer):

            info = {}
            connections = []
            output_index = []
            for node in layer._inbound_nodes:
                if relevant_nodes and node not in relevant_nodes:
                    continue
                for ( inbound_layer, node_index, tensor_index, _,) in node.iterate_inbound(): 
                    connections.append(inbound_layer.name)
                    output_index.append(tensor_index)

            name = layer.name
            info["type"] = layer.__class__.__name__
            info["parents"] = connections
            info["output_index"] = output_index
            return info

        self.connections = {}
        layers = model.layers
        for layer in layers:
            info = get_layer_summary_with_connections(layer)
            self.connections[layer.name] = info

    @tf.function
    def forward(self, input):

        output_dict = {}
        for layer in self.layers:
            connections = self.connections[layer.name]["parents"]
            index = self.connections[layer.name]["output_index"]

            if len(connections) == 0:
                input_tensors = input
            elif len(connections) == 1:
                input_tensors = output_dict[connections[0]]
                if type(input_tensors) == type([]):
                    if len(index)==1:
                        input_tensors = input_tensors[index[0]]
                    else :
                        input_tensors = [input_tensors[index[i]] for i in index]
            else:
                # TODO 単アウトプットと複数アウトプットが混ざっているからどう対応するべきか
                input_tensors = [output_dict[i][j] for i,j in zip(connections,index)]


            output_dict[layer.name] = layer.forward(input_tensors)

            if type(output_dict[layer.name]) != type([]):
                output_dict[layer.name] = [output_dict[layer.name]]

        output = output_dict[self.output][0]

        return output

    @tf.function
    def forward_si(self, input_si):

        output_si_dict = {}
        for layer in self.layers:
            connections = self.connections[layer.name]["parents"]
            index = self.connections[layer.name]["output_index"]

            if len(connections) == 0:
                x, bias, a, b, l, u = input_si
            elif len(connections) == 1:
                # TODO 単アウトプットと複数アウトプットが混ざっているからどう対応するべきか
                x, bias, a, b, l, u = output_si_dict[connections[0]]
                x = x[0]
            else:
                if len(index)>1:
                    # TODO 単アウトプットと複数アウトプットが混ざっているからどう対応するべきか
                    x = [output_si_dict[i][0][j] for i,j in zip(connections,index)]
                else :
                    x = [output_si_dict[i][0] for i in connections]
                bias = [output_si_dict[i][1] for i in connections]
                a = [output_si_dict[i][2] for i in connections]
                b = [output_si_dict[i][3] for i in connections]
                l_list = [output_si_dict[i][4] for i in connections]
                u_list = [output_si_dict[i][5] for i in connections]
                l = tf.reduce_max(l_list)
                u = tf.reduce_min(u_list)

            x,bias,a,b,l,u = layer.forward_si(x, bias, a, b, l, u)

            if type(x) != type([]):
                x = [x]

            output_si_dict[layer.name] = x,bias,a,b,l,u

        output_x, output_bias, output_a, output_b, l, u = output_si_dict[self.output]

        return l, u, output_x[0]
