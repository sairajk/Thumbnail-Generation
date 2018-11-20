import numpy as np
import tensorflow as tf


params = {'batch_size': 1,
		'lr': 0.001,
		'cell_type': 'LSTM',
        'n_b_boxes': 3
		}


# FUNCTION TO GENERATE RECURRENT CELLS
def Recurrent_Cell(output_size, cell_type='Basic'):
    if cell_type == 'Basic':
        return tf.contrib.rnn.BasicRNNCell(output_size)
    if cell_type == 'LSTM':
        return tf.contrib.rnn.BasicLSTMCell(output_size)
    if cell_type == 'GRU':
        return tf.contrib.rnn.GRUCell(output_size)
    if cell_type == 'LayerNorm':
        return tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)


# FUNCTION TO PERFORM GLOBAL CONTEXT AGGREGATION - Recurrent Part
# 'inputs' is a 4D tensor from a convolution operation
# The output returned is also a 4D tensor
def GCA_RNN(inputs, direction='LR', result_op='Sum'):
    shape_list = inputs.get_shape().as_list()
    height, width, channels = shape_list[1], shape_list[2], shape_list[3]
    
	# To scan Left-Right
    if direction == 'LR':
        time_step_size = width
        inputs_reshaped = tf.reshape(inputs, (params['batch_size']*height, width, channels))
        with tf.variable_scope('LR') as scope:
            cell_fw = Recurrent_Cell(output_size=channels, cell_type=params['cell_type'])
            cell_bw = Recurrent_Cell(output_size=channels, cell_type=params['cell_type'])
            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                                                                                 inputs_reshaped, dtype=tf.float32)
            output_fw = tf.reshape(output_fw, [-1, height, width, channels])
            output_bw = tf.reshape(output_bw, [-1, height, width, channels])
    
	# To scan Up-Down
    if direction == 'UD':
        time_step_size = height
        input_transposed = tf.transpose(inputs, perm=[0, 2, 1, 3])
        inputs_reshaped = tf.reshape(input_transposed, (params['batch_size']*width, height, channels))
        with tf.variable_scope('UD') as scope:
            cell_fw = Recurrent_Cell(output_size=channels, cell_type=params['cell_type'])
            cell_bw = Recurrent_Cell(output_size=channels, cell_type=params['cell_type'])
            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                                                                                 inputs_reshaped, dtype=tf.float32)
            output_fw = tf.reshape(output_fw, [-1, width, height, channels])
            output_bw = tf.reshape(output_bw, [-1, width, height, channels])
            
            output_fw = tf.transpose(output_fw, perm=[0, 2, 1, 3])
            output_bw = tf.transpose(output_bw, perm=[0, 2, 1, 3])
    
    if result_op == 'Sum':
        return output_fw + output_bw
    
    if result_op == 'Concatenate':
        return tf.concat([output_fw, output_bw], 3)
		

# FUNCTION TO PERFORM GLOBAL CONTEXT AGGREGATION - Attention Part
def GCA(inputs):
    shape_list = inputs.get_shape().as_list()
    height, width, channels = shape_list[1], shape_list[2], shape_list[3]
    
    # Scan input Left-Right
    output_lr = GCA_RNN(inputs, direction='LR', result_op='Sum')
    # Scan input Up-Down
    output_ud = GCA_RNN(output_lr, direction='UD', result_op='Sum')
    
    conv_output = conv2d(output_ud, height*width, k_h=3, k_w=3, d_h=1, d_w=1)
    # Apply softmax to Normalize the last dimension
    conv_softmax = tf.nn.softmax(conv_output)
    
    # Attend
    conv_softmax_2d = tf.reshape(conv_softmax, (-1, conv_softmax.get_shape().as_list()[-1]))
    inputs_2d = tf.reshape(inputs, (-1, channels))
    
    # print("inputs \t", inputs.get_shape().as_list())
	# print("inputs 2d \t", inputs_2d.get_shape().as_list())
    # print("conv_softmax \t", conv_softmax.get_shape().as_list())
    # print("conv_softmax 2d \t", conv_softmax_2d.get_shape().as_list())
    
    # F_attn. if the final attended input
    f_attn = tf.matmul(conv_softmax_2d, inputs_2d)
    f_attn = tf.reshape(f_attn, shape_list)
    
    return f_attn


# FUNCTION TO DYNAMICALLY GENERATE CONVOLUTION KERNEL AND BIAS FROM GIVEN INPUT ASPECT-RATIO AND SHAPE
# aspect-ratio is a tensor of shape [batch_size, 1] and shape is a list of 4 elements
def Gen_Adap_Weights(aspect_ratio, shape, name="adap_conv_layer"): 
    with tf.variable_scope(name):
        # Number of units in the final layer (for convolution)
        n_units = shape[0] * shape[1] * shape[2] * shape[3] + shape[3]

        adap_weights = linear(aspect_ratio, 16, 'l_1')
        adap_weights = linear(adap_weights, 32, 'l_2')
        adap_weights = linear(adap_weights, n_units, 'l_3')
    
        bias = adap_weights[:, -shape[3]:]
        bias = tf.reshape(bias, [-1])
        
        weights = tf.reshape(adap_weights[:, :-shape[3]], shape)
        
        return weights, bias
	

# Modified Region Proposal Network with Adaptive Convolutions
def RPN(conv_features, aspect_ratio):
    # The first convolution of RPN module
    conv_op_1 = conv2d(conv_features, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_op_1")
    
    # predict objectness-score (3 boxes per position)
    w2, b2 = Gen_Adap_Weights(aspect_ratio, [1, 1, 512, params['n_b_boxes']], "ad_obj_scores")
    objectness_op = tf.nn.conv2d(conv_op_1, w2, strides=[1, 1, 1, 1], padding='SAME')
    objectness_op = tf.reshape(tf.nn.bias_add(objectness_op, b2), objectness_op.get_shape())
    
    # predict bounding box
    w3, b3 = Gen_Adap_Weights(aspect_ratio, [1, 1, 512, params['n_b_boxes']*4], "ad_bb_coord")
    b_box_op = tf.nn.conv2d(conv_op_1, w3, strides=[1, 1, 1, 1], padding='SAME')
    b_box_op = tf.reshape(tf.nn.bias_add(b_box_op, b3), b_box_op.get_shape())
    bb_shape = b_box_op.get_shape().as_list()
    b_box_op = tf.reshape(b_box_op, [bb_shape[0], bb_shape[1], bb_shape[2], params['n_b_boxes'], 4])
    
    return b_box_op, objectness_op
