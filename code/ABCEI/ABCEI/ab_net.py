import tensorflow as tf
import numpy as np

from ABCEI.util import *

class ab_net(object):


    def __init__(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, r_beta, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, z_norm, FLAGS, r_lambda, r_beta, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    # def _name_constructor(self, prefix, seq=0):
    #     return prefix+'_'+str(seq)


    def _build_graph(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, r_beta, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for causal effect inference.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_lambda = r_lambda
        self.r_beta = r_beta
        self.do_in = do_in
        self.do_out = do_out
        self.z_norm = z_norm

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_mi = dims[3]
        dim_d = dims[4]

        

        ''' Construct input/representation layers '''
        with tf.variable_scope('encoder') as scope:
            weights_in = []; biases_in = []

            if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
                dim_in = dim_input
            if FLAGS.n_out == 0:
                if FLAGS.split_output == False:
                    dim_out = dim_in+1
                else:
                    dim_out = dim_in

            if FLAGS.batch_norm:
                bn_biases = []
                bn_scales = []

            h_in = [x]

            for i in range(0, FLAGS.n_in):
                if i==0:
                    ''' If using variable selection, first layer is just rescaling'''
                    if FLAGS.varsel:
                        weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                    else:
                        weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], \
                            stddev=FLAGS.weight_init/np.sqrt(dim_input))))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], \
                        stddev=FLAGS.weight_init/np.sqrt(dim_in))))

                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel and i==0:
                    biases_in.append([])
                    h_in.append(tf.mul(h_in[i],weights_in[i]))
                else:
                    biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                    z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                    if FLAGS.batch_norm:
                        batch_mean, batch_var = tf.nn.moments(z, [0])

                        if FLAGS.normalization == 'bn_fixed':
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                        else:
                            bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                            bn_scales.append(tf.Variable(tf.ones([dim_in])))
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                    h_in.append(self.nonlin(z))
                    h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)

            h_rep = h_in[len(h_in)-1]

            if FLAGS.normalization == 'divide':
                h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
            else:
                h_rep_norm = 1.0*h_rep

        #estimate Global MI
        T_xy, T_x_y, weights_mi_x, weights_mi_y, weights_mi_pred = self._build_discriminator_graph_Mine(x, h_rep_norm, dim_input, dim_in, dim_mi, FLAGS) 

        #Global MI loss/representation error
        #compute the negative loss (maximise loss == minimise -loss)
        #neg_loss = -(tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y))))   #DV-based measure
        gmi_neg_loss = -(tf.reduce_mean(tf.log(2.)-tf.nn.softplus(-T_xy)) - \
        tf.reduce_mean(tf.nn.softplus(-T_x_y) + T_x_y - tf.log(2.)))    #JSD-based measure

        #adversarial balancing
        d0, d1, dp, weights_dis, weights_discore = self._build_adversarial_graph(h_rep_norm, t, dim_in, dim_d, do_out, FLAGS)
        
        #discriminator

        #with sigmoid
        # discriminator_loss = tf.reduce_mean(tf.nn.softplus(-d0)) + tf.reduce_mean(tf.nn.softplus(-d1) + d1) + dp
        
        #without sigmoid
        discriminator_loss = -tf.reduce_mean(d0) + tf.reduce_mean(d1) + r_beta*dp

        #encoder

        #with sigmoid
        # rep_loss = tf.reduce_mean(tf.nn.softplus(-d1))

        #without sigmoid
        rep_loss = -tf.reduce_mean(d1)

        ''' Construct ouput layers '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(res)
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])


        ''' Total error '''
        tot_error = risk

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.gmi_neg_loss = gmi_neg_loss
        self.discriminator_loss = discriminator_loss
        self.rep_loss = rep_loss
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_mi_x = weights_mi_x
        self.weights_mi_y = weights_mi_y
        self.weights_mi_pred = weights_mi_pred
        self.weights_dis = weights_dis
        self.weights_discore = weights_discore
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.dp = dp

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)
        with tf.variable_scope('pred') as scope:
            weights_out = []; biases_out = []

            for i in range(0, FLAGS.n_out):
                wo = self._create_variable_with_weight_decay(
                        tf.random_normal([dims[i], dims[i+1]],
                            stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                        'out_w_%d' % i, 1.0)
                weights_out.append(wo)

                biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
                z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

                h_out.append(self.nonlin(z))
                h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

            weights_pred = self._create_variable(tf.random_normal([dim_out,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
            bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

            if FLAGS.varsel or FLAGS.n_out == 0:
                self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
            else:
                self.wd_loss += tf.nn.l2_loss(weights_pred)

            ''' Construct linear classifier '''
            h_pred = h_out[-1]
            y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1,[rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred

    def _build_discriminator_graph_Mine(self, x, hrep, dim_input, dim_in, dim_mi, FLAGS):
        ''' Construct MI estimation layers '''
        with tf.variable_scope('gmi') as scope:
            input_size = tf.shape(x)[0]
            x_shuffle = tf.random_shuffle(x)
            x_conc = tf.concat([x, x_shuffle], axis=0)
            y_conc = tf.concat([hrep, hrep], axis=0)
            
    	    # forward
            weights_mi_x = self._create_variable(tf.random_normal([dim_input, dim_mi], 
                    stddev=FLAGS.weight_init/np.sqrt(dim_input)),'weights_mi_x')
            biases_mi_x = self._create_variable(tf.zeros([1,dim_mi]),'biases_mi_x')
            lin_x = tf.matmul(x_conc, weights_mi_x)+biases_mi_x

            weights_mi_y = self._create_variable(tf.random_normal([dim_in, dim_mi], 
                    stddev=FLAGS.weight_init/np.sqrt(dim_in)),'weights_mi_y')
            biases_mi_y = self._create_variable(tf.zeros([1,dim_mi]),'biases_mi_y')
            lin_y = tf.matmul(y_conc, weights_mi_y)+biases_mi_y

            #lin_conc = tf.nn.relu(lin_x + lin_y)
            lin_conc = self.nonlin(lin_x + lin_y)

            weights_mi_pred = self._create_variable(tf.random_normal([dim_mi, 1], 
                    stddev=FLAGS.weight_init/np.sqrt(dim_mi)),'gmi_p')
            biases_mi_pred = self._create_variable(tf.zeros([1,dim_mi]),'biases_mi_pred')
            gmi_output = tf.matmul(lin_conc,weights_mi_pred)+biases_mi_pred

            T_xy = gmi_output[:input_size]
            T_x_y = gmi_output[input_size:]

        return T_xy, T_x_y, weights_mi_x, weights_mi_y, weights_mi_pred

    def _build_discriminator_adversarial(self, hrep, dim_in, dim_d, do_out, FLAGS, reuse=False):
        ''' Construct adversarial discriminator layers '''
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h_dis = [hrep]
            
            weights_dis = []
            biases_dis = []
            for i in range(0, FLAGS.n_dc):

                if i==0:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_in,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_in))))
                else:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_d,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_d))))
                biases_dis.append(tf.Variable(tf.zeros([1,dim_d])))
                z = tf.matmul(h_dis[i], weights_dis[i])+biases_dis[i]
                h_dis.append(self.nonlin(z))
                h_dis[i+1] = tf.nn.dropout(h_dis[i+1], do_out)

            weights_discore = self._create_variable(tf.random_normal([dim_d,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_d)), 'dc_p')
            bias_dc = self._create_variable(tf.zeros([1]), 'dc_b_p')

            h_score = h_dis[-1]
            dis_score = tf.matmul(h_score, weights_discore) + bias_dc

        return dis_score, weights_dis, weights_discore

    def _build_adversarial_graph(self, rep, t, dim_in, dim_d, do_out, FLAGS):
        ''' Construct adversarial discriminator '''

        i0 = tf.to_int32(tf.where(t < 1)[:,0])
        i1 = tf.to_int32(tf.where(t > 0)[:,0])

        rep0 = tf.gather(rep, i0)
        rep1 = tf.gather(rep, i1)

        z_rep0 = tf.reduce_max(rep0, axis=0, keep_dims=True)
        z_rep1 = tf.reduce_max(rep1, axis=0, keep_dims=True)

        z_rep0_conc = tf.concat([z_rep0, self.z_norm],axis=1)
        z_rep1_conc = tf.concat([z_rep1, self.z_norm],axis=1)

        d0, weights_dis, weights_discore = self._build_discriminator_adversarial(z_rep0_conc, dim_in+dim_in, dim_d, do_out, FLAGS)
        d1, weights_dis, weights_discore = self._build_discriminator_adversarial(z_rep1_conc, dim_in+dim_in, dim_d, do_out, FLAGS, reuse=True)

        #gradient penalty
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((1, 1))
        interpolated = z_rep1 + alpha*(z_rep0-z_rep1)
        interpolated_conc = tf.concat([interpolated, self.z_norm],axis=1)
        inte_logit, weights_dis, weights_discore = self._build_discriminator_adversarial(interpolated_conc, dim_in+dim_in, dim_d, do_out, FLAGS, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_l2-1.0))

        return d0, d1, gradient_penalty, weights_dis, weights_discore


