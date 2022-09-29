import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

import ABCEI.ab_net as abnet
from ABCEI.util import *
from sklearn import metrics

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_integer('n_dc', 2, """Number of discriminator layers. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_beta', 10.0, """Gradient penalty weight. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_mi', 100, """MI estimation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_d', 100, """Discriminator layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/ihdp/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True

def train(ABNet, sess, train_step, train_discriminator_step, train_encoder_step, train_hrep_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains a ABNet model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    z_norm = np.random.normal(0.,1.,(1,FLAGS.dim_in))

    ''' Set up loss feed_dicts'''
    dict_factual = {ABNet.x: D['x'][I_train,:], ABNet.t: D['t'][I_train,:], ABNet.y_: D['yf'][I_train,:], \
      ABNet.do_in: 1.0, ABNet.do_out: 1.0, ABNet.r_lambda: FLAGS.p_lambda, ABNet.r_beta: FLAGS.p_beta, ABNet.p_t: p_treated, ABNet.z_norm: z_norm}

    if FLAGS.val_part > 0:
        dict_valid = {ABNet.x: D['x'][I_valid,:], ABNet.t: D['t'][I_valid,:], ABNet.y_: D['yf'][I_valid,:], \
          ABNet.do_in: 1.0, ABNet.do_out: 1.0, ABNet.r_lambda: FLAGS.p_lambda, ABNet.r_beta: FLAGS.p_beta, ABNet.p_t: p_treated, ABNet.z_norm: z_norm}

    if D['HAVE_TRUTH']:
        dict_cfactual = {ABNet.x: D['x'][I_train,:], ABNet.t: 1-D['t'][I_train,:], ABNet.y_: D['ycf'][I_train,:], \
          ABNet.do_in: 1.0, ABNet.do_out: 1.0, ABNet.z_norm: z_norm}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, gmi_err, discriminator_loss, rep_loss, gradient_pen = \
    sess.run([ABNet.tot_loss, ABNet.pred_loss, ABNet.gmi_neg_loss, ABNet.discriminator_loss, ABNet.rep_loss, ABNet.dp],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(ABNet.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp = \
        sess.run([ABNet.tot_loss, ABNet.pred_loss, ABNet.gmi_neg_loss, ABNet.discriminator_loss, ABNet.rep_loss, ABNet.dp],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, gmi_err, discriminator_loss, rep_loss, gradient_pen,\
        valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        # I = random.sample(range(0, n_train), FLAGS.batch_size)
        # x_batch = D['x'][I_train,:][I,:]
        # t_batch = D['t'][I_train,:][I]
        # y_batch = D['yf'][I_train,:][I]

        I = list(range(0, n_train))
        np.random.shuffle(I)
        for i_batch in range(n_train // FLAGS.batch_size):
            if i_batch < n_train // FLAGS.batch_size - 1:
                I_b = I[i_batch * FLAGS.batch_size:(i_batch+1) * FLAGS.batch_size]
            else:
                I_b = I[i_batch * FLAGS.batch_size:]
            x_batch = D['x'][I_train,:][I_b,:]
            t_batch = D['t'][I_train,:][I_b]
            y_batch = D['yf'][I_train,:][I_b]

            z_norm_batch = np.random.normal(0.,1.,(1,FLAGS.dim_in))

            if __DEBUG__:
                M = sess.run(ABNet.pop_dist(ABNet.x, ABNet.t), feed_dict={ABNet.x: x_batch, ABNet.t: t_batch})
                log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

            ''' Do one step of gradient descent '''
            if not objnan:
                sess.run(train_hrep_step, feed_dict={ABNet.x: x_batch, \
                    ABNet.do_in: FLAGS.dropout_in, ABNet.do_out: FLAGS.dropout_out})

                #train discriminator
                for sub_dc in range(0,3):
                    sess.run(train_discriminator_step, feed_dict={ABNet.x: x_batch, ABNet.t: t_batch, ABNet.r_beta: FLAGS.p_beta, \
                        ABNet.do_in: FLAGS.dropout_in, ABNet.do_out: FLAGS.dropout_out, ABNet.z_norm: z_norm_batch})
                #train encoder
                # for sub_enc in range(0,2):
                sess.run(train_encoder_step, feed_dict={ABNet.x: x_batch, ABNet.t: t_batch, \
                    ABNet.do_in: FLAGS.dropout_in, ABNet.do_out: FLAGS.dropout_out, ABNet.z_norm: z_norm_batch})

                sess.run(train_step, feed_dict={ABNet.x: x_batch, ABNet.t: t_batch, \
                    ABNet.y_: y_batch, ABNet.do_in: FLAGS.dropout_in, ABNet.do_out: FLAGS.dropout_out, \
                    ABNet.r_lambda: FLAGS.p_lambda, ABNet.p_t: p_treated})

            ''' Project variable selection weights '''
            if FLAGS.varsel:
                wip = simplex_project(sess.run(ABNet.weights_in[0]), 1)
                sess.run(ABNet.projection, feed_dict={ABNet.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,gmi_err, discriminator_loss, rep_loss, gradient_pen = \
            sess.run([ABNet.tot_loss, ABNet.pred_loss, ABNet.gmi_neg_loss, ABNet.discriminator_loss, ABNet.rep_loss, ABNet.dp],
                feed_dict=dict_factual)

            rep = sess.run(ABNet.h_rep_norm, feed_dict={ABNet.x: D['x'], ABNet.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(ABNet.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp = \
                sess.run([ABNet.tot_loss, ABNet.pred_loss, ABNet.gmi_neg_loss, ABNet.discriminator_loss, ABNet.rep_loss, ABNet.dp], \
                    feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, gmi_err, discriminator_loss, rep_loss, gradient_pen,\
                valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f, \tGMI: %.3f, \tdc_loss: %.3f, \trep_loss: %.3f, \tdp: %.3f, \tVal: %.3f, \tValGMI: %.3f, \tValdc: %.3f, \tValrep: %.3f, \tValdp: %.3f, \tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, -gmi_err, discriminator_loss, rep_loss, gradient_pen, valid_f_error, -valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(ABNet.output, feed_dict={ABNet.x: x_batch, \
                    ABNet.t: t_batch, ABNet.do_in: 1.0, ABNet.do_out: 1.0})
                # y_pred = 1.0*(y_pred > 0.5)
                # acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))

                fpr, tpr, thresholds = metrics.roc_curve(y_batch, y_pred)
                auc = metrics.auc(fpr, tpr)

                loss_str += ',\tAuc_batch: %.2f' % auc

            # log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(ABNet.output, feed_dict={ABNet.x: D['x'], \
                ABNet.t: D['t'], ABNet.do_in: 1.0, ABNet.do_out: 1.0})
            y_pred_cf = sess.run(ABNet.output, feed_dict={ABNet.x: D['x'], \
                ABNet.t: 1-D['t'], ABNet.do_in: 1.0, ABNet.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if FLAGS.loss == 'log' and D['HAVE_TRUTH']:
                fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D['yf'], D['ycf']),axis=0), \
                    np.concatenate((y_pred_f, y_pred_cf),axis=0))
                auc = metrics.auc(fpr, tpr)
                loss_str += ',\tAuc_train: %.2f' % auc

            if D_test is not None:
                y_pred_f_test = sess.run(ABNet.output, feed_dict={ABNet.x: D_test['x'], \
                    ABNet.t: D_test['t'], ABNet.do_in: 1.0, ABNet.do_out: 1.0})
                y_pred_cf_test = sess.run(ABNet.output, feed_dict={ABNet.x: D_test['x'], \
                    ABNet.t: 1-D_test['t'], ABNet.do_in: 1.0, ABNet.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

                if FLAGS.loss == 'log' and D['HAVE_TRUTH']:
                    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D_test['yf'], D_test['ycf']),axis=0), \
                        np.concatenate((y_pred_f_test, y_pred_cf_test),axis=0))
                    auc = metrics.auc(fpr, tpr)
                    loss_str += ',\tAuc_test: %.2f' % auc

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([ABNet.h_rep], feed_dict={ABNet.x: D['x'], \
                    ABNet.do_in: 1.0, ABNet.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([ABNet.h_rep], feed_dict={ABNet.x: D_test['x'], \
                        ABNet.do_in: 1.0, ABNet.do_out: 0.0})
                    reps_test.append(reps_test_i)

            log(logfile, loss_str)

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: beta=%.2g, lambda=%.2g' % (FLAGS.p_beta,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    znorm = tf.placeholder("float", shape=[None, FLAGS.dim_in], name='z_norm')

    ''' Parameter placeholders '''
    r_lambda = tf.placeholder("float", name='r_lambda')
    r_beta = tf.placeholder("float", name='r_beta')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, FLAGS.dim_mi, FLAGS.dim_d]
    ABNet = abnet.ab_net(x, t, y_, p, znorm, FLAGS, r_lambda, r_beta, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    lr_gan = 5e-5

    counter_enc = tf.Variable(0, trainable=False)
    lr_enc = tf.train.exponential_decay(lr_gan, counter_enc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_dc = tf.Variable(0, trainable=False)
    lr_dc = tf.train.exponential_decay(lr_gan, counter_dc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_gmi = tf.Variable(0, trainable=False)
    lr_gmi = tf.train.exponential_decay(FLAGS.lrate, counter_gmi, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=lr_enc, 
            beta1=0.5, 
            beta2=0.9)
        opt_dc = tf.train.AdamOptimizer(
            learning_rate=lr_dc, 
            beta1=0.5, 
            beta2=0.9)
        opt_gmi = tf.train.AdamOptimizer(lr_gmi)
    else:
        opt = tf.train.RMSPropOptimizer(lr_gan)
        opt_enc = tf.train.RMSPropOptimizer(lr_gan)
        opt_dc = tf.train.RMSPropOptimizer(lr_gan)
        opt_gmi = tf.train.RMSPropOptimizer(lr_gan)
    

    ''' Unused gradient clipping '''
    #gvs = opt.compute_gradients(ABNet.tot_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    #train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    #var_scope_get
    var_enc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    var_gmi = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='gmi')
    var_gmi.extend(var_enc)
    var_dc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    var_pred = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
    var_pred.extend(var_enc)

    print("var_enc:",[v.name for v in var_enc])
    print()
    print("var_gmi:",[v.name for v in var_gmi])
    print()
    print("var_dc:",[v.name for v in var_dc])
    print()
    print("var_pred:",[v.name for v in var_pred])
    print()

    train_hrep_step = opt_gmi.minimize(ABNet.gmi_neg_loss,global_step=counter_gmi,var_list=var_gmi)
    train_discriminator_step = opt_dc.minimize(ABNet.discriminator_loss,global_step=counter_dc,var_list=var_dc)
    train_encoder_step = opt_enc.minimize(ABNet.rep_loss,global_step=counter_enc,var_list=var_enc)
    train_step = opt.minimize(ABNet.tot_loss,global_step=global_step,var_list=var_pred)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(ABNet, sess, train_step, train_discriminator_step, train_encoder_step, train_hrep_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        # print(all_losses)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(ABNet.weights_in[0])
                all_beta = sess.run(ABNet.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(ABNet.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(ABNet.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
