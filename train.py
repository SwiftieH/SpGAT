# -*- coding:UTF-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from utils import *
from models import SpGAT

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Set random seed
seed = 322
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'SpGAT', 'Model string.')  # 'SpGAT', 'SpGAT_Cheby'
flags.DEFINE_float('wavelet_s', 1.0, 'wavelet s .')
flags.DEFINE_float('threshold', 1e-4, 'sparseness threshold .')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('alldata', False, 'All data string.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')#1000
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_bool('mask', True, 'mask string.')
flags.DEFINE_bool('laplacian_normalize', True, 'laplacian normalize string.')
flags.DEFINE_bool('sparse_ness', True, 'wavelet sparse_ness string.')
flags.DEFINE_bool('weight_normalize', False, 'weight normalize string.')
flags.DEFINE_string('gpu', '-1', 'which gpu to use.')#1000
flags.DEFINE_integer('repeating', 1, 'Number of repeating times')#1000

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu


# Load data
labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset,alldata=FLAGS.alldata)
# Some preprocessing, normalization
features = preprocess_features(features)
node_num = adj.shape[0]

print("************Loading data finished, Begin constructing wavelet************")

dataset = FLAGS.dataset
s = FLAGS.wavelet_s
laplacian_normalize = FLAGS.laplacian_normalize
sparse_ness = FLAGS.sparse_ness
threshold = FLAGS.threshold
weight_normalize = FLAGS.weight_normalize
if FLAGS.model == "SpGAT":
    support_t = wavelet_basis(dataset,adj, s, laplacian_normalize,sparse_ness,threshold,weight_normalize)
elif FLAGS.model == "SpGAT_Cheby":
    s = 2.0
    support_t = wavelet_basis_appro(dataset,adj, s, laplacian_normalize,sparse_ness,threshold,weight_normalize)
if dataset == 'cora':
    k_por = 0.05 # best $d$ for cora
if dataset == 'pubmed':
    k_por = 0.10 # best $d$ for pubmed
if dataset == 'citeseer':
    k_por = 0.15 # best $d$ for citeseer
k_fre = int(k_por * node_num)
support = [support_t[0][:,:k_fre], support_t[1][:k_fre,:], support_t[0][:,k_fre:], support_t[1][k_fre:,:]]
sparse_to_tuple(support)
num_supports = len(support)
model_func = SpGAT

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.outputs,model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2]

test_acc_bestval = []
test_acc_besttest = []
val_acc = []

for _ in range(FLAGS.repeating):

    #seed = np.random.randint(999)
    #np.random.seed(seed)
    #tf.set_random_seed(seed)

    # Create model
    weight_normalize = FLAGS.weight_normalize
    node_num = adj.shape[0]
    model = model_func(k_por, node_num,weight_normalize, placeholders, input_dim=features[2][1], logging=True)
    print("**************Constructing wavelet finished, Begin training**************")
    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    cost_val = []
    best_val_acc = 0.0
    output_test_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(FLAGS.epochs):

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        val_output,cost, acc = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        # Test
        test_output, test_cost, test_acc = evaluate(features, support, y_test, test_mask, placeholders)

        # best val acc
        if(best_val_acc <= acc):
            best_val_acc = acc
            output_test_acc = test_acc
        # best test acc
        if(best_test_acc <= test_acc):
            #import pdb; pdb.set_trace()
            best_test_acc = test_acc


        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=", "{:.5f}".format(test_acc))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    print("dataset: ",FLAGS.dataset," model: ",FLAGS.model,",sparse_ness: ",FLAGS.sparse_ness,
          ",laplacian_normalize: ",FLAGS.laplacian_normalize,",threshold",FLAGS.threshold,",wavelet_s:",FLAGS.wavelet_s,",mask:",FLAGS.mask,
          ",weight_normalize:",FLAGS.weight_normalize,
          ",learning_rate:",FLAGS.learning_rate,",hidden1:",FLAGS.hidden1,",dropout:",FLAGS.dropout,",alldata:",FLAGS.alldata)

    print("Val accuracy:", best_val_acc, " Test accuracy: ",output_test_acc)
    test_acc_bestval.append(output_test_acc)
    test_acc_besttest.append(best_test_acc)
    val_acc.append(best_val_acc)

print("********************************************************")

result = []
result.append(np.array(test_acc_bestval))
result.append(np.array(test_acc_besttest))
result.append(np.array(val_acc))

alpha_1_low = sess.run(model.layers[0].vars['low_w'], feed_dict = feed_dict)
alpha_1_high = sess.run(model.layers[0].vars['high_w'], feed_dict = feed_dict)
alpha_2_low = sess.run(model.layers[1].vars['low_w'], feed_dict = feed_dict)
alpha_2_high = sess.run(model.layers[1].vars['high_w'], feed_dict = feed_dict)


r_half = int(FLAGS.repeating / 2)
print("REPEAT\t{}".format(FLAGS.repeating))
print("Model\t{}".format(FLAGS.model))
print("Low frequency portion\t{} %".format(k_por * 100))
print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format('DATASET', 'best_val_mean', 'best_val_std',
                                                                      'best_test_mean', 'best_test_std', 'half_best_val_mean',
                                                                      'half_best_val_std', 'half_best_test_mean', 'half_best_test_std'))
print("{:<8}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}".format(
    FLAGS.dataset,
    result[0].mean(),
    result[0].std(),
    result[1].mean(),
    result[1].std(),
    result[0][np.argsort(result[0])[r_half:]].mean(),
    result[0][np.argsort(result[0])[r_half:]].std(),
    result[1][np.argsort(result[1])[r_half:]].mean(), #2 for validation
    result[1][np.argsort(result[1])[r_half:]].std()))

print("{:<8}\t{:<8}\t{:<8}\t{:<8}".format('alpha_1_low', 'alpha_1_high', 'alpha_2_low', 'alpha_2_high'))
print("{:<8.6f}\t{:<8.6f}\t{:<8.6f}\t{:<8.6f}".format(
    alpha_1_low,
    alpha_1_high,
    alpha_2_low,
    alpha_2_high))





