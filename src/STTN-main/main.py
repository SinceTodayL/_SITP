import os
import tensorflow as tf
import numpy as np
import argparse
from os.path import join as pjoin

from utils.math_graph import scaled_laplacian, cheb_poly_approx
from models.trainer import model_train
from models.tester import model_test
from data_loader.data_utils import load_custom_dataset  # <-- 你新写的加载器

# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=1127)
parser.add_argument('--n_his', type=int, default=8)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
args = parser.parse_args()

# Graph structure
if args.graph == 'default':
    W = np.identity(args.n_route)
else:
    W = np.loadtxt(pjoin('./dataset', args.graph), delimiter=',')
L = scaled_laplacian(W)
Lk = cheb_poly_approx(L, args.ks, args.n_route)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Spatial embeddings
V, U = np.linalg.eig(L)
U_ini = U[:, :32]
tf.add_to_collection(name='intial_spatial_embeddings', value=tf.cast(tf.constant(U_ini), tf.float32))

# Load custom dataset
csv_path = './data_loader/Maglev/Data.csv'
PeMS = load_custom_dataset(csv_path, args.n_his, args.n_pred)

print(">> Data loaded, start training...")
model_train(PeMS, blocks=[[64, 64]], args=args)
model_test(PeMS, args.batch_size, args.n_his, args.n_pred, args.inf_mode)
