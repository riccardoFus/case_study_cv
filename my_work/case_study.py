import sys, os, random, shutil
from sys import platform

_base_path = "."
sys.path.append(_base_path)

from src.helpers.utils import get_device
import numpy as np
from monai.utils import set_determinism
from src.helpers.utils import make_dataset, get_device
import src.modules.plotting as plot
from src.modules.training import train_test_splitting, training_model, predict_ae
from src.helpers.config import get_config
from src.modules.preprocessing import get_transformations
from src.models.autoencoder3d import AutoEncoder3D
from src.helpers.graph import image_to_graph
from src.my_work.graph2 import image_to_graph_2
from src.modules.training import training_model, predict_gnn
import torch
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.data import Batch
from src.models.gnn import GraphSAGE, GAT, ChebNet
torch.serialization.add_safe_globals([Data, DataEdgeAttr, Batch, DataTensorAttr, GlobalStorage])
from torch_geometric.explain import Explainer, ModelConfig, ThresholdConfig
from torch_geometric.explain.algorithm import GNNExplainer
import pandas as pd

data_path = "data\\glioma\\glioma\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

set_determinism(seed=3)
random.seed(3)

saved_path = "saved"
reports_path = "reports"
logs_path = "logs"
graph_path = "graph\\graphs"

(
	base_image_transform,
	autoencoder_train_transform,
	autoencoder_eval_transform,
	_, _
) = get_transformations()

model = AutoEncoder3D(
	spatial_dims=3,
	in_channels=4,
	out_channels=4,
	channels=(5,),
	strides=(2,),
	inter_channels=(8, 8, 16),
	inter_dilations=(1, 2, 4)
)


train_data, eval_data, test_data = train_test_splitting(data_path, reports_path=reports_path, load_from_file=False)

## NOTE: uncomment to execute the training phase.

# train_metrics = training_model(
# 	model = model,
# 	data = [train_data[:50], eval_data[:20]],
# 	transforms = [autoencoder_train_transform, autoencoder_eval_transform],
# 	epochs = 10,
# 	device = get_device(), # get_device(), # 'mps' not supported
# 	paths = [saved_path, reports_path, logs_path, ''],
# 	num_workers=0,
# 	verbose = True
# )

l = len(train_data) + len(eval_data) + len(test_data)
data = np.concatenate([train_data, eval_data, test_data]).reshape(l)

write_to_file = "graph"
# In your main script where you generate graphs
num_node_features = 50 # This should match your GNN definition

# for item in data:
#     graph, centroids = image_to_graph_2(
#             item,
#             model=model,
#             write_to_file=write_to_file,
#             saved_path=saved_path,
#             # final_feature_dim=num_node_features # Pass the correct dimension
#         )
    

# defining default settings
num_node_features = 50			# Input feature size
num_classes = 4					# Number of output classes
lr = 1e-4						# Learning rate for the optimizier
weight_decay = 1e-5				# Weight decay for the optimizier
dropout = .1					# Dropout probability (for features)
hidden_channels = [512, 512, 512, 512, 512, 512, 512] # No. of hidden units (input layer included output layer excluded)
# GRAPHSAGE PARAMS
aggr = 'mean'				# Apply pooling operation as aggregator
# GAT PARAMS
heads = 14					# Number of attention heads
attention_dropout = .2		# Dropout probability (for attention mechanism)
# CHEBNET PARAMS
k = 3						# Chebyshev polynomial order


# defining models
_models = {
	'AutoEncoder3D': AutoEncoder3D(
		spatial_dims=3,
		in_channels=4,
		out_channels=4,
		channels=(5,),
		strides=(2,),
		inter_channels=(8, 8, 16),
		inter_dilations=(1, 2, 4)
	),
	'GraphSAGE': GraphSAGE(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		aggr = aggr
	),
	'GAT': GAT(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		heads = heads,
		attention_dropout = attention_dropout
	),
	'ChebNet': ChebNet(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		K = k
	)
}
model = _models['ChebNet']


# get data transformations pipelines
(
	_,
	autoencoder_train_transform,
	autoencoder_eval_transform,
	gnn_train_eval_transform,
	gnn_test_transform
) = get_transformations(graph_path)
transforms = [autoencoder_train_transform, autoencoder_eval_transform] if model.name == 'AutoEncoder3D' else [gnn_train_eval_transform, gnn_train_eval_transform]
# 
# torch.cuda.empty_cache()
# 
# train_metrics = training_model(
# 	model = model,
# 	data = [train_data, eval_data],
# 	transforms = transforms,
# 	epochs = 100 if model.name == 'AutoEncoder3D' else 500,
# 	device = get_device(),
# 	paths = [saved_path, reports_path, logs_path, graph_path],
# 	early_stopping = 10 if model.name == 'AutoEncoder3D' else 25,
# 	ministep = 14 if model.name == 'AutoEncoder3D' else 6,
# 	lr = 1e-4 if model.name == 'AutoEncoder3D' else lr,
# 	weight_decay = 1e-5 if model.name == 'AutoEncoder3D' else weight_decay
# )
# 
# best_runs = plot.best_config(reports_path)
# 
# plot.training_values(reports_path, best_runs)
# 
# for i, t in enumerate(test_data):
#    if i % 50 == 0 and i > 0:
#        print(f"inference {i}/{len(test_data)}")
#    test_metrics, predictions = predict_gnn(
#        model = model,
#        data = [t],
#        transforms = gnn_test_transform,
#        device = get_device(), # get_device(), # 'mps' not supported
#        paths = [saved_path, reports_path, logs_path],
#        return_predictions = True,
#        verbose = False, 
#    )
# for m in ['GRAPHSAGE_testing.csv', 'GAT_testing.csv','CHEBNET_testing.csv']:
# 	print('\n')
# 	df = pd.read_csv(os.path.join(reports_path, m), encoding='UTF-8')
# 	print(df['node_dice_score'].idxmax(), df['dice_score_et'].idxmax(), df['dice_score_tc'].idxmax(), df['dice_score_wt'].idxmax())
# 	print(df['node_dice_score'].idxmin(), df['dice_score_et'].idxmin(), df['dice_score_tc'].idxmin(), df['dice_score_wt'].idxmin())
# 	print(df['node_dice_score'].mean(), df['dice_score_et'].mean(), df['dice_score_tc'].mean(), df['dice_score_wt'].mean())
# 	af = df[(df['node_dice_score'] == .0) | (df['dice_score_et'] == .0) | (df['dice_score_tc'] == .0) | (df['dice_score_wt'] == .0)]
# 	bad_df = df.index.isin(list(af.index))
# 	print(af.index)
# 	print(df[~bad_df]['node_dice_score'].mean(), df[~bad_df]['dice_score_et'].mean(), df[~bad_df]['dice_score_tc'].mean(), df[~bad_df]['dice_score_wt'].mean())

example = test_data[223]
trans_test_data = gnn_test_transform(example)

data = trans_test_data['graph']

model_config = ModelConfig(
	mode="multiclass_classification",
	task_level="node",
	return_type="log_probs",
)
gnn_explainer = GNNExplainer(epochs=200, lr=1e-4)
explainer = Explainer(
	model=model,
	algorithm=gnn_explainer,
	explanation_type="model",
	model_config=model_config,
	node_mask_type="attributes",
	edge_mask_type=None,
	threshold_config=ThresholdConfig(threshold_type="topk", value=int(0.5 * data.x.shape[0]))
	# threshold_config=ThresholdConfig(threshold_type="hard", value=.8)
)
predicted_labels = model(data.x, data.edge_index.type(torch.int64)).argmax(dim=1)
explanation = explainer(
	x=data.x,
	edge_index=data.edge_index.type(torch.int64),
	target=predicted_labels
)