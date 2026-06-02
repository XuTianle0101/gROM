# Copyright 2023 Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import torch as th
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import tools.io_utils as io
from network1d.meshgraphnet import MeshGraphNet
import json
from dataclasses import dataclass
from pathlib import Path
from network1d.rollout import rollout
import tools.plot_tools as pt


@dataclass
class EvaluationConfig:
    model_path: str
    data_location: str = None
    graphs_folder: str = 'graphs/'
    results_dir: str = 'results'
    device: str = None
    plot: bool = False


def with_trailing_slash(path):
    if path is None:
        return None
    return str(path).replace('\\', '/').rstrip('/') + '/'


def resolve_device(device_name = None):
    if device_name is not None:
        return th.device(device_name)
    return th.device("cuda:0" if th.cuda.is_available() else "cpu")


def plot_rollout(features, graph, params, folder, filename = 'all_nodes.mp4'):
    """
    Saves videos with all nodal values for pressure and flow rate, for all
    timesteps

    Arguments:
        features: 3D array containing the GNN prediction. 
                  1 dim: graph nodes, 2 dim: pressure (0), florate (1),
                  3 dim: timestep index
        params: dictionary of parameters
        folder (string): path where the video should be saved
        file_name: name of output file. Default -> 'all_nodes.mp4'

    """
    pt.video_all_nodes(features, graph, params, 5, str(Path(folder) / filename))


def evaluate_all_models(dataset, split_name, gnn_model, params, doplot = False,
                        results_dir = 'results', device = None):
    """
    Runs the rollout phase for all models and computes errors.

    Arguments:
        dataset: dictionary containing two keys, 'train' and 'test', with
                 the different datasets
        split_name (string): either 'train' or 'test'
        gnn_model: the GNN
        params: dictionary of parameters
        doplot: if True, the functions creates and saves one video per simulation. Default -> False
    Returns:
        2D array containing average pressure and flow rate normalized errors
        2D array containing average pressure and flow rate errors, 
        Average continuity loss
        Average run time
        Average number of timesteps

    """
    print('==========' + split_name + '==========')
    dataset = dataset[split_name]
    device = resolve_device(device)
    if doplot:
        (Path(results_dir) / split_name).mkdir(parents=True, exist_ok=True)

    total_timesteps = 0
    total_time = 0
    tot_errs_normalized = 0
    tot_errs = 0
    tot_cont_loss = 0
    for i in range(0,len(dataset.graphs)):
        print('model name = {}'.format(dataset.graph_names[i]))
        fdr = Path(results_dir) / split_name / dataset.graph_names[i]
        if doplot:
            fdr.mkdir(parents=True, exist_ok=True)
        graph = dataset.graphs[i].to(device)
        with th.no_grad():
            r_features, errs_normalized, errs, _, elaps = rollout(
                gnn_model, params, graph
            )
        total_time = total_time + elaps
        total_timesteps = total_timesteps + r_features.shape[2]
        print('Errors')
        print(errs)
        if doplot:
            plot_rollout(r_features, dataset.graphs[i], params, fdr)
        tot_errs_normalized = tot_errs_normalized + errs_normalized
        tot_errs = tot_errs + errs

    N = len(dataset.graphs)
    print('-------------------------------------')
    print('Global statistics')
    print('Errors')
    print(tot_errs / N)
    print('Average time = {:.2f}'.format(total_time / N))
    print('Average n timesteps = {:.2f}'.format(total_timesteps / N))

    return tot_errs_normalized/N, tot_errs/N, tot_cont_loss/N, \
           total_time / N, total_timesteps / N

def get_gnn_and_graphs(path, graphs_folder = 'graphs/',
                       data_location = None, device = None):

    """
    Get GNN and list of graphs given the path to a saved model folder.

    Arguments:
        path (string): path to the GNN model folder. This should be the output
                       generated when launching the 'network1d/training.py'
                       script
        graphs_folder: name of folder containing graphs
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None
    Returns:
        GNN model
        List of graphs
        Dictionary containing parameters
    """
    
    device = resolve_device(device)
    params = json.load(open(Path(path) / 'parameters.json'))

    gnn_model = MeshGraphNet(params)

    state_dict = th.load(Path(path) / 'trained_gnn.pms', map_location=device)
    gnn_model.load_state_dict(state_dict)

    gnn_model = gnn_model.to(device)
    gnn_model.eval()

    if data_location == None:
        data_location = io.data_location()
    input_dir = Path(data_location) / graphs_folder
    graphs, _  = gng.generate_normalized_graphs(with_trailing_slash(input_dir),
                                                params['statistics']
                                                      ['normalization_type'],
                                                params['bc_type'],
                                                statistics = params 
                                                             ['statistics'])

    return gnn_model, graphs, params

def get_dataset_and_gnn(path, graphs_folder = 'graphs/', data_location = None,
                        device = None):
    """
    Get datasets and GNN given the path to a saved model folder.

    Arguments:
        path (string): path to the GNN model folder. This should be the output
                       generated when launching the 'network1d/training.py'
                       script
        graphs_folder: name of folder containing graphs
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None
    Returns:
        Dictionary containing train and test datasets
        GNN model
        Dictionary containing parameters

    """
    gnn_model, graphs, params = get_gnn_and_graphs(path,
                                                   graphs_folder,
                                                   data_location,
                                                   device)

    dataset = dset.generate_dataset_from_params(graphs, params)
    return dataset, gnn_model, params

def parse_args():
    """Parse command line arguments for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a trained gROM model')
    parser.add_argument('model_path', help='Path to trained model directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate rollout videos in results/')
    parser.add_argument('--data-location', help='folder containing graph data',
                        default=None)
    parser.add_argument('--graphs-folder', help='folder of graphs under data location',
                        default='graphs/')
    parser.add_argument('--results-dir', help='folder where rollout results are saved',
                        default='results')
    parser.add_argument('--device', help='torch device, for example cpu or cuda:0',
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = EvaluationConfig(
        model_path=args.model_path,
        data_location=args.data_location,
        graphs_folder=args.graphs_folder,
        results_dir=args.results_dir,
        device=args.device,
        plot=args.plot,
    )
    device = resolve_device(config.device)
    print(f"Using device: {device}")

    dataset, gnn_model, params = get_dataset_and_gnn(
        config.model_path,
        config.graphs_folder,
        config.data_location,
        device,
    )
    params['results_dir'] = config.results_dir

    evaluate_all_models(dataset, 'train', gnn_model, params, config.plot,
                        config.results_dir, device)
    evaluate_all_models(dataset, 'test', gnn_model, params, config.plot,
                        config.results_dir, device)
