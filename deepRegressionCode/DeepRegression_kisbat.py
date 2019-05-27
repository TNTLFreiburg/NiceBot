
import logging
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger()


import braindecode
import numpy as np
from numpy.random import RandomState
import mne

from collections import OrderedDict

import os
os.sys.path.append('/home/fiederer/nicebot/deepRegressionCode/')
from arg_parser import parse_run_args
import time
import fnmatch
import shortuuid
import glob
import pandas as pd
import joblib # to save models
#from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing, cut_cos, CutCosineAnnealing
os.sys.path.append('/home/fiederer/adamw-eeg-eval/')
from adamweegeval.schedulers import ScheduledOptimizer, CosineAnnealing, cut_cos, CutCosineAnnealing
from adamweegeval.optimizers import AdamW

from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.iterators import ClassBalancedBatchSizeIterator
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.datautil.splitters import concatenate_sets
from braindecode.experiments.monitors import AveragePerClassMisclassMonitor
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, CroppedTrialMisclassMonitor, MisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.mne_ext.signalproc import resample_cnt, mne_apply
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint

import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations

from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX
from braindecode.visualization.plot import ax_scalp

from braindecode.util import wrap_reshape_apply_fn, corr
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import var_to_np

import torch as th
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import elu
from torch.nn import init

from braindecode.torch_ext.modules import Expression
import scipy.io
from scipy.stats import pearsonr

from copy import deepcopy

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


############################################## class definitions ##########################################################


class EEGResNet(object):
    """
    Residual Network for EEG.
    """
    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 final_pool_length,
                 n_first_filters,
                 n_layers_per_block=2,
                 first_filter_length=3,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 batch_norm_epsilon=1e-4,
                 conv_weight_init_fn=lambda w: init.kaiming_normal(w, a=0)):
        if final_pool_length == 'auto':
            assert input_time_length is not None
        assert first_filter_length % 2 == 1
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        print('creating ResNet!')
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module('dimshuffle', Expression(_transpose_time_to_spat))
            model.add_module('conv_time', nn.Conv2d(1, self.n_first_filters,
                                                    (
                                                    self.first_filter_length, 1),
                                                    stride=1,
                                                    padding=(self.first_filter_length // 2, 0)))
            model.add_module('conv_spat',
                             nn.Conv2d(self.n_first_filters, self.n_first_filters,
                                       (1, self.in_chans),
                                       stride=(1, 1),
                                       bias=False))
        else:
            model.add_module('conv_time',
                             nn.Conv2d(self.in_chans, self.n_first_filters,
                                       (self.first_filter_length, 1),
                                       stride=(1, 1),
                                       padding=(self.first_filter_length // 2, 0),
                                       bias=False,))
        n_filters_conv = self.n_first_filters
        model.add_module('bnorm',
                         nn.BatchNorm2d(n_filters_conv,
                                        momentum=self.batch_norm_alpha,
                                        affine=True,
                                        eps=1e-5),)
        model.add_module('conv_nonlin', Expression(self.nonlinearity))
        cur_dilation  = np.array([1,1])
        n_cur_filters = n_filters_conv
        i_block = 1
        for i_layer in range(self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(2* n_cur_filters)
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_out_filters,
                                           dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(1.5* n_cur_filters)
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_out_filters,
                                           dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))


        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))


        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            model.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                             ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        model.add_module('res_{:d}_{:d}'.format(i_block, 0),
                                 ResidualBlock(n_cur_filters, n_cur_filters,
                                               dilation=cur_dilation, ))



        model.eval()
        if self.final_pool_length == 'auto':
            print('Final Pool length is auto!')
            out = model(np_to_var(np.ones(
                (1, self.in_chans, self.input_time_length,1),
                dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_pool_length = n_out_time
      #  model.add_module('mean_pool', AvgPool2dWithConv(
      #      (self.final_pool_length, 1), (1,1), dilation=(int(cur_dilation[0]),
       #                                                   int(cur_dilation[1]))))
       # model.add_module('conv_classifier',
        #                    nn.Conv2d(n_cur_filters, self.n_classes,
         #                              (1, 1), bias=True))
        
        # start added code martin
        model.add_module('conv_classifier',
                             nn.Conv2d(n_cur_filters, self.n_classes,
                                       (self.final_pool_length, 1), bias=True))
    #end added code martin

        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze',  Expression(_squeeze_final_output))


        # Initialize all weights
        model.apply(lambda module: weights_init(module, self.conv_weight_init_fn))

        # Start in eval mode
        model.eval()
        return model


def weights_init(module, conv_weight_init_fn):
    classname = module.__class__.__name__
    if 'Conv' in classname and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            init.constant(module.bias, 0)
    elif 'BatchNorm' in classname:
        init.constant(module.weight, 1)
        init.constant(module.bias, 0)


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:,:,:,0]
    if x.size()[2] == 1:
        x = x[:,:,0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


# create a residual learning building block with two stacked 3x3 convlayers as in paper
class ResidualBlock(nn.Module):
    def __init__(
        self, in_filters,
            out_num_filters,
            dilation,
            filter_time_length=3,
            nonlinearity=elu,
            batch_norm_alpha=0.1, batch_norm_epsilon=1e-4,
        ):
        super(ResidualBlock, self).__init__()
        time_padding = int((filter_time_length - 1) * dilation[0])
        assert time_padding % 2 == 0
        time_padding = int(time_padding // 2)
        dilation = (int(dilation[0]), int(dilation[1]))
        assert (out_num_filters - in_filters) % 2 == 0, (
            "Need even number of extra channels in order to be able to "
            "pad correctly")
        self.n_pad_chans = out_num_filters - in_filters

        self.conv_1 = nn.Conv2d(
            in_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0))
        self.bn1 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha, affine=True,
            eps=batch_norm_epsilon)
        self.conv_2 = nn.Conv2d(
           out_num_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
           dilation=dilation,
           padding=(time_padding, 0))
        self.bn2 = nn.BatchNorm2d(
           out_num_filters, momentum=batch_norm_alpha,
            affine=True, eps=batch_norm_epsilon)
        # also see https://mail.google.com/mail/u/0/#search/ilya+joos/1576137dd34c3127
        # for resnet options as ilya used them
        self.nonlinearity = nonlinearity


    def forward(self, x):
        stack_1 = self.nonlinearity(self.bn1(self.conv_1(x)))
        stack_2 = self.bn2(self.conv_2(stack_1)) # next nonlin after sum
        if self.n_pad_chans != 0:
            zeros_for_padding = th.autograd.Variable(
                th.zeros(x.size()[0], self.n_pad_chans // 2,
                         x.size()[2], x.size()[3]))
            if x.is_cuda:
                zeros_for_padding = zeros_for_padding.cuda()
            x = th.cat((zeros_for_padding, x, zeros_for_padding), dim=1)
        out = self.nonlinearity(x + stack_2)
        return out



def compute_amplitude_prediction_correlations_voltage(pred_fn, examples, n_iterations,
                                              perturb_fn=None,
                                              batch_size=30,
                                              seed=((2017, 7, 10))):
    """
    Changed function to calculate time-resolved voltage pertubations, and not frequency as original in compute_amplitude_prediction_correlations

    Perturb input amplitudes and compute correlation between amplitude
    perturbations and prediction changes when pushing perturbed input through
    the prediction function.    
    For more details, see [EEGDeepLearning]_.
    Parameters
    ----------
    pred_fn: function
    Function accepting an numpy input and returning prediction.
    examples: ndarray
    Numpy examples, first axis should be example axis.
    n_iterations: int
    Number of iterations to compute.
    perturb_fn: function, optional
    Function accepting amplitude array and random generator and returning
    perturbation. Default is Gaussian perturbation.
    batch_size: int, optional
    Batch size for computing predictions.
    seed: int, optional
    Random generator seed
    Returns
    -------
    amplitude_pred_corrs: ndarray
    Correlations between amplitude perturbations and prediction changes
    for all sensors and frequency bins.
    References
    ----------
    .. [EEGDeepLearning] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
    Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017).
    Deep learning with convolutional neural networks for EEG decoding and
    visualization.
    arXiv preprint arXiv:1703.05051.
    """
    inds_per_batch = get_balanced_batches(
    n_trials=len(examples), rng=None, shuffle=False, batch_size=batch_size)
    log.info("Compute original predictions...")
    orig_preds = [pred_fn(examples[example_inds])
              for example_inds in inds_per_batch]
    orig_preds_arr = np.concatenate(orig_preds)
    rng = RandomState(seed)
    fft_input = np.fft.rfft(examples, axis=2)
    amps = np.abs(fft_input)
    phases = np.angle(fft_input)

    amp_pred_corrs = []
    for i_iteration in range(n_iterations):
        log.info("Iteration {:d}...".format(i_iteration))
        log.info("Sample perturbation...")
        #modified part start
        perturbation = rng.randn(*examples.shape)
        new_in = examples + perturbation
        #modified part end
        log.info("Compute new predictions...")
        new_in = new_in.astype('float32')
        new_preds = [pred_fn(new_in[example_inds])
                     for example_inds in inds_per_batch]

        new_preds_arr = np.concatenate(new_preds)

        diff_preds = new_preds_arr - orig_preds_arr

        log.info("Compute correlation...")
        amp_pred_corr = wrap_reshape_apply_fn(corr, perturbation[:, :, :, 0],
                                              diff_preds,
                                              axis_a=(0,), axis_b=(0))
        amp_pred_corrs.append(amp_pred_corr)
    return amp_pred_corrs


 # %% monitor for correlation
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops

class CorrelationMonitor1d(object):
    """
    Compute correlation between 1d predictions

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        # this will be timeseries of predictions
        # for each trial
        # braindecode functions expect classes x time predictions
        # so add fake class dimension and remove it again
        preds_2d = [p[:, None] for p in all_preds]
        preds_per_trial = compute_preds_per_trial_from_crops(preds_2d,
                                                          self.input_time_length,
                                                          dataset.X)
        preds_per_trial = [p[0] for p in preds_per_trial]
        pred_timeseries = np.concatenate(preds_per_trial, axis=0)
        ys_2d = [y[:, None] for y in all_targets]
        targets_per_trial = compute_preds_per_trial_from_crops(ys_2d,
                                                            self.input_time_length,
                                                            dataset.X)
        targets_per_trial = [t[0] for t in targets_per_trial]
        target_timeseries = np.concatenate(targets_per_trial, axis=0)

        corr = np.corrcoef(target_timeseries, pred_timeseries)[0, 1]
        key = setname + '_corr'

        return {key: float(corr)}

# %% visualization (Kay): Phase and Amplitude perturbation
import torch
import numpy as np
from braindecode.util import wrap_reshape_apply_fn, corr
from braindecode.datautil.iterators import get_balanced_batches

class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_list):
        """
        Returns intermediate activations of a network during forward pass
        
        to_select: list of module names for which activation should be returned
        modules_list: Modules of the network in the form [[name1, mod1],[name2,mod2]...)
        
        Important: modules_list has to include all modules of the network, not only those of interest
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/8
        """
        super(SelectiveSequential, self).__init__()
        for key, module in modules_list:
            self.add_module(key, module)
            self._modules[key].load_state_dict(module.state_dict())
        self._to_select = to_select
    
    def forward(self,x):
        # Call modules individually and append activation to output if module is in to_select
        o = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                o.append(x)
        return o
    

def phase_perturbation(amps,phases,rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Shifts spectral phases randomly for input and frequencies, but same for all channels
    
    amps: Spectral amplitude (not used)
    phases: Spectral phases
    rng: Random Seed
    
    Output:
        amps_pert: Input amps (not modified)
        phases_pert: Shifted phases
        pert_vals: Absolute phase shifts
    """
    noise_shape = list(phases.shape)
    noise_shape[1] = 1 # Do not sample noise for channels individually
        
    # Sample phase perturbation noise
    phase_noise = rng.normal(0,np.pi,noise_shape).astype(np.float32)
    phase_noise = phase_noise.repeat(phases.shape[1],axis=1)
    # Apply noise to inputs
    phases_pert = phases+phase_noise
    phases_pert[phases_pert<-np.pi] += 2*np.pi
    phases_pert[phases_pert>np.pi] -= 2*np.pi
    
    return amps,phases_pert,np.abs(phase_noise)

def amp_perturbation_additive(amps,phases,rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Adds additive noise to amplitudes
    
    amps: Spectral amplitude
    phases: Spectral phases (not used)
    rng: Random Seed
    
    Output:
        amps_pert: Scaled amplitudes
        phases_pert: Input phases (not modified)
        pert_vals: Amplitude noise
    """
    amp_noise = rng.normal(0,0.02,amps.shape).astype(np.float32)
    amps_pert = amps+amp_noise
    amps_pert[amps_pert<0] = 0
    return amps_pert,phases,amp_noise

def amp_perturbation_multiplicative(amps,phases,rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Adds multiplicative noise to amplitudes
    
    amps: Spectral amplitude
    phases: Spectral phases (not used)
    rng: Random Seed
    
    Output:
        amps_pert: Scaled amplitudes
        phases_pert: Input phases (not modified)
        pert_vals: Amplitude scaling factor
    """
    amp_noise = rng.normal(1,0.02,amps.shape).astype(np.float32)
    amps_pert = amps*amp_noise
    amps_pert[amps_pert<0] = 0
    return amps_pert,phases,amp_noise

def correlate_feature_maps(x,y):
    """
    Takes two activation matrices of the form Bx[F]xT where B is batch size, F number of filters (optional) and T time points
    Returns correlations of the corresponding activations over T
    
    Input: Bx[F]xT (x,y)
    Returns: Bx[F]
    """
    shape_x = x.shape
    shape_y = y.shape
    assert np.array_equal(shape_x,shape_y)
    assert len(shape_x)<4
    x = x.reshape((-1,shape_x[-1]))
    y = y.reshape((-1,shape_y[-1]))
    
    corr_ = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        # Correlation of standardized variables
        corr_[i] = np.correlate((x[i]-x[i].mean())/x[i].std(),(y[i]-y[i].mean())/y[i].std())
    
    return corr_.reshape(*shape_x[:-1])

def mean_diff_feature_maps(x,y):
    """
    Takes two activation matrices of the form BxFxT where B is batch size, F number of filters and T time points
    Returns mean difference between feature map activations
    
    Input: BxFxT (x,y)
    Returns: BxF
    """
    return np.mean(x-y,axis=2)

def perturbation_correlation(pert_fn, diff_fn, pred_fn, n_layers, inputs, n_iterations,
                                                  batch_size=30,
                                                  seed=((2017, 7, 10))):
    """
    Calculates phase perturbation correlation for layers in network
    
    pred_fn: Function that returns a list of activations.
             Each entry in the list corresponds to the output of 1 layer in a network
    n_layers: Number of layers pred_fn returns activations for.
    inputs: Original inputs that are used for perturbation [B,X,T,1]
            Phase perturbations are sampled for each input individually, but applied to all X of that input
    n_iterations: Number of iterations of correlation computation. The higher the better
    batch_size: Number of inputs that are used for one forward pass. (Concatenated for all inputs)
    """
    rng = np.random.RandomState(seed)
    
    # Get batch indeces
    batch_inds = get_balanced_batches(
        n_trials=len(inputs), rng=rng, shuffle=False, batch_size=batch_size)
    
    # Calculate layer activations and reshape
    orig_preds = [pred_fn(inputs[inds])
                  for inds in batch_inds]
    orig_preds_layers = [np.concatenate([orig_preds[o][l] for o in range(len(orig_preds))])
                        for l in range(n_layers)]
    
    # Compute FFT of inputs
    fft_input = np.fft.rfft(inputs, n=inputs.shape[2], axis=2)
    amps = np.abs(fft_input)
    phases = np.angle(fft_input)
    
    pert_corrs = [0]*n_layers
    for i in range(n_iterations):
        #print('Iteration%d'%i)
        
        amps_pert,phases_pert,pert_vals = pert_fn(amps,phases,rng=rng)
        
        # Compute perturbed inputs
        fft_pert = amps_pert*np.exp(1j*phases_pert)
        inputs_pert = np.fft.irfft(fft_pert, n=inputs.shape[2], axis=2).astype(np.float32)
        
        # Calculate layer activations for perturbed inputs
        new_preds = [pred_fn(inputs_pert[inds])
                     for inds in batch_inds]
        new_preds_layers = [np.concatenate([new_preds[o][l] for o in range(len(new_preds))])
                        for l in range(n_layers)]
        
        for l in range(n_layers):
            # Calculate correlations of original and perturbed feature map activations
            preds_diff = diff_fn(orig_preds_layers[l][:,:,:,0],new_preds_layers[l][:,:,:,0])
            
            # Calculate feature map correlations with absolute phase perturbations
            pert_corrs_tmp = wrap_reshape_apply_fn(corr,
                                                   pert_vals[:,:,:,0],preds_diff,
                                                   axis_a=(0), axis_b=(0))
            pert_corrs[l] += pert_corrs_tmp
            
    pert_corrs = [pert_corrs[l]/n_iterations for l in range(n_layers)] #mean over iterations
    return pert_corrs

def save_params(exp):
    filename = exp.model_base_name + '.model_params.pkl'
    log.info("Save model params to {:s}".format(filename))
    th.save(exp.model.state_dict(), filename)
    filename = exp.model_base_name + '.trainer_params.pkl'
    log.info("Save trainer params to {:s}".format(filename))
    th.save(exp.optimizer.state_dict(), filename)

########################################################################################################


def run_experiment(
    unique_id=shortuuid.uuid(),
    data_folder='/data/schirrmr/fiederer/nicebot/data',
    batch_size=64,
    max_epochs=200,
    cuda=True,
    result_folder='/data/schirrmr/fiederer/nicebot/results',
    model_name='eegnet',
    init_lr=0.001,
    weight_decay=0,
    band_pass=[None, None],
    electrodes='*',
    sampling_rate=256,
    n_seconds_test_set=180,
    n_seconds_valid_set=180,
    data='onlyRobotData'
):

    # Set if you want to use GPU
        # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = cuda
    gpu_index = 0

    subjects =['noExp', 'moderateExp', 'substantialExp']

    n_seconds_test_set = n_seconds_test_set
    n_seconds_valid_set = n_seconds_valid_set

    save_addon_text_orig = '_' + str(n_seconds_test_set) + 'sTest'
    batch_size = batch_size
    pool_time_stride = 2

    sampling_rate = sampling_rate

    max_train_epochs = max_epochs

    # viz
    calc_viz = False
    n_perturbations = 20

    # which model
    model_name=model_name # deep4, resnet, eegnet, lin_reg, lin_svr, rbf_svr, rf_reg
    deep4 = False
    res_net = False
    eeg_net_v4 = False
    lin_reg = False
    lin_svr = False
    rbf_svr = False
    rf_reg = False
    if model_name == 'deep4':
        deep4 = True
    elif model_name == 'resnet':
        res_net = True
    elif model_name == 'eegnet':
        eeg_net_v4 = True
    elif model_name == 'lin_reg':
        lin_reg = True
    elif model_name == 'lin_svr':
        lin_svr = True
    elif model_name == 'rbf_svr':
        rbf_svr = True
    elif model_name == 'rf_reg':
        rf_reg = True
    else:
        print('Wrong model_name {}. model_name can be deep4, resnet, eegnet, lin_reg, lin_svr, rbf_svr, rf_reg.'.format(model_name))
        return

    # which EEG frequency band
    band_pass = band_pass

    # which EEG electrodes
    electrodes = electrodes

    #storage
    dir_output_data = result_folder
    if not os.path.exists(dir_output_data):
        os.makedirs(dir_output_data)

    if data == 'onlyRobotData':
        only_robot_data = True
        only_eeg_data = False
        robot_eeg = False
        robot_eeg_aux = False
        only_aux = False
        robot_aux = False
    elif data == 'onlyEEGData':
        only_robot_data = False
        only_eeg_data = True
        robot_eeg = False
        robot_eeg_aux = False
        only_aux = False
        robot_aux = False
    elif data == 'onlyAux':
        only_robot_data = False
        only_eeg_data = False
        robot_eeg = False
        robot_eeg_aux = False
        only_aux = True
        robot_aux = False
    elif data == 'RobotEEGAux':
        only_robot_data = False
        only_eeg_data = False
        robot_eeg = False
        robot_eeg_aux = True
        only_aux = False
        robot_aux = False
    elif data == 'RobotEEG':
        only_robot_data = False
        only_eeg_data = False
        robot_eeg = True
        robot_eeg_aux = False
        only_aux = False
        robot_aux = False
    else:
        print('Wrong data type {}. data can be onlyRobotData, onlyEEGData, onlyAux, RobotEEGAux, RobotEEG.'.format(data))
        return

    if only_robot_data:
        save_addon_text_tmp = save_addon_text_orig  + '_onlyRobotData'
    elif only_eeg_data:
        save_addon_text_tmp = save_addon_text_orig  + '_onlyEEGData'
    elif robot_eeg:
        save_addon_text_tmp = save_addon_text_orig  + '_RobotEEG'
    elif robot_eeg_aux:
        save_addon_text_tmp = save_addon_text_orig  + '_RobotEEGAux'
    elif only_aux:
        save_addon_text_tmp = save_addon_text_orig  + '_onlyAuxData'
    elif robot_aux:
        save_addon_text_tmp = save_addon_text_orig  + '_RobotAux'

    for adam in [False]: # [True, False]:

        if adam:
            save_addon_text = save_addon_text_tmp + '_adam'
        else:
            save_addon_text = save_addon_text_tmp + '_adamW'

        if deep4:
            save_addon_text = save_addon_text + '_Deep4Net_stride' + str(pool_time_stride)
            if pool_time_stride is 2:
                time_window_duration = 1825 #ms
            elif pool_time_stride is 3:
                time_window_duration = 3661  # ms
            else:
                print('Pooling of time with stride {:d} not implemented'.format(pool_time_stride))
                return
            # deep4 stride 2: 1825
            # deep4 stride 3: 3661
        elif res_net:
            time_window_duration = 1005 #ms
            save_addon_text = save_addon_text + '_ResNet'
        elif eeg_net_v4:
            time_window_duration = 1335
            save_addon_text = save_addon_text + '_EEGNetv4'
        else:
            save_addon_text = save_addon_text + '_' + model_name

        # Add unique id of experiment to filenames
        save_addon_text = save_addon_text + '_' + unique_id

        train_set = []
        valid_set = []
        test_set = []
        for subjName in subjects:

            train_filename = data_folder + '/BBCIformat/' + subjName + '_' + str(sampling_rate) + 'Hz_CAR.BBCI.mat'

            sensor_names = BBCIDataset.get_all_sensors(train_filename, pattern=None)
            sensor_names_aux = ['ECG', 'Respiration']
            sensor_names_robot = ['robotHandPos_x', 'robotHandPos_y', 'robotHandPos_z']
            sensor_names_robot_aux = ['robotHandPos_x', 'robotHandPos_y', 'robotHandPos_z', 'ECG', 'Respiration']

            print('Loading data...')
            if only_robot_data:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names_robot).load() # robot pos channels
            elif only_eeg_data:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names).load() # all channels
                cnt = cnt.drop_channels(sensor_names_robot_aux)
            elif robot_eeg:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names).load() # all channels
                cnt = cnt.drop_channels(sensor_names_aux)
            elif robot_eeg_aux:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names).load() # all channels
            elif only_aux:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names_aux).load() # aux channels
            elif robot_aux:
                cnt = BBCIDataset(train_filename, load_sensor_names=sensor_names_robot_aux).load() # robot pos and aux channels

            # load score
            score_filename = data_folder + '/BBCIformat/' + subjName + '_score.mat'
            score_tmp = scipy.io.loadmat(score_filename)
            score = score_tmp['score_resample']

            # Remove stimulus channel
            cnt = cnt.drop_channels(['STI 014'])

            # resample if wrong sf
            resample_to_hz = sampling_rate
            cnt = resample_cnt(cnt, resample_to_hz)

            # Keep only selected EEG channels and frequency band
            if only_eeg_data or robot_eeg or robot_eeg_aux:
                print('Keeping only EEG channels matching patter {:s}'.format(electrodes))
                cnt.pick_channels(fnmatch.filter(cnt.ch_names, electrodes) + sensor_names_aux + sensor_names_robot)
                print('Band-passing data from {:s} to {:s} Hz'.format(str(band_pass[0]), str(band_pass[1])))
                if electrodes == '*':
                    cnt.filter(band_pass[0], band_pass[1], picks=range(32)) # This is somewhat dangerous but the first
                # 32 channels should always be EEG channels in the selected data configs. Unfortunately it does not look
                # like the types of the channels have been set properly to allow selecting using picks='eeg'
                elif electrodes == '*C*':
                        cnt.filter(band_pass[0], band_pass[1], picks=range(13))
                elif electrodes == '*z':
                        cnt.filter(band_pass[0], band_pass[1], picks=range(8))
                else:
                    print('Unsupported electrode selection {:s}. Electrode selection can be * or *C* or *z'.format(
                        electrodes))
                    return


            # mne apply will apply the function to the data (a 2d-numpy-array)
            # have to transpose data back and forth, since
            # exponential_running_standardize expects time x chans order
            # while mne object has chans x time order
            cnt = mne_apply(lambda a: exponential_running_standardize(
            a, init_block_size=1000,factor_new=0.001, eps=1e-4),
            cnt)

            name_to_start_codes = OrderedDict([('ScoreExp', 1)])
            name_to_stop_codes = OrderedDict([('ScoreExp', 2)])

            print('Splitting data...')
            train_set.append(create_signal_target_from_raw_mne(cnt, name_to_start_codes, [0,0], name_to_stop_codes))

            train_set[-1].y = score[:,:-1]
            # split data and test set
            cut_ind_test = int(np.size(train_set[-1].y) - n_seconds_test_set*sampling_rate) # use last nSecondsTestSet as test set
            test_set.append(deepcopy(train_set[-1]))
            test_set[-1].X[0] = np.array(np.float32(test_set[-1].X[0][:, cut_ind_test:]))
            test_set[-1].y = np.float32(test_set[-1].y[:,cut_ind_test:])

            if n_seconds_valid_set > 0:
                cut_ind_valid = int(np.size(train_set[-1].y) - (n_seconds_valid_set+n_seconds_test_set)*sampling_rate) # use last nSecondsTestSet as test set
                valid_set.append(deepcopy(train_set[-1]))
                valid_set[-1].X[0] = np.array(np.float32(valid_set[-1].X[0][:, cut_ind_valid:cut_ind_test]))
                valid_set[-1].y = np.float32(valid_set[-1].y[:,cut_ind_valid:cut_ind_test])
            elif n_seconds_valid_set == 0:
                cut_ind_valid = cut_ind_test
                valid_set.append(None)
            else:
                print('Negative validation set seconds not supported!')
                return

            train_set[-1].X[0] = np.array(np.float32(train_set[-1].X[0][:, :cut_ind_valid]))
            train_set[-1].y = np.float32(train_set[-1].y[:,:cut_ind_valid])

            # Normalize targets
            # train_set_y_mean = np.mean(train_set[-1].y)
            # train_set_y_std = np.std(train_set[-1].y)
            # train_set[-1].y = (train_set[-1].y - train_set_y_mean) / train_set_y_std
            # valid_set[-1].y = (valid_set[-1].y - train_set_y_mean) / train_set_y_std  # Use only training data
            # test_set[-1].y = (test_set[-1].y - train_set_y_mean) / train_set_y_std  # Use only training data

        for i_subject in range(len(subjects)):
            
            subjName = subjects[i_subject]

            # Check if experiment has already been run. If so go to next subject
            model_base_name = dir_output_data + '/' + subjName + save_addon_text
            result_files = glob.glob(model_base_name + '_' + subjName + '.csv')
            if result_files:
                print(subjName + save_addon_text + ' has already been run. Trying next subject.')
                continue

            if model_name in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
                # insert traditional ML here
                if lin_reg:
                    regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
                elif lin_svr:
                    regr = LinearSVR(verbose=3, random_state=20170629)#, max_iter=1000)# epsilon=0.0, tol=0.0001, C=1.0, loss=’epsilon_insensitive’, fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
                elif rbf_svr:
                    regr = SVR(verbose=3, max_iter=100000)#kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
                elif rf_reg:
                    regr = RandomForestRegressor(n_estimators=100, n_jobs=16, verbose=3, random_state=20170629)# (n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)

                # Train the model using the training sets
                print('Training traditional regression model...')
                regr.fit(train_set[i_subject].X[0].T, train_set[i_subject].y.T.squeeze())

                # Save model
                joblib.dump(regr, model_base_name + '_model.pkl.z')  # z to compress using zlib. Compression should be
                # lossless https://zlib.net/


                for i_eval_subject in range(len(subjects)):
                    # Make predictions using the training set
                    print('Testing on train data...')
                    train_set_pred = regr.predict(train_set[i_eval_subject].X[0].T)
                    print('Saving training set predictions...')
                    joblib.dump([train_set[i_eval_subject].y.T, train_set_pred], model_base_name + '_' + subjects[i_eval_subject] + '_train_preds.plk.z')

                    # Metrics
                    mse_train = mean_squared_error(train_set[i_eval_subject].y.T, train_set_pred)
                    var_score_train = r2_score(train_set[i_eval_subject].y.T, train_set_pred)
                    corrcoef_train, pval_train = pearsonr(train_set_pred, train_set[i_eval_subject].y.T[:, 0])
                    print("Train {:s}: MSE {:.4f}, var_score {:.4f}, corr {:.4f}, p-value {:.4f}".format(model_name, mse_train,
                                                                                                         var_score_train,
                                                                                                         corrcoef_train,
                                                                                                         pval_train))

                    # Make predictions using the validation set
                    if n_seconds_valid_set > 0:
                        print('Testing on validation data...')
                        valid_set_pred = regr.predict(valid_set[i_eval_subject].X[0].T)
                        print('Saving validation set predictions...')
                        joblib.dump([valid_set[i_eval_subject].y.T, valid_set_pred], model_base_name + '_' + subjects[i_eval_subject] + '_valid_preds.pkl.z')

                        # Metrics
                        mse_valid = mean_squared_error(valid_set[i_eval_subject].y.T, valid_set_pred)
                        var_score_valid = r2_score(valid_set[i_eval_subject].y.T, valid_set_pred)
                        corrcoef_valid, pval_valid = pearsonr(valid_set_pred, valid_set[i_eval_subject].y.T[:, 0])
                        print("Validation {:s}: MSE {:.4f}, var_score {:.4f}, corr {:.4f}, p-value {:.4f}".format(model_name,
                                                                                                                  mse_valid,
                                                                                                                  var_score_valid,
                                                                                                                  corrcoef_valid,
                                                                                                                  pval_valid))
                    else:
                        mse_valid = np.nan
                        var_score_valid = np.nan
                        corrcoef_valid = np.nan
                        pval_valid = np.nan

                    # Make predictions using the test set
                    print('Testing on test data...')
                    test_set_pred = regr.predict(test_set[i_eval_subject].X[0].T)
                    print('Saving test set predictions...')
                    joblib.dump([test_set[i_eval_subject].y.T, test_set_pred], model_base_name + '_' + subjects[i_eval_subject] + '_test_preds.pkl.z')

                    # Metrics
                    mse_test = mean_squared_error(test_set[i_eval_subject].y.T, test_set_pred)
                    var_score_test = r2_score(test_set[i_eval_subject].y.T, test_set_pred)
                    corrcoef_test, pval_test = pearsonr(test_set_pred, test_set[i_eval_subject].y.T[:, 0])
                    print(
                        "Test {:s}: MSE {:.4f}, var_score {:.4f}, corr {:.4f}, p-value {:.4f}".format(model_name, mse_test,
                                                                                                      var_score_test,
                                                                                                      corrcoef_test,
                                                                                                      pval_test))

                    # Save metrics
                    print('Saving results...')
                    result_df = pd.Series({'Train mse': mse_train, 'Train corr': corrcoef_train, 'Train corr p':
                                              pval_train, 'Train explained variance': var_score_train,
                                              'Validation mse': mse_valid, 'Validation corr': corrcoef_valid, 'Validation corr p':
                                              pval_valid, 'Validation explained variance': var_score_valid,
                                              'Test mse': mse_test, 'Test corr': corrcoef_test, 'Test corr p':
                                              pval_test, 'Test explained variance': var_score_test,
                                              }).to_frame(subjName + save_addon_text + '_' + subjects[i_eval_subject])
                    result_df.to_csv(model_base_name + '_' + subjects[i_eval_subject] + '.csv', sep=',', header=True)
                    # Explained variance score: 1 is perfect prediction

                    # Plot outputs
                    print('Plotting...')
                    plt.rcParams.update({'font.size': 24})
                    plt.figure(figsize=(32, 12))
                    t = np.arange(train_set_pred.shape[0]) / sampling_rate
                    plt.plot(t, train_set_pred)
                    plt.plot(t, train_set[i_subject].y.T)
                    plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                    plt.title(
                        'Train {:s}: mse = {:f}, r = {:f}, p = {:f}'.format(model_name, mse_train, corrcoef_train,
                                                                             pval_train))
                    plt.xlabel('time (s)')
                    plt.ylabel('subjective rating')
                    plt.ylim(-1, 1)
                    plt.xlim(0, int(np.round(train_set_pred.shape[0] / sampling_rate)))
                    # plt.show()
                    plt.savefig(model_base_name + '_' + subjects[i_eval_subject] + '_fig_pred_train.png',
                                bbox_inches='tight', dpi=300)

                    if n_seconds_valid_set > 0:
                        # Explained variance score: 1 is perfect prediction

                        # Plot outputs
                        plt.rcParams.update({'font.size': 24})
                        plt.figure(figsize=(32, 12))
                        t = np.arange(valid_set_pred.shape[0])/sampling_rate
                        plt.plot(t, valid_set_pred)
                        plt.plot(t, valid_set[i_subject].y.T)
                        plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                        plt.title('{:g}s validation {:s}: mse = {:f}, r = {:f}, p = {:f}'.format(n_seconds_valid_set, model_name, mse_valid, corrcoef_valid, pval_valid))
                        plt.xlabel('time (s)')
                        plt.ylabel('subjective rating')
                        plt.ylim(-1, 1)
                        plt.xlim(0,  int(np.round(valid_set_pred.shape[0]/sampling_rate)))
                        # plt.show()
                        plt.savefig(model_base_name + '_' + subjects[i_eval_subject] + '_fig_pred_valid.png',
                                    bbox_inches='tight', dpi=300)

                    if n_seconds_test_set > 0:
                        # Explained variance score: 1 is perfect prediction

                        # Plot outputs
                        plt.rcParams.update({'font.size': 24})
                        plt.figure(figsize=(32, 12))
                        t = np.arange(test_set_pred.shape[0]) / sampling_rate
                        plt.plot(t, test_set_pred)
                        plt.plot(t, test_set[i_subject].y.T)
                        plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                        plt.title('{:g}s test {:s}: mse = {:f}, r = {:f}, p = {:f}'.format(n_seconds_test_set, model_name, mse_test,
                                                                                       corrcoef_test, pval_test))
                        plt.xlabel('time (s)')
                        plt.ylabel('subjective rating')
                        plt.ylim(-1, 1)
                        plt.xlim(0, int(np.round(test_set_pred.shape[0] / sampling_rate)))
                        # plt.show()
                        plt.savefig(model_base_name + '_' + subjects[i_eval_subject] + '_fig_pred_test.png',
                                    bbox_inches='tight', dpi=300)
                    # ADD  DISTANCE AND SPEED TO BBCI FILE!!!

            elif model_name in ['deep4', 'resnet', 'eegnet']:
                set_random_seeds(seed=20170629, cuda=cuda)
                torch.cuda.set_device(gpu_index)

                # This will determine how many crops are processed in parallel
                input_time_length = int(time_window_duration/1000*sampling_rate) # train_set[i_subject].X.shape[1]
                in_chans=train_set[i_subject].X[0].shape[0]

                if deep4:
                    # final_conv_length determines the size of the receptive field of the ConvNet
                    model = Deep4Net(in_chans=in_chans, n_classes=1, input_time_length=input_time_length,
                                     pool_time_stride=pool_time_stride,
                                     final_conv_length=2, stride_before_pool=True).create_network()
                elif res_net:
                    model_name = 'resnet-xavier-uniform'
                    init_name = model_name.lstrip('resnet-')
                    init_fn = {'he-uniform': lambda w: init.kaiming_uniform(w, a=0),
                               'he-normal': lambda w: init.kaiming_normal(w, a=0),
                               'xavier-uniform': lambda w: init.xavier_uniform(w, gain=1),
                               'xavier-normal': lambda w: init.xavier_normal(w, gain=1)}[init_name]

                    model = EEGResNet(in_chans=in_chans, n_classes=1, input_time_length=input_time_length,
                                      final_pool_length=2, n_first_filters=48,
                                      conv_weight_init_fn=init_fn).create_network()
                elif eeg_net_v4:
                    model = EEGNetv4(in_chans=in_chans, n_classes=1, final_conv_length=2, input_time_length=input_time_length).create_network()

                # remove softmax
                new_model = nn.Sequential()
                for name, module in model.named_children():
                    if name == 'softmax':
                        continue
                    new_model.add_module(name, module)


                # lets remove empty final dimension
                def squeeze_out(x):
                    # Remove single "class" dimension
                    assert x.size()[1] == 1
                    return x[:, 0]


                new_model.add_module('squeeze_again', Expression(squeeze_out))
                model = new_model


                if cuda:
                    model.cuda()

                if not res_net:
                    to_dense_prediction_model(model)


                start_param_values = deepcopy(new_model.state_dict())

                # %% # determine output size
                test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
                if cuda:
                    test_input = test_input.cuda()
                out = model(test_input)
                n_preds_per_input = out.cpu().data.numpy().shape[1]
                log.info("predictor length = {:d} samples".format(n_preds_per_input))
                log.info("predictor length = {:f} s".format(n_preds_per_input/sampling_rate))


                iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,n_preds_per_input=n_preds_per_input)

                 # %% Loss function takes predictions as they come out of the network and the targets and returns a loss
                loss_function = F.mse_loss

                # Could be used to apply some constraint on the models, then should be object with apply method that accepts a module
                model_constraint = None

                # %% Monitors log the training progress
                monitors = [LossMonitor(),
                        CorrelationMonitor1d(input_time_length),
                        RuntimeMonitor(), ]

                # %% Stop criterion determines when the first stop happens
                stop_criterion = MaxEpochs(max_train_epochs)


                 # %% re-initialize model
                model.load_state_dict(deepcopy(start_param_values))

                if adam:
                    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay,
                                               lr=init_lr)
                else:
                    weight_decay = np.float32(weight_decay)
                    init_lr = np.float32(init_lr)
                    # np_th_seed = 321312
                    scheduler_name = 'cosine'
                    schedule_weight_decay = True

                    optimizer_name = 'adamw'
                    if optimizer_name == 'adam':
                        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay,
                                               lr=init_lr)
                    elif optimizer_name == 'adamw':
                        optimizer = AdamW(model.parameters(), weight_decay=weight_decay,
                                          lr=init_lr)


                    restarts = None
                    if scheduler_name is not None:
                        assert schedule_weight_decay == (optimizer_name == 'adamw')
                        if scheduler_name == 'cosine':
                            n_updates_per_epoch = sum(
                                [1 for _ in iterator.get_batches(train_set[i_subject], shuffle=True)])
                            if restarts is None:
                                n_updates_per_period = n_updates_per_epoch * max_train_epochs
                            else:
                                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
                            scheduler = CosineAnnealing(n_updates_per_period)
                            optimizer = ScheduledOptimizer(scheduler, optimizer,
                                                           schedule_weight_decay=schedule_weight_decay)
                        elif scheduler_name == 'cut_cosine':
                            # TODO: integrate with if clause before, now just separate
                            # to avoid messing with code
                            n_updates_per_epoch = sum(
                                [1 for _ in iterator.get_batches(train_set[i_subject], shuffle=True)])
                            if restarts is None:
                                n_updates_per_period = n_updates_per_epoch * max_train_epochs
                            else:
                                n_updates_per_period = np.array(restarts) * n_updates_per_epoch
                            scheduler = CutCosineAnnealing(n_updates_per_period)
                            optimizer = ScheduledOptimizer(scheduler, optimizer,
                                                           schedule_weight_decay=schedule_weight_decay)
                        else:
                            raise ValueError("Unknown scheduler")

                # set up experiment, run
                exp = Experiment(model, train_set[i_subject], valid_set[i_subject], test_set[i_subject], iterator,
                                 loss_function, optimizer, model_constraint,
                                 monitors, stop_criterion,
                                 remember_best_column=None, do_early_stop=False,
                                 run_after_early_stop=False, batch_modifier=None, cuda=cuda)
                exp.run()

                # %% save values: CC, pred, resp
                exp.model_base_name = model_base_name
                if (len(exp.epochs_df) != max_train_epochs+1):
                    print('WARNING: the epoch dataframe has too few epochs: {:d}'.format(len(exp.epochs_df)))

                print('Saving epoch dataframe...')
                exp.epochs_df.to_csv(exp.model_base_name + '.csv', sep=',', header=True)

                # %% Save model
                print('Saving model...')
                save_params(exp)

                # %% plot learning curves

                print('Plotting & saving...')
                f, axarr = plt.subplots(2, figsize=(15,15))
                exp.epochs_df.loc[:,['train_loss','valid_loss','test_loss']].plot(ax=axarr[0], title='loss function',  logy=True)
                exp.epochs_df.loc[:,['train_corr','valid_corr','test_corr']].plot(ax=axarr[1], title='correlation')
                plt.savefig(exp.model_base_name + '_fig_lc.png', bbox_inches='tight')

                for i_eval_subject in range(len(subjects)):

                    # %% evaluation on train set
                    all_preds = []
                    all_targets = []
                    dataset = train_set[i_eval_subject]
                    for batch in exp.iterator.get_batches(dataset, shuffle=False):
                        preds, loss = exp.eval_on_batch(batch[0], batch[1])
                        all_preds.append(preds)
                        all_targets.append(batch[1])

                    preds_2d = [p[:, None] for p in all_preds]
                    preds_per_trial = compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
                    ys_2d = [y[:, None] for y in all_targets]
                    targets_per_trial = compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
                    assert preds_per_trial.shape == targets_per_trial.shape

                    # corrcoefs = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
                    (corrcoefs, pval) = pearsonr(targets_per_trial, preds_per_trial)
                    mse = mean_squared_error(targets_per_trial, preds_per_trial)
                    print('Saving training set predictions...')
                    joblib.dump([targets_per_trial, preds_per_trial], exp.model_base_name + subjects[i_eval_subject] + '_train_preds.pkl.z')

                    # %% plot predicted rating
                    plt.rcParams.update({'font.size': 24})
                    plt.figure(figsize=(32, 12))
                    t = np.arange(preds_per_trial.shape[0]) / sampling_rate
                    plt.plot(t, preds_per_trial)
                    plt.plot(t, targets_per_trial)
                    plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                    plt.title('Train set: mse = {:f}, r = {:f}, p = {:f}'.format(mse, corrcoefs, pval))
                    plt.xlabel('time (s)')
                    plt.ylabel('subjective rating')
                    plt.ylim(-1, 1)
                    plt.xlim(0, int(np.round(preds_per_trial.shape[0] / sampling_rate)))
                    plt.savefig(exp.model_base_name + subjects[i_eval_subject] + '_fig_pred_train.png',
                                bbox_inches='tight', dpi=300)

                    # %% evaluation on validation set
                    if n_seconds_valid_set > 0:
                        all_preds = []
                        all_targets = []
                        dataset = valid_set[i_eval_subject]
                        for batch in exp.iterator.get_batches(dataset, shuffle=False):
                            preds, loss = exp.eval_on_batch(batch[0], batch[1])
                            all_preds.append(preds)
                            all_targets.append(batch[1])

                        preds_2d = [p[:, None] for p in all_preds]
                        preds_per_trial = compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
                        ys_2d = [y[:, None] for y in all_targets]
                        targets_per_trial = compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
                        assert preds_per_trial.shape == targets_per_trial.shape

                        # corrcoefs = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
                        (corrcoefs, pval) = pearsonr(targets_per_trial, preds_per_trial)
                        mse = mean_squared_error(targets_per_trial, preds_per_trial)
                        print('Saving validation set predictions...')
                        joblib.dump([targets_per_trial, preds_per_trial], exp.model_base_name + subjects[i_eval_subject] + '_valid_preds.pkl.z')

                        # %% plot predicted rating
                        plt.rcParams.update({'font.size': 24})
                        plt.figure(figsize=(32, 12))
                        t = np.arange(preds_per_trial.shape[0]) / sampling_rate
                        plt.plot(t, preds_per_trial)
                        plt.plot(t, targets_per_trial)
                        plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                        plt.title('Validation set: mse = {:f}, r = {:f}, p = {:f}'.format(mse, corrcoefs, pval))
                        plt.xlabel('time (s)')
                        plt.ylabel('subjective rating')
                        plt.ylim(-1, 1)
                        plt.xlim(0, int(np.round(preds_per_trial.shape[0] / sampling_rate)))
                        plt.savefig(exp.model_base_name + subjects[i_eval_subject] + '_fig_pred_valid.png',
                                    bbox_inches='tight', dpi=300)

                    # %% evaluation on test set
                    if n_seconds_test_set > 0:
                        all_preds = []
                        all_targets = []
                        dataset = test_set[i_eval_subject]
                        for batch in exp.iterator.get_batches(dataset, shuffle=False):
                            preds, loss = exp.eval_on_batch(batch[0], batch[1])
                            all_preds.append(preds)
                            all_targets.append(batch[1])

                        preds_2d = [p[:, None] for p in all_preds]
                        preds_per_trial = compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
                        ys_2d = [y[:, None] for y in all_targets]
                        targets_per_trial = compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
                        assert preds_per_trial.shape == targets_per_trial.shape

                        #corrcoefs = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
                        (corrcoefs, pval) = pearsonr(targets_per_trial, preds_per_trial)
                        mse = mean_squared_error(targets_per_trial, preds_per_trial)
                        print('Saving test set predictions...')
                        joblib.dump([targets_per_trial, preds_per_trial], exp.model_base_name + subjects[i_eval_subject] + '_test_preds.pkl.z')

                        # %% plot predicted rating
                        plt.rcParams.update({'font.size': 24})
                        plt.figure(figsize=(32, 12))
                        t = np.arange(preds_per_trial.shape[0])/sampling_rate
                        plt.plot(t, preds_per_trial)
                        plt.plot(t, targets_per_trial)
                        plt.legend(('Predicted', 'Actual'), fontsize=24, loc='best')
                        plt.title('Test set: mse = {:f}, r = {:f}, p = {:f}'.format(mse, corrcoefs, pval))
                        plt.xlabel('time (s)')
                        plt.ylabel('subjective rating')
                        plt.ylim(-1, 1)
                        plt.xlim(0,  int(np.round(preds_per_trial.shape[0]/sampling_rate)))
                        plt.savefig(exp.model_base_name + subjects[i_eval_subject] + '_fig_pred_test.png', bbox_inches='tight', dpi=300)
                        log.info("-----------------------------------------")

                    plt.close('all')

                if calc_viz:
                    # %% visualization (Kay): Wrap Model into SelectiveSequential and set up pred_fn
                    assert(len(list(model.children()))==len(list(model.named_children()))) # All modules gotta have names!

                    modules = list(model.named_children()) # Extract modules from model
                    select_modules = ['conv_spat','conv_2','conv_3','conv_4'] # Specify intermediate outputs

                    model_pert = SelectiveSequential(select_modules,modules) # Wrap modules
                    # Prediction function that is used in phase_perturbation_correlation
                    model_pert.eval();
                    pred_fn = lambda x: [layer_out.data.numpy() for
                                         layer_out in model_pert.forward(torch.autograd.Variable(torch.from_numpy(x)).float())]

                    # Gotta change pred_fn a bit for cuda case
                    if cuda:
                        model_pert.cuda()
                        pred_fn = lambda x: [layer_out.data.cpu().numpy() for
                                             layer_out in model_pert.forward(torch.autograd.Variable(torch.from_numpy(x)).float().cuda())]

                    perm_X = np.expand_dims(train_set[i_subject].X,3) # Input gotta have dimension BxCxTx1

                    # %% visualization (Kay): Run phase and amplitude perturbations
                    log.info("visualization: perturbation computation ...")
                    phase_pert_corrs = perturbation_correlation(phase_perturbation, correlate_feature_maps, pred_fn,4,perm_X,n_perturbations,batch_size=2000)
                    amp_pert_corrs = perturbation_correlation(amp_perturbation_multiplicative, mean_diff_feature_maps, pred_fn,4,perm_X,n_perturbations,batch_size=2000)

                    # %% save perturbation over layers
                    freqs = np.fft.rfftfreq(perm_X.shape[2],d=1/250.)
                    for l in range(len(phase_pert_corrs)):
                        layer_cc = phase_pert_corrs[l]
                        scipy.io.savemat(exp.model_base_name + '_phiPrtCC' + '_layer{:d}'.format(l) + '.mat', {'layer_cc':layer_cc})
                        layer_cc = amp_pert_corrs[l]
                        scipy.io.savemat(exp.model_base_name + '_ampPrtCC' + '_layer{:d}'.format(l) + '.mat', {'layer_cc':layer_cc})

                print('Deleting model and experiment...')
                del exp, model

def main():
    logging.basicConfig(level=logging.DEBUG)
    kwargs = parse_run_args()
    print(kwargs)
    start_time = time.time()
    run_experiment(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info("Experiment runtime: {:.2f} sec".format(run_time))
    #
    # write_kwargs_and_epochs_dfs(kwargs, exp)
    # make_final_predictions(kwargs, exp)
    # save_model(kwargs, exp)




if __name__ == '__main__':
    main()
