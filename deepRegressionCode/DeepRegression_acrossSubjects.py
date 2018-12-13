
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

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
os.sys.path.append('/home/martin/braindecode/code/adamw-eeg-eval/')
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
from torch.nn.functional import elu
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

from braindecode.torch_ext.modules import Expression
import scipy.io
from scipy.stats import pearsonr

from copy import deepcopy

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


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

########################################################################################################




# Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True


Subjects =['noExp', 'moderateExp', 'substantialExp'];


nSecondsTestSet = 180



saveAddonText_orig = '_3minTest'
batch_size = 64
pool_time_stride = 3

samplingRate = 256

maxTrainEpochs = 200


#viz
calcViz = False
N_perturbations = 20


# which network
Deep4 = True
ResNet = False
EEGNet_v4 = False

#storage
dir_outputData = './outputData'

# optimizer
adam = True


onlyRobotData = True
onlyEEGData = False
RobotEEG = False
RobotEEGAux = False
onlyAux = False
RobotAux = False


if onlyRobotData:
    saveAddonText_tmp = saveAddonText_orig  + '_onlyRobotData'
elif onlyEEGData:
    saveAddonText_tmp = saveAddonText_orig  + 'onlyEEGData'
elif RobotEEG:
    saveAddonText_tmp = saveAddonText_orig  + '_RobotEEG'
elif RobotEEGAux:
    saveAddonText_tmp = saveAddonText_orig  + '_RobotEEGAux'
elif onlyAux:
    saveAddonText_tmp = saveAddonText_orig  + '_onlyAuxData'
elif RobotAux:
    saveAddonText_tmp = saveAddonText_orig  + '_RobotAux'


if adam:
    saveAddonText = saveAddonText_tmp + '_adam'
else:
    saveAddonText = saveAddonText_tmp + '_adamW'


if Deep4:
    saveAddonText = saveAddonText + '_Deep4Net_stride3'
    timeWindowDuration = 3661 #ms
    # deep4 stride 2: 1825
    # deep4 stride 3: 3661
elif ResNet:
    timeWindowDuration = 1005 #ms
    saveAddonText = saveAddonText + '_ResNet'
elif EEGNet_v4:
    timeWindowDuration = 1335
    saveAddonText = saveAddonText + '_EEGNetv4'


saveAddonText_tmp = saveAddonText_orig  + '_onlyRobotData'


cnt = []
train_sets = []
test_sets = []

# load and process data
for iSubject, subjName in enumerate(Subjects): 



    train_filename = './data/BBCIformat/' + subjName +  '_' + str(samplingRate) + 'Hz_CAR.BBCI.mat'


    sensor_names = BBCIDataset.get_all_sensors(train_filename, pattern=None)
    sensor_names_aux = ['ECG', 'Respiration']
    sensor_names_robot = ['robotHandPos_x','robotHandPos_y','robotHandPos_z']
    sensor_names_robot_aux = ['robotHandPos_x','robotHandPos_y','robotHandPos_z','ECG', 'Respiration'] 

    if onlyRobotData:
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names_robot).load()) # robot pos channels
    elif onlyEEGData:
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names).load()) # all channels
        cnt[iSubject] = cnt[iSubject].drop_channels(sensor_names_robot_aux)
    elif RobotEEG:        
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names).load()) # all channels
        cnt[iSubject] = cnt[iSubject].drop_channels(sensor_names_aux)
    elif RobotEEGAux:
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names).load()) # all channels
    elif onlyAux:
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names_aux).load()) # aux channels
    elif RobotAux:
        cnt.append(BBCIDataset(train_filename, load_sensor_names=sensor_names_robot_aux).load()) # robot pos and aux channels


    
    #load score
    score_filename = './data/BBCIformat/' + subjName + '_score.mat'
    Score_tmp = scipy.io.loadmat(score_filename)
    Score = Score_tmp['score_resample']

    # Remove stimulus channel
    cnt[iSubject] = cnt[iSubject].drop_channels(['STI 014'])

    # for now, remove also robot,ecg and respiration channels
    #cnt[iSubject] = cnt[iSubject].drop_channels(sensor_names_robot_aux)


    # resample if wrong sf
    resampleToHz = samplingRate
    cnt[iSubject] = resample_cnt(cnt[iSubject], resampleToHz)

    # mne apply will apply the function to the data (a 2d-numpy-array)
    # have to transpose data back and forth, since
    # exponential_running_standardize expects time x chans order
    # while mne object has chans x time order
    cnt[iSubject] = mne_apply(lambda a: exponential_running_standardize(
    a.T, init_block_size=1000,factor_new=0.001, eps=1e-4).T,
    cnt[iSubject])



    name_to_start_codes = OrderedDict([('ScoreExp', 1)])
    name_to_stop_codes = OrderedDict([('ScoreExp', 2)])


    train_sets.append(create_signal_target_from_raw_mne(cnt[iSubject], name_to_start_codes, [0,0], name_to_stop_codes))

    train_sets[iSubject].y = Score[:,:-1]


    cutInd = int( np.size(train_sets[iSubject].y) - nSecondsTestSet*samplingRate ) # use last nSecondsTestSet as test set

    test_sets.append(deepcopy(train_sets[iSubject]))
    test_sets[iSubject].X[0] = np.array(np.float32(test_sets[iSubject].X[0][:, cutInd:]))
    test_sets[iSubject].y = np.float32(test_sets[iSubject].y[:, cutInd:])

    train_sets[iSubject].X[0] = np.array(np.float32(train_sets[iSubject].X[0][:, :cutInd]))
    train_sets[iSubject].y = np.float32(train_sets[iSubject].y[:, :cutInd])

    valid_set = None




for iSubject, subjName in enumerate(Subjects): 


    train_set = deepcopy(train_sets[iSubject])


    testSubjects = list(range(np.size(Subjects)))
    del(testSubjects[iSubject])

    for iSubjectTest in testSubjects:


        txtToPrint = 'Training on ' + Subjects[iSubject] + ', testing on ' + Subjects[iSubjectTest]
        print(txtToPrint)
    

        #test_set = deepcopy(train_sets[iSubjectTest]) #whole session
        test_set = deepcopy(test_sets[iSubjectTest]) #3-min at the end

    

        set_random_seeds(seed=20170629, cuda=cuda)

        # This will determine how many crops are processed in parallel
        input_time_length = int(timeWindowDuration/1000*samplingRate) # train_set.X.shape[1]
        in_chans=train_set.X[0].shape[0]


        if Deep4:    
            # final_conv_length determines the size of the receptive field of the ConvNet
            model = Deep4Net(in_chans=in_chans, n_classes=1, input_time_length=input_time_length,
                         pool_time_stride=pool_time_stride,
                                final_conv_length=2, stride_before_pool=True).create_network()
        elif ResNet:
            model_name = 'resnet-xavier-uniform'
            init_name = model_name.lstrip('resnet-')
            from torch.nn import init
            init_fn = {'he-uniform': lambda w: init.kaiming_uniform(w, a=0),
            'he-normal': lambda w: init.kaiming_normal(w, a=0),
            'xavier-uniform': lambda w: init.xavier_uniform(w, gain=1),
            'xavier-normal': lambda w: init.xavier_normal(w, gain=1)}[init_name]

            model = EEGResNet(in_chans=in_chans, n_classes=1, input_time_length=input_time_length,
            final_pool_length=2, n_first_filters=48,
            conv_weight_init_fn=init_fn).create_network()
        elif EEGNet_v4:
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

        if not ResNet:
            to_dense_prediction_model(model)


        start_param_values = deepcopy(new_model.state_dict())

        # %% setup optimizer -> new for each x-val fold
        from torch import optim

        # %% # determine output size
        from braindecode.torch_ext.util import np_to_var    
        test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
        if cuda:
            test_input = test_input.cuda()
        out = model(test_input)
        n_preds_per_input = out.cpu().data.numpy().shape[1]
        log.info("predictor length = {:d} samples".format(n_preds_per_input))
        log.info("predictor length = {:f} s".format(n_preds_per_input/samplingRate))


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
        stop_criterion = MaxEpochs(maxTrainEpochs)



         # %% re-initialize model
        model.load_state_dict(deepcopy(start_param_values))

        if adam:
            optimizer = optim.Adam(model.parameters())
        else:
            weight_decay = np.float32(2.0 * 0.001)
            init_lr = np.float32((1/32.0) * 0.01)
            np_th_seed = 321312
            scheduler_name = 'cosine'
            schedule_weight_decay= True

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
                        [1 for _ in iterator.get_batches(train_set, shuffle=True)])
                    if restarts is None:
                        n_updates_per_period = n_updates_per_epoch * maxTrainEpochs
                    else:
                        n_updates_per_period = np.array(restarts) * n_updates_per_epoch
                    scheduler = CosineAnnealing(n_updates_per_period)
                    optimizer = ScheduledOptimizer(scheduler, optimizer,
                                                   schedule_weight_decay=schedule_weight_decay)
                elif scheduler_name == 'cut_cosine':
                    # TODO: integrate with if clause before, now just separate
                    # to avoid messing with code
                    n_updates_per_epoch = sum(
                        [1 for _ in iterator.get_batches(train_set, shuffle=True)])
                    if restarts is None:
                        n_updates_per_period = n_updates_per_epoch * maxTrainEpochs
                    else:
                        n_updates_per_period = np.array(restarts) * n_updates_per_epoch
                    scheduler = CutCosineAnnealing(n_updates_per_period)
                    optimizer = ScheduledOptimizer(scheduler, optimizer,
                                                   schedule_weight_decay=schedule_weight_decay)
                else:
                    raise ValueError("Unknown scheduler")

        # set up experiment, run
        exp = Experiment(model, train_set, valid_set, test_set, iterator,
                         loss_function, optimizer, model_constraint,
                         monitors, stop_criterion,
                         remember_best_column='train_loss', do_early_stop = False,
                         run_after_early_stop=False, batch_modifier=None, cuda=cuda)        
        exp.run()



        # %% plot learning curves

        f, axarr = plt.subplots(2, figsize=(15,15))
        exp.epochs_df.loc[:,['train_loss','valid_loss','test_loss']].plot(ax=axarr[0], title='loss function')
        exp.epochs_df.loc[:,['train_corr','valid_corr','test_corr']].plot(ax=axarr[1], title='correlation')        
        plt.savefig(dir_outputData + '/' + subjName + '_train_' + Subjects[iSubjectTest] + '_test' +  saveAddonText + '_fig_lc_fold.png', bbox_inches='tight')

        # %% evaluation
        all_preds = []
        all_targets = []
        dataset = test_set
        for batch in exp.iterator.get_batches(dataset, shuffle=False):
            preds, loss = exp.eval_on_batch(batch[0], batch[1])
            all_preds.append(preds)
            all_targets.append(batch[1])

        preds_2d = [p[:, None] for p in all_preds]
        preds_per_trial = compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
        ys_2d = [y[:, None] for y in all_targets]
        targets_per_trial = compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
        assert preds_per_trial.shape == targets_per_trial.shape

        # %% save values: CC, pred, resp
        exp.epochs_df.to_csv(dir_outputData + '/' + subjName + '_train_' + Subjects[iSubjectTest] + '_test' + saveAddonText , sep=',', header=False)
        #exp.epochs_df.to_excel(dir_outputData + '/' + fileName + '_epochs_fold{:d}.xls'.format(n))
        #corrcoefs = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
        (corrcoefs, pval) = pearsonr(preds_per_trial, targets_per_trial)
        pred_vals = preds_per_trial
        resp_vals = targets_per_trial

        # %% plot predicted trajectory
        plt.rcParams.update({'font.size': 24})
        plt.figure(figsize=(32, 12))
        t = np.arange(preds_per_trial.shape[0])/samplingRate
        plt.plot(t, preds_per_trial)
        plt.plot(t, targets_per_trial)
        #plt.legend(('Predicted', 'Actual'), fontsize=24)
        plt.title('r = {:f}, p = {:f}'.format(corrcoefs, pval))
        plt.xlabel('time (s)')
        plt.ylabel('subjective rating')
        plt.ylim(-1, 1)
        plt.xlim(0,  int(np.round(preds_per_trial.shape[0]/samplingRate)))
        plt.savefig(dir_outputData + '/' + subjName + '_train_' + Subjects[iSubjectTest] + '_test' +  saveAddonText + '_fig_predResp.png', bbox_inches='tight', dpi=300)   
        log.info("-----------------------------------------")

        plt.close('all')

        if calcViz:
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

            perm_X = np.expand_dims(train_set.X,3) # Input gotta have dimension BxCxTx1

            # %% visualization (Kay): Run phase and amplitude perturbations
            log.info("visualization: perturbation computation ...")
            phase_pert_corrs = perturbation_correlation(phase_perturbation, correlate_feature_maps, pred_fn,4,perm_X,N_perturbations,batch_size=2000)
            amp_pert_corrs = perturbation_correlation(amp_perturbation_multiplicative, mean_diff_feature_maps, pred_fn,4,perm_X,N_perturbations,batch_size=2000)

            # %% save perturbation over layers
            freqs = np.fft.rfftfreq(perm_X.shape[2],d=1/250.)
            for l in range(len(phase_pert_corrs)):
                layer_cc = phase_pert_corrs[l]
                scipy.io.savemat(dir_outputData + '/' + subjName + '_train_' + Subjects[iSubjectTest] + '_test' +   saveAddonText + '_phiPrtCC' + '_layer{:d}'.format(l) + '.mat', {'layer_cc':layer_cc})
                layer_cc = amp_pert_corrs[l]
                scipy.io.savemat(dir_outputData + '/' + subjName + '_train_' + Subjects[iSubjectTest] + '_test' +   saveAddonText + '_ampPrtCC' + '_layer{:d}'.format(l) + '.mat', {'layer_cc':layer_cc})
                    










