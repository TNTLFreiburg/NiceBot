import numpy as np
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.util import wrap_reshape_apply_fn, corr
from numpy.random.mtrand import RandomState
from torch import nn


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
    visualizations.
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

# %% visualizations (Kay): Phase and Amplitude perturbation
from deepRegressionCode.DeepRegression_kisbat import log


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
