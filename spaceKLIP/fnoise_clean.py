import os
import numpy as np

from jwst.stpipe import Step
from jwst import datamodels

from webbpsf_ext import robust

class kTCSubtractStep(Step):
    """
    kTCSubtractStep: Subtract kTC noise from ramp data.
    """

    class_alias = "removektc"

    spec = """
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
    """

    def process(self, input):
        """Subtract kTC noise from data."""
        
        with datamodels.open(input) as input_model:
            datamodel = input_model.copy()
            bias_arr, _ = fit_slopes_to_ramp_data(datamodel, sat_frac=self.sat_frac)

            # Subtract bias from each integration
            nints = datamodel.meta.exposure.nints
            for i in range(nints):
                datamodel.data[i] -= bias_arr[i]

        return datamodel


class OneOverfStep(Step):
    """
    OneOverfStep: Apply 1/f noise correction to group-level.
    """

    class_alias = "1overf"

    spec = """
        model_type = option('median', 'mean', 'savgol', default='savgol') # Type of model to fit
        horizontal_corr = boolean(default=True) # Apply horizontal correction
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
    """

    def process(self, input):
        """Apply 1/f noise correction to data.

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be linearized.

        Returns
        =======
        output : JWST data model
            Output science data model with 1/f noise correction applied.
        """

        # Get the input data model
        with datamodels.open(input) as input_model:

            is_full_frame = 'FULL' in input_model.meta.subarray.name.upper()
            nints    = input_model.meta.exposure.nints
            ngroups  = input_model.meta.exposure.ngroups
            noutputs = input_model.meta.exposure.noutputs

            ny, nx = input_model.data.shape[-2:]
            chsize = ny // noutputs

            # Fit slopes to get signal mask
            # Grab slopes and bias values for each integration
            bias_arr, slope_arr = fit_slopes_to_ramp_data(input_model, sat_frac=self.sat_frac)
            # Remove bias from slopes and calculate mean slope
            for i in range(nints):
                slope_arr[i] -= bias_arr[i]
            slope_mean = robust.mean(slope_arr, axis=0)

            # Generate a mean signal ramp to subtract from each group
            group_time = input_model.meta.exposure.group_time
            ngroups = input_model.meta.exposure.ngroups
            tarr = np.arange(1, ngroups+1) * group_time
            signal_mean_ramp = slope_mean * tarr.reshape([-1,1,1])

            # Subtract 1/f noise from each integration
            datamodel = input_model.copy()
            data = datamodel.data
            for i in range(nints):
                cube = data[i] - bias_arr[i]
                groupdq = datamodel.groupdq[i]
                # Cumulative sum of group DQ flags
                bpmask_arr = np.cumsum(groupdq, axis=0) > 0
                for j in range(ngroups):

                    # Exclude bad pixels
                    im_mask = ~bpmask_arr[j] # Good pixel mask
                    # Clean channel by channel
                    for ch in range(noutputs):
                        # Get channel x-indices
                        x1 = int(ch*chsize)
                        x2 = int(x1 + chsize)

                        # Channel subarrays
                        imch = cube[j, :, x1:x2]
                        sigch = signal_mean_ramp[j, :, x1:x2]
                        good_mask = im_mask[:, x1:x2]

                        # Remove averaged signal goup
                        imch_diff = imch - sigch

                        # Subtract 1/f noise
                        nf_clean = CleanSubarray(imch_diff, good_mask)
                        nf_clean.fit(model_type=self.model_type, horizontal_corr=self.horizontal_corr)
                        # Subtract model from data
                        data[i,j,:,x1:x2] -= nf_clean.model
                        del nf_clean

        return datamodel
    

def fit_slopes_to_ramp_data(input, sat_frac=0.5):
    """Fit slopes to each integration
    
    Uses custom `cube_fit` function to fit slopes to each integration.
    Returns array of slopes and bias values for each integration.
    Bias and slope arrays have shape (nints, ny, nx).

    Parameters
    ----------
    input : jwst.datamodel
        Input JWST datamodel housing the data to be fit.
    sat_frac : float
        Saturation threshold for fitting. Values above
        this fraction of the saturation level are ignored.
        Default is 0.5 to ensure that the fit is within 
        the linear range.
    """
    from jwst.datamodels import SaturationModel
    from jwst.saturation.saturation_step import SaturationStep
    from jwst.lib import reffile_utils
    from .utils import cube_fit

    # Get saturation reference file
    # Get the name of the saturation reference file
    sat_name = SaturationStep().get_reference_file(input, 'saturation')

    # Open the reference file data model
    sat_model = SaturationModel(sat_name)

    # Extract subarray from saturation reference file, if necessary
    if reffile_utils.ref_matches_sci(input, sat_model):
        sat_thresh = sat_model.data.copy()
    else:
        ref_sub_model = reffile_utils.get_subarray_model(input, sat_model)
        sat_thresh = ref_sub_model.data.copy()
        ref_sub_model.close()

    # Close the reference file
    sat_model.close()

    # Perform ramp fit to data to get bias offset
    group_time = input.meta.exposure.group_time
    ngroups = input.meta.exposure.ngroups
    nints = input.meta.exposure.nints
    tarr = np.arange(1, ngroups+1) * group_time
    data = input.data

    bias_arr = []
    slope_arr = []
    for i in range(nints):
        # Get group-level bpmask for this integration
        groupdq = input.groupdq[i]
        # Make sure to accumulate the group-level dq mask
        bpmask_arr = np.cumsum(groupdq, axis=0) > 0
        cf = cube_fit(tarr, data[i], bpmask_arr=bpmask_arr,
                        sat_vals=sat_thresh, sat_frac=sat_frac)
        bias_arr.append(cf[0])
        slope_arr.append(cf[1])
    bias_arr = np.asarray(bias_arr)
    slope_arr = np.asarray(slope_arr)

    # bias and slope arrays have shape [nints, ny, nx]
    # bias values are in units of DN and slope in DN/sec
    return bias_arr, slope_arr




class CleanSubarray:
    """
    CleanSubarray is the base class for removing residual correlated
    read noise from generic JWST near-IR Subarray images.  It is
    intended for use on Level 1 or 2 pipeline products on any type of
    image product (group data, rateint, calint, rate, cal, etc). 
    Suggest removing average signal levels from data and running
    this on the residuals, which better reveal the 1/f noise structure.

    Inspired by NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data. This is significantly faster than the FFT approach and provides 
    similar results. 
    """
            
    def __init__(self, data, mask, exclude_outliers=True):
        """ 1/f noise modeling and subtraction for HAWAII-2RG subarrays.

        This function assumes such that the slow-scan runs vertically along the
        y-axis, while the fast scan direction runs horizontally. Direction 
        (left to right or right to left) is irrelevant. 

        Parameters
        ==========
        D : ndarray
            Two-dimensional input data.
        M : ndarray
            Two dimensional background pixels mask. 
            Pixels =True are modeled as background.
            Pixels=False are excluded from the background model.
        exclude_outliers : bool
            Exclude statistical outliers and their nearest neighbors
            from the background pixels mask.
        """
        from .utils import expand_mask

        # Definitions
        self.D = np.array(data, dtype=np.float32)
        self.M = np.array(mask, dtype=np.bool_)
        
        # The mask potentially contains NaNs. Exclude them.
        self.M[np.isnan(self.D)] = False
        
        # The mask potentially contains statistical outliers.
        # Optionally exclude them.
        if exclude_outliers is True:
            gdpx = robust.mean(self.D, Cut=3, return_mask=True)
            bdpx = ~gdpx # Bad pixel mask
            bdpx = expand_mask(bdpx, 1, grow_diagonal=False) # Also flag 4 nearest neighbors
            gdpx = ~bdpx # Good pixel mask
            self.M = self.M & gdpx

        # Median subtract
        self.D = self.D - np.nanmedian(self.D[self.M])

    @property
    def nx(self):
        """ Number of pixels in fast scan direction"""
        return np.int32(self.D.shape[1])
    @property
    def ny(self):
        """ Number of pixels in slow scan direction"""
        return np.int32(self.D.shape[0])

    def fit(self, model_type='savgol', horizontal_corr=False, **kwargs):
        """ Return the model which is just median of each row
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        horizontal_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.
        """

        # Fit the model
        if 'median' in model_type:
            self.model = self._fit_median(robust_mean=False)
        elif 'mean' in model_type:
            self.model = self._fit_median(robust_mean=True)
        elif 'savgol' in model_type:
            self.model = self._fit_savgol(**kwargs)
        else:
            raise ValueError(f"Do not recognize model_type={model_type}")
        
        # Apply horizontal correction
        if horizontal_corr:
            data = self.D.copy()
            model = self.model.copy()
            # Rotate model subtracted data by 90 degrees
            self.D = np.rot90(data - model, k=1)
            self.M = np.rot90(self.M, k=1)
            # Fit the model
            self.fit(model_type=model_type, horizontal_corr=False, **kwargs)
            
            # Rotate data and model back to original orientation
            self.D = np.rot90(self.D, k=-1)
            self.M = np.rot90(self.M, k=-1)
            self.model = np.rot90(self.model, k=-1) + model

            del model, data

    def _fit_median(self, robust_mean=False):
        """ Return the model which is just median of each row
        
        Option to use robust mean instead of median.
        """

        mean_func = robust.mean if robust_mean else np.nanmedian

        # Fit the model
        data = self.D.copy()
        data[~self.M] = np.nan
        return mean_func(data, axis=1).repeat(self.nx).reshape(data.shape)
        
    def _fit_savgol(self, niter=5, **kwargs):
        """ Use a Savitzky-Golay filter to smooth the masked row data

        Parameters
        ==========
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.

        Keyword Args
        ============
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """

        bpmask = ~self.M # Bad pixel mask
        model = channel_smooth_savgol(self.D, mask=bpmask, **kwargs)

        for i in range(niter):
            # Get median model
            model_med = self._fit_median(robust_mean=False)

            # Find standard deviation of difference between model and median values
            bpmask = ~self.M 
            diff = model - model_med
            diff[bpmask] = np.nan
            sig = np.nanstd(diff, ddof=1)

            # Flag new outliers
            bp_sig = np.abs(diff) > 3*sig
            bpmask_new = bpmask | bp_sig
            if bpmask_new.sum() == bpmask.sum():
                break

            # Update mask and refit model
            self.M = ~bpmask_new
            model = channel_smooth_savgol(self.D, mask=bpmask_new, **kwargs)

        return model
        
    def clean(self, model_type='savgol', horizontal_corr=False, **kwargs):
        """ Clean the data

        Overwrites data in-place with the cleaned data.
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        horizontal_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """ 
        # Fit the background model
        self.fit(model_type=model_type, horizontal_corr=horizontal_corr, **kwargs) 
        self.D -= self.model # Overwrite data with cleaned data
        return self.D

def mask_helper():
    """Helper to handle indices and logical indices of a mask

    Output: index, a function, with signature indices = index(logical_indices),
    to convert logical indices of a mask to 'equivalent' indices

    Example:
        >>> # linear interpolation of NaNs
        >>> mask = np.isnan(y)
        >>> x = mask_helper(y)
        >>> y[mask]= np.interp(x(mask), x(~mask), y[~mask])
    """
    return lambda z: np.nonzero(z)[0]

def channel_smooth_savgol(im_arr, winsize=127, order=3, per_line=False, 
    mask=None, **kwargs):
    """Channel smoothing using savgol filter

    Function for generating a map of the 1/f noise within a series of input images.

    Parameters
    ==========
    im_arr : ndarray
        Input array of images (intended to be a cube of output channels).
        Shape should either be (ny, chsize) to smooth a single channel or
        (nchan, ny, chsize) for  multiple channels.
        Each image is operated on separately. If only two dimensions,
        then only a single input image is assumed. NaN's will be
        interpolated over.

    Keyword Args
    ============
    winsize : int
        Size of the window filter. Should be an odd number.
    order : int
        Order of the polynomial used to fit the samples.
    per_line : bool
        Smooth each channel line separately with the hopes of avoiding
        edge discontinuities.
    mask : bool image or None
        An image mask of pixels to ignore. Should be same size as im_arr.
        This can be used to mask pixels that the filter should ignore, 
        such as stellar sources or pixel outliers. A value of True indicates
        that pixel should be ignored.
    mode : str
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`. 
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : float
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    """
    from scipy.signal import savgol_filter

    sh = im_arr.shape
    if len(sh)==2:
        nz = 1
        ny, nx = sh
    else:
        nz, ny, nx = sh

    # Check that winsize is odd
    winsize = winsize-1 if winsize % 2 == 0 else winsize

    # Reshape in case of nz=1
    im_arr = im_arr.reshape([nz, -1])
    if mask is not None:
        mask = mask.reshape([nz, -1])

    res_arr = []
    for i, im in enumerate(im_arr):
        # im should be a 1D array

        # Interpolate over masked data and NaN's
        nans = np.isnan(im)
        im_mask = nans if mask is None else nans | mask[i].flatten()
        if im_mask.any():
            # Create a copy so as to not change the original data
            im = np.copy(im)

            # Use a savgol filter to smooth out any outliers
            res = im.copy()
            res[~im_mask] = savgol_filter(im[~im_mask], 33, 3, mode='interp')

            # Replace masked pixels with linear interpolation
            x = mask_helper() # Returns the nonzero (True) indices of a mask
            im[im_mask]= np.interp(x(im_mask), x(~im_mask), res[~im_mask])

        if per_line:
            im = im.reshape([ny,-1])

            res = savgol_filter(im, winsize, order, axis=1, delta=1, **kwargs)
            res_arr.append(res)
        else:
            res = savgol_filter(im, winsize, order, delta=1, **kwargs)
            res_arr.append(res.reshape([ny,-1]))
    res_arr = np.asarray(res_arr)

    if len(sh)==2:
        return res_arr[0]
    else:
        return res_arr
