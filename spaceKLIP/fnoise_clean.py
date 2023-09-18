import os
import numpy as np

from webbpsf_ext import robust
from scipy.signal import savgol_filter

# class CleanFullFrame:

class CleanSubarray:
    """
    CleanSubarray is the base class for removing residual correlated
    read noise from generic JWST near-IR Subarray images.  It is
    intended for use on Level 1 pipeline products.

    Adapted from NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data.  This is much faster than the FFT approach. 
    """
    
    # Class variables. These are the same for all instances.
    nloh = np.int32(12)       # New line overhead in pixels
    tpix = np.float32(10.e-6) # Pixel dwell time in seconds
    sigrej = np.float32(4.0)  # Standard deviation threshold for flagging
                              #   statistical outliers.
        
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
        self.ny = np.int32(data.shape[0]) # Number of pixels in slow scan direction
        self.nx = np.int32(data.shape[1]) # Number of pixels in fast scan direction
        self.n = np.int32(self.ny * (self.nx + self.nloh)) # Number of ticks in clocking pattern
        
        # The mask potentially contains NaNs. Exclude them.
        self.M[np.isnan(self.D)] = False
        
        # The mask potentially contains statistical outliers.
        # Optionally exclude them.
        if exclude_outliers is True:
            # Compute median and median absolute deviation
            m = np.nanmedian(self.D[self.M])
            s = robust.medabsdev(self.D[self.M]) 
            vmin = m - self.sigrej*s # Minimum value to keep
            vmax = m + self.sigrej*s # Maximum value to keep

            bdpx = np.isnan(self.D) | (self.D<vmin) | (self.D>vmax) # Flag statistical outliers
            bdpx = expand_mask(bdpx, 1, grow_diagonal=False) # Also flag 4 nearest neighbors

            # bdpx now contains the pixels to exclude from the background pixels
            # mask. Exclude them.
            self.M[bdpx] = False

        # Median subtract
        self.D = self.D - np.nanmedian(self.D[self.M]) 

    def fit(self, savgol=False, **kwargs):
        """ Return the model which is just median of each row"""

        # Fit the model
        if savgol:
            self.fit_savgol(**kwargs)
        else:
            self.fit_median(**kwargs)

    def fit_median(self, **kwargs):
        """ Return the model which is just median of each row"""

        # Fit the model
        data = self.D.copy()
        data[~self.M] = np.nan
        self.model = np.nanmedian(data, axis=1).repeat(self.nx).reshape(data.shape)
        
    def fit_savgol(self, **kwargs):
        """ Use a Savitzky-Golay filter to smooth the data

        Parameters
        ==========
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
        self.model = channel_smooth_savgol(self.D, mask=bpmask, **kwargs)
        
    def clean(self, weight_fit=True):
        """
        Clean the data
        
        Parameters: weight_fit:bool
                      Use weighted least squares as described in the NSClean paper.
                      Otherwise, it is a simple unweighted fit.

        """ 
        self.fit(weight_fit=weight_fit)           # Fit the background model
        self.D -= self.model # Overwrite data with cleaned data
        return(self.D)

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
            res[~im_mask] = savgol_filter(im[~im_mask], 31, 3, mode='interp')

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
