from warnings import warn
import numpy as np
import numexpr as ne
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.ndimage.filters import convolve

from skimage.util.dtype import dtype_range
from skimage.util.arraycrop import crop
from skimage._shared.utils import warn, check_shape_equality

def _as_floats(image0, image1):

    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1

def structural_similarity(im1, im2,
                          *,
                          win_size=None, gradient=False, data_range=None,
                          multichannel=True, gaussian_weights=False,
                          full=False, **kwargs):
    check_shape_equality(im1, im2)

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = im1.shape[-1]
        mssim = np.empty(nch)
        cs = np.empty(nch)
        if gradient:
            G = np.empty(im1.shape)
        if full:
            S = np.empty(im1.shape)
        for ch in range(nch):
            ch_result = structural_similarity(im1[..., ch],
                                              im2[..., ch], **args)
            if gradient and full:
                mssim[..., ch], cs[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], cs[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], cs[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch], cs[..., ch] = ch_result
        mssim = mssim.mean()
        cs = cs.mean()

        if gradient and full:
            return mssim, cs, G, S
        elif gradient:
            return mssim, cs, G
        elif full:
            return mssim, cs, S
        else:
            return mssim, cs

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(ne.evaluate("truncate * sigma + 0.5"))  # radius as in ndimage
            win_size = ne.evaluate("2 * r + 1")
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(ne.evaluate("im1 * im1"), **filter_args)
    uyy = filter_func(ne.evaluate("im2 * im2"), **filter_args)
    uxy = filter_func(ne.evaluate("im1 * im2"), **filter_args)
    vx = ne.evaluate("cov_norm * (uxx - ux * ux)")
    vy = ne.evaluate("cov_norm * (uyy - uy * uy)")
    vxy = ne.evaluate("cov_norm * (uxy - ux * uy)")

    R = data_range
    C1 = ne.evaluate("(K1 * R) ** 2")
    C2 = ne.evaluate("(K2 * R) ** 2")

    A1, A2, B1, B2 = ((ne.evaluate("2 * ux * uy + C1"),
                       ne.evaluate("2 * vxy + C2"),
                       ne.evaluate("ux ** 2 + uy ** 2 + C1"),
                       ne.evaluate("vx + vy + C2")))
    D = ne.evaluate("B1 * B2")
    S = ne.evaluate("(A1 * A2) / D")

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = 2*crop(S, pad).mean()
    cs=2*np.mean(ne.evaluate("A2/B2")) #used for multiscaled

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(ne.evaluate("A1 / D"), **filter_args) * im1
        grad += filter_func(ne.evaluate("-S / B2"), **filter_args) * im2
        grad += filter_func(ne.evaluate("(ux * (A2 - A1) - uy * (B2 - B1) * S) / D"),
                            **filter_args)
        grad *= (2 / im1.size)

        if full:
            return mssim, cs, grad, S
        else:
            return mssim, cs, grad
    else:
        if full:
            return mssim, cs, S
        else:
            return mssim, cs


def multiscale_structural_similarity(im1, im2,*,win_size=None, data_range=None, multichannel=True, gaussian_weights=False):
    check_shape_equality(im1, im2)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((2, 2, 1)) / 4.0
    mssim = np.array([])
    mcs = np.array([])

    for _ in range(levels):
        ssim, cs = structural_similarity(im1, im2, win_size=win_size,data_range=data_range, gaussian_weights=gaussian_weights)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect') for im in [im1, im2]]
        im1, im2 = [x[::2, ::2, :] for x in filtered]

    return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) * (mssim[levels-1] ** weights[levels-1]))


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean(ne.evaluate("(image0 - image1) ** 2"), dtype=np.float64)