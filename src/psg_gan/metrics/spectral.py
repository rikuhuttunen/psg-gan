import jax
import jax.numpy as jnp
import jax.scipy as jscipy


def compute_psds(
        s: jax.Array,
        fs: float = 1.0,
        epoch_length: int | None = None,  # Epoch length in seconds
        window: str = 'hann',
        nperseg: int | None = None,
        noverlap: int | None = None,
        nfft: int | None = None,
        detrend: str = 'constant',
        return_onesided: bool = True,
        scaling: str = 'density',
        axis: int = -2,  # Axis -1 are the channels, -2 is the signal dimension
        average: str = 'mean') -> jax.Array:
    """Compute power spectral densities channelwise.

    The last dimension of s is the channels. If there are multiple channels,
    e.g. EEG and PPG, the PSDs are calculated for both separately.
    """
    assert s.ndim == 2, 's should be [NC] dimensional'
    shape = jnp.shape(s)
    if epoch_length is not None:
        epoch_ns = int(epoch_length * fs)
        assert shape[0] // epoch_ns * epoch_ns == shape[0], 's is not divisible with epoch_length'
        s = jnp.reshape(s, [shape[0] // epoch_ns, epoch_ns, shape[1]])

    f, psd = jscipy.signal.welch(s, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                 detrend=detrend, return_onesided=return_onesided, scaling=scaling,
                                 axis=axis, average=average)
    return f, psd
