# This is an utility function to work with SBIG AllSky-340C color images stored in fits format.

import numpy as np
from astropy.io import fits
import os


def split_sbig_fits(filename):
    """
    Read fits file from an SBIG AllSky-340C (or similar) and split it into the RGB colors.

    :param filename:
    :return:
    """

    hdu = fits.open(filename)

    NAXIS1 = hdu[0].header['NAXIS1']
    NAXIS2 = hdu[0].header['NAXIS2']
    dateobs = None

    if 'DATE-OBS' in hdu[0].header:
        dateobs = hdu[0].header['DATE-OBS']

    mask_r = np.zeros((NAXIS2, NAXIS1)) == 0
    for i in range(len(mask_r)):
        mask_r[i] = (-1 * ((i + 1) % 2)) ** np.arange(len(mask_r[i])) == -1

    mask_g = np.zeros((NAXIS2, NAXIS1)) == 0
    for i in range(len(mask_g)):
        mask_g[i] = (-1 + np.zeros_like(mask_g[i])) ** (i + np.arange(len(mask_g[i]))) == -1

    mask_b = np.zeros((NAXIS2, NAXIS1)) == 0
    for i in range(len(mask_b)):
        mask_b[i] = (-1 * (i % 2)) ** np.arange(len(mask_b[i])) == -1

    # Build green color-matrix
    g = hdu[0].data[mask_g].reshape(NAXIS2 / 2, NAXIS1)
    g1 = g[::, :NAXIS1 / 2:]
    g2 = g[::, NAXIS1 / 2::]

    g = np.mean([g1, g2], axis=0)

    # Build red color-matrix
    r = hdu[0].data[mask_r].reshape(NAXIS2 / 2, NAXIS1 / 2)

    # Build blue color-matrix
    b = hdu[0].data[mask_b].reshape(NAXIS2 / 2, NAXIS1 / 2)

    return r, g, b, mask_r, (NAXIS2 / 2, NAXIS1 / 2), dateobs
