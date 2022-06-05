from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import astropy.units as u

from astropy.visualization import ZScaleInterval, ImageNormalize

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class FITSIMAGE:
    """Handling Fits images
    """
    def __init__(self, fitsfname, index=0):
        """
        Args:
            fitsfname (str): file name for fits file
            index (int, optional): index of the hdulist of interest. Defaults to 0.
        """
        with fits.open(fitsfname) as hdul:
            self.header = hdul[index].header
            self.data = np.squeeze(hdul[index].data)
            self.wcs = WCS(self.header).celestial
            
    def cutout(self, ra, dec, radius=60):
        """Perform cutout for this fits file

        Args:
            ra (float): R.A. for the position of interest
            dec (float): Decl. for the position of interest
            radius (int/float, optional): radius of the cutout in arcsec. Defaults to 60.

        Returns:
            astropy.nddata.utils.Cutout2D
        """
        sourcecoord = SkyCoord(ra, dec, unit=u.degree)
        cutoutsize = (2*radius * u.arcsec, 2*radius * u.arcsec)
        return Cutout2D(self.data, sourcecoord, cutoutsize, wcs=self.wcs)

    def PlotData(self, fig=None, ax=None, norm=None, **kwargs):
        """Plot self.data with imshow function

        Args:
            fig (matplotlib.figure.Figure/None, optional): Figure to be plot. Defaults to None
            ax (matplotlib.axes._subplots.AxesSubplot/None, optional): axes to be plot. Defaults to None.
            norm (astropy.visualization.ImageNormalize/None, optional): Defaults to None.
        Returns:
            fig (matplotlib.figure.Figure)
            ax (matplotlib.axes._subplots.AxesSubplot)
            im (matplotlib.image.AxesImage)
        """
        if fig is None: fig = plt.figure(facecolor='w')
        if ax is None: ax = fig.add_subplot(1, 1, 1, projection=self.wcs)
        if norm is None: norm = ImageNormalize(self.data, interval=ZScaleInterval())

        kwargs.setdefault('cmap', 'gray_r')

        im = ax.imshow(self.data, norm=norm, **kwargs)

        return fig, ax, im

    def _getRMS(self):
        """Calculate RMS of self.data
        """
        sigmaclip = SigmaClip()
        filtered_data = sigmaclip(self.data)
        return filtered_data.std()

    def _getDataShape(self):
        """Calculate the shape of the data

        Returns:
            (float, float): data size (in arcsec) on two axes
        """
        pixel2sec_x = self.header['PC1_1'] * 3600.
        pixel2sec_y = self.header['PC2_2'] * 3600.
        pixelx, pixely = self.data.shape

        return pixel2sec_x * pixelx, pixel2sec_y * pixely

    def PlotContour(self, con_image, numlevels=5, **kwargs):

        fig, ax, _ = self.PlotData()

        ### check if size of con_image is smaller than that for self
        baseXsec, baseYsec = self._getDataShape()
        conXsec, conYsec = con_image._getDataShape()

        if min(baseXsec, baseYsec) < max(conXsec, conYsec):
            raise NotImplementedError("Not Implemented... Please make sure size of contour image is smaller")

        if not kwargs.get('levels'):
            max_value = con_image.data.max()
            std_value = con_image._getRMS()
            max_snr = max_value / std_value

            snr_seperation = (max_snr - 3) / numlevels
            levels = (np.arange(numlevels) + 3)*snr_seperation*std_value

            kwargs['levels'] = levels

        ct = ax.contour(con_image.data, transform=ax.get_transform(con_image.wcs), **kwargs)

        return fig, ax, ct

class CUTOUTIMAGE(FITSIMAGE):
    def __init__(self, cutout):
        """
        Args:
            cutout (astropy.nddata.utils.Cutout2D): cutout object
        """
        self.header = cutout.wcs.to_header()
        self.data = cutout.data
        self.wcs = cutout.wcs

