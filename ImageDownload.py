import numpy as np
import pandas as pd

from astropy.table import Table, Column
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import clear_download_cache

# from astroquery.ipac.ned import Ned
from astroquery.skyview import SkyView
from astroquery.cadc import Cadc
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.image_cutouts.first import First

import requests
import io
import os

def _saveHdulist_(hdulist, fname, overwrite=True):
    """Save an hdulist to local file

    Args:
        hdulist (astropy.io.fits.HDUList): An HDUList you want to save
        fname (str): path and file name you want to save your HDUList to be
        overwrite (bool, optional): Whether overwrite or not. Defaults to True.
    """
    hdulist.writeto(fname, overwrite=overwrite)

# Various Functions for performing cutout on different surveys
#################################################################
#  Optical or NIR
#################################################################

### PanSTARRS
def _getimages_PanSTARRS_(ra,dec,size=240,filters="grizy"):
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    return Table.read(url, format='ascii')

def _geturl_PanSTARRS(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):

    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = _getimages_PanSTARRS_(ra,dec,size=size,filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url += f"&output_size={output_size}"
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = f"{url}&{param}={table['filename'][i]}"
    else:
        urlbase = f"{url}&red="
        url = [urlbase+filename for filename in table['filename']]
    return url

### Skymapper
def _geturl_skymapper(ra, dec, radius):
    """Fetch cutout data via Skymapper API.

    Args:
        ra (float): R.A. for the source of interest
        dec (float): Decl. of the source of interest
        radius (float/int): cutout radius

    Returns:
        str: link for downloading Skymapper image
    """

    radius_degree = radius / 3600
    linka = 'http://api.skymapper.nci.org.au/aus/siap/dr2/'
    linkb = f'query?POS={ra:.5f},{dec:.5f}&SIZE={radius_degree:.3f}&BAND=all&RESPONSEFORMAT=CSV'
    linkc = '&VERB=3&INTERSECT=covers'
    sm_query = linka + linkb + linkc

    link = linka + 'get_image?IMAGE={}&SIZE={}&POS={},{}&FORMAT=fits'

    table = requests.get(sm_query.format(ra, dec, radius))
    df = pd.read_csv(io.StringIO(table.text))
    impos = f'{ra:.2f}, {dec:.2f}'
    assert len(df) > 0, f'No Skymapper image at {impos}'
    assert 'Gateway Time-out' not in df.iloc[0], f'Skymapper Gateway Time-out for image at {impos}'

    df = df[df.band == 'z']
    link = df.iloc[0].get_image
    return link

### DECam LS
def _geturl_decam(ra, dec, radius, band='g'):
    """Fetch cutout data via DECam LS API.
    credit: Joshua Pritchard

    Args:
        ra (float): R.A. for the source of interest
        dec (float): Decl. of the source of interest
        radius (float/int): cutout radius in arcsec

    Returns:
        str: link for downloading DECam LS image
    """

    # Ensure requested image size is less than DECam maximum of 512x512
    size = int(radius / 0.262)
    if size > 512:
       size = 512
       radius = size * 0.262 / 3600

    link = f"http://legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}"
    link += f"&size={size}&layer=dr8&pixscale=0.262&bands={band}"

    return link

#################################################################
#  Radio
#################################################################

### VLASS
def _getimage_vlass(ra, dec, radius=60.):
    """Download VLASS HDUlists

    Args:
        ra (float): R.A. for the source of interest
        dec (float): Decl. of the source of interest
        radius (float/int, optional): cutout radius in arcsec. Defaults to 60..

    Returns:
        astropy.io.fits.HDUList: VLASS HDULists
        None: if there is an error when downloading data
    """

    coord = SkyCoord(ra, dec, unit=u.degree)
    radius = radius * u.arcsec

    cadc = Cadc()
    try: hdulists = cadc.get_images(coord, radius, collection='VLASS')
    except: return None # likely source not in the VLASS footprint
    return hdulists

def _getVLASS_epoch(hdulist):
    """Get VLASS epoch from HDUList

    Args:
        hdulist (astropy.io.fits.HDUList): VLASS HDUList

    Returns:
        str: epoch of VLASS image
    """
    header = hdulist[0].header
    return f"{header['FILNAM01']}.{header['FILNAM02']}"

#################################################################
#  Download Function
#################################################################

def _checkDuplicated_(fname):
    return os.path.exists(fname)

def DownloadImage(ra, dec, radius, survey, savedir, cache=False, clearcache=True):
    """Download Cutout Fits images for a survey

    Args:
        ra (float): R.A. for the source of interest
        dec (float): Decl. for the source of interest
        radius (float): Cutout radius in arcsec
        survey (str): Survey you need to download
        savedir (str): Directory to put all fits files
        cache (bool, optional): Whether use the cache. Defaults to False.
        clearcache (bool, optional): Whether to clear cache after downloading. Defaults to True.

    Returns:
        int: -1 for unsuccessful, 0 for successful
    """

    _survey = survey.replace(' ', '_')
    fits_fname = '{}_{}.fits'.format(_survey, int(radius))
    fitspath = os.path.join(savedir, fits_fname)

    if os.path.exists(fitspath): return 0

    if survey == 'VLASS': # VLASS need to be handled with care
        hdulists = _getimage_vlass(ra, dec, radius)
        if hdulists is None: return -1
        for hdulist in hdulists:
            _survey = _getVLASS_epoch(hdulist)
            fits_fname = f'{_survey}_{int(radius)}.fits'
            fitspath = os.path.join(savedir, fits_fname)
            _saveHdulist_(hdulist, fitspath)

        if clearcache: clear_download_cache()
        return 0

    if survey == 'PanSTARRS':
        urls = _geturl_PanSTARRS(ra, dec, size=radius*4,filters="g",format='fits')
        if len(urls) == 0: return -1
        hdulist = fits.open(urls[0]); hdulists = [hdulist]

    elif survey == 'SkyMapper':
        try: url = _geturl_skymapper(ra, dec, radius)
        except: return -1
        hdulist = fits.open(url); hdulists = [hdulist]

    elif survey == 'DECam':
        try:
            url = _geturl_decam(ra, dec, radius)
            hdulist = fits.open(url)
            hdulists = [hdulist]
        except: return -1
    elif survey == 'FIRST':
        try:
            coord = SkyCoord(ra, dec, unit=u.degree)
            radius = radius * u.arcsec
            hdulists = First.get_images(coord, image_size=2*radius)
        except: return -1
    else:
        try:
            hdulists = SkyView.get_images(position=f'{ra} {dec}',survey=[survey], radius=radius*u.arcsec,cache=cache)
        except:
            return -1
    
    _saveHdulist_(hdulists[0], fitspath)
    if clearcache: clear_download_cache()
    return 0