
import numpy as np
from scipy import misc
from scipy.stats import mode
import logging
from split_sbig_fits import split_sbig_fits

log = logging.getLogger(__name__)

class CloudSensor():

    mask = None
    split_mask = None
    images = []
    dateobs = None
    shape = None

    def readMask(self, filename):
        """

        :param filename:
        :return:
        """
        log.debug('Reading in %s ...' % filename)
        self.mask = misc.imread(filename)
        if self.split_mask is not None:
            self.mask = self.mask[self.split_mask].reshape(self.shape)

        log.debug('Image shape is %s ' % (self.mask.shape,))
        self.mask = self.mask == 0

    def readAllSkyImages_SBIGFITS(self,imagelist):
        """

        :return:
        """
        log.debug('Reading in %i images' % len(imagelist))
        self.dateobs = np.zeros(len(imagelist),
                                dtype='S24')

        for itr, filename in enumerate(imagelist):
            log.debug('Reading and splitting rgb channels from %s ...' % filename)
            r, g, b, mask, self.shape, self.dateobs[itr] = split_sbig_fits(filename)
            self.images.append((r, g, b))
            if itr == 0:
                self.split_mask = mask

def main(argv):

    import argparse

    parser = argparse.ArgumentParser(description='Process a list of FITS file and store MJD, median, std of the'
                                                 'normalized b/r channels.')

    parser.add_argument('-f', '--file',
                        help='File containing list of input images to process.',
                        type=str)

    parser.add_argument('-m', '--mask',
                        help='Horizon mask.',
                        type=str)

    parser.add_argument('-o', '--output',
                        help='Output name. A .npy file to store the results.',
                        type=str)

    args = parser.parse_args(argv[1:])

    logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                        level=logging.DEBUG)

    import pylab as py
    from datetime import datetime
    from astropy.time import Time
    from astropy import coordinates, units
    import os

    cs = CloudSensor()
    images = np.loadtxt(args.file,
                        dtype='S',
                        ndmin=1)

    cs.readAllSkyImages_SBIGFITS(images)
    cs.readMask(args.mask)

    strtime = '%Y-%m-%dT%H:%M:%S.%f'

    cloud_stats = np.zeros(len(images),dtype=[('mjd', np.float),
                                              ('mean', np.float),
                                              ('median', np.float),
                                              ('std', np.float),
                                              ('sun_alt', np.float),
                                              ('sun_az', np.float),
                                              ('filename', 'S25')])
    ax1 = py.subplot(111)
    # ax2 = py.subplot(212)
    obs_lat = coordinates.Latitude("-30:10:04.31", unit='deg')
    obs_long = coordinates.Longitude("-70:48:20.48", unit='deg')
    obs_coord = coordinates.EarthLocation(lat=obs_lat, lon=obs_long, height=2700.*units.m)

    for itr in range(len(images)):
        r_dimg = np.array(cs.images[itr][0], dtype=np.float)
        g_dimg = np.array(cs.images[itr][1], dtype=np.float)
        b_dimg = np.array(cs.images[itr][2], dtype=np.float)

        mask = np.bitwise_and(np.bitwise_and(cs.images[itr][0] > 30000,
                                             cs.images[itr][1] > 30000),
                              cs.mask)
        # mask = np.bitwise_and(mask, r_dimg == 0)
        # mask = np.bitwise_and(mask, b_dimg == 0)
        r_dimg[mask] = 1
        g_dimg[mask] = 1
        b_dimg[mask] = 0
        r_dimg[r_dimg == 0] = 1
        g_dimg[g_dimg == 0] = 1

        # print r_dimg[r_dimg == 0]
        b_r = b_dimg/r_dimg
        b_r = b_r[np.bitwise_not(mask)]
        # print np.median(b_r)
        # b_r /= np.mean(b_dimg)
        dt = Time(datetime.strptime(cs.dateobs[itr], strtime))
        suncoord = coordinates.get_sun(dt)
        sun_altaz = suncoord.transform_to(coordinates.AltAz(obstime=dt, location=obs_coord))
        # print sun_altaz.alt.deg, sun_altaz.az.deg

        cloud_stats['mjd'][itr], cloud_stats['mean'][itr], \
        cloud_stats['median'][itr], cloud_stats['std'][itr], \
        cloud_stats['sun_alt'][itr], cloud_stats['sun_az'][itr], \
        cloud_stats['filename'][itr] = dt.mjd, \
                                       np.mean(b_r), \
                                       np.median(b_r), \
                                       np.std(b_r), \
                                       sun_altaz.alt.deg, \
                                       sun_altaz.az.deg, \
                                       os.path.basename(images[itr])

        dohist = b_dimg/r_dimg
        dohist = dohist.flatten()
        mean = np.mean(dohist)
        std = np.std(dohist)
        if itr == 0:
            bins = np.linspace(mean-3*std, mean+3*std, 200)
            log.info('computing histogram with %i elements between %f and %f' % (len(bins),
                                                                                 mean-std,
                                                                                 mean+std))
        # ax1.hist(dohist, bins=bins, alpha = 0.5)
        # ax2.hist(r_dimg.flatten(), bins=bins, alpha = 0.5)
        # log.info('done')
    # py.show()
        # py.plot(b_dimg.flatten()/g_dimg.flatten(),g_dimg.flatten()/r_dimg.flatten(),'.')
    # py.show()

    if args.output is not None:
        np.save(args.output,
                cloud_stats)
        # py.imshow(b_dimg/r_dimg)
    #     bins = np.arange(-.01/2., 1., 0.01)
    #     py.hist(b_r,bins = bins)
    # py.show()

if __name__ == '__main__':
    import sys
    main(sys.argv)