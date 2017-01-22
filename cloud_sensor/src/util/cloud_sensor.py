
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
    import os

    cs = CloudSensor()
    images = np.loadtxt(args.file,
                        dtype='S')

    cs.readAllSkyImages_SBIGFITS(images)
    cs.readMask(args.mask)

    strtime = '%Y-%m-%dT%H:%M:%S.%f'

    cloud_stats = np.zeros(len(images),dtype=[('mjd', np.float),
                                              ('mean', np.float),
                                              ('std', np.float),
                                              ('filename', 'S25')])

    for itr in range(len(images)):
        r_dimg = np.array(cs.images[itr][0], dtype=np.float)

        b_dimg = np.array(cs.images[itr][2], dtype=np.float)
        mask = np.bitwise_and(np.bitwise_and(cs.images[itr][0] > 30000,
                                             cs.images[itr][1] > 30000),
                              cs.mask)
        # mask = np.bitwise_and(mask, r_dimg == 0)
        # mask = np.bitwise_and(mask, b_dimg == 0)
        r_dimg[mask] = 1
        b_dimg[mask] = 0
        r_dimg[r_dimg == 0] = 1

        # print r_dimg[r_dimg == 0]
        b_r = b_dimg/r_dimg
        b_r = b_r[np.bitwise_not(mask)]
        # print np.median(b_r)
        # b_r /= np.max(b_r)
        dt = Time(datetime.strptime(cs.dateobs[itr], strtime))
        cloud_stats['mjd'][itr], cloud_stats['mean'][itr], cloud_stats['std'][itr] = dt.mjd, np.mean(b_r), np.std(b_r)
        cloud_stats['filename'][itr] = os.path.basename(images[itr])

    np.save(args.output,
            cloud_stats)
        # py.imshow(b_dimg/r_dimg)
    #     bins = np.arange(-.01/2., 1., 0.01)
    #     py.hist(b_r,bins = bins)
    # py.show()

if __name__ == '__main__':
    import sys
    main(sys.argv)