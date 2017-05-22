import argparse

import numpy as np
from skimage.io import imread, imsave

class GlitchEffect(object):
    """Class for applying a glitch effect to images.
    """

    def __init__(self):
        self.hshift_min = 0.0
        self.hshift_max = 0.025
        self.vband_min = 0.1
        self.vband_max = 0.2

    @staticmethod
    def _split_channels(image):
        """Splits an image into component channels.
        """
        return [image[:, :, x] for x in range(image.shape[2])]

    @staticmethod
    def _merge_channels(*channels):
        """Merges channels back together into a single image.
        """
        return np.stack(channels, axis=2)

    @staticmethod
    def _random_uniform(low, high):
        return low + np.random.random() * (high - low)

    def _shift(self, channel):
        """Distorts a channel with horizontal displacement.

        Repeats edge pixels as needed.
        """
        shifted = np.zeros_like(channel)
        _, width = channel.shape

        shift = self._random_uniform(self.hshift_min, self.hshift_max)
        shift = int(width * shift)

        if np.random.random() > 0.5:
            shifted[:, shift:] = channel[:, :(width-shift)]
            shifted[:, :shift] = channel[:, 0][:, None]
        else:
            shifted[:, :(width-shift)] = channel[:, shift:]
            shifted[:, (width-shift):] = channel[:, -1][:, None]

        return shifted

    def _generate_bands(self, channel):
        height, _ = channel.shape
        lower = 0

        while lower < height:
            upper = self._random_uniform(self.vband_min, self.vband_max)
            upper = lower + int(height * upper)
            upper = min(upper, height)
            yield lower, upper
            lower = upper

    def _wave(self, channel):
        """Distorts a channel with banded horizontal displacements.
        """
        banded = np.zeros_like(channel)
        bands = self._generate_bands(channel)

        for lower, upper in bands:
            banded[lower:upper, :] = self._shift(channel[lower:upper, :])

        return banded

    def render(self, image):
        """Renders a glitch effect on an image.
        """
        r, g, b = self._split_channels(image)
        r = self._wave(r)
        g = self._wave(g)

        return self._merge_channels(r, g, b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    image = imread(args.infile)
    glitcher = GlitchEffect()
    glitched = glitcher.render(image)
    imsave(args.outfile, glitched)

if __name__ =='__main__':
    main()
