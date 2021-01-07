'''
Created on Sep 19, 2019
author: landeros10

Implements analysis functions and outlines data structures for storing sample
images, channel and sample names.
'''
from typing import List, Dict, Set, Iterable, Callable, Tuple
import numpy as np
import multiprocessing
import itertools
import logging
from time import time
from skimage.registration import phase_cross_correlation
from skimage.measure import label, regionprops_table
from skimage.morphology import (binary_opening, binary_closing,
                                binary_erosion, binary_dilation, disk)
from skimage.filters import threshold_triangle
from scipy.spatial import distance
from tifffile import imread
from os.path import join


MASK_PARAMS = {
    "cutoff": 0.9,
    "dapi_dil": 2,
    "cd45_dil": 2,
    "dapi_f": 1.0,
    "cd45_f": 1.0
}


def channel2cycle(name):
    return name.split("_")[0]


class SampleImage:
    """Data structure for a single image (channel) in patient sample.

    Args:
        imFile (str): full directory and filename for image
        name (str): image (channel) name

    Attributes:
        imFile (str): full directory and filename for image
        name (str): image (channel) name
        cycle (type): substring in channel, indicates imaging cycle
    """

    def __init__(self, imFile: str, name: str):
        self.imFile = imFile
        self.name = name
        self.cycle = channel2cycle(name)

    def get_array(self, normalize=True, dtype=None) -> np.ndarray:
        """Open image file(s) as numpy array.
        Returns:
            np.ndarray: image file() as array.
        """
        array = imread(self.imFile)
        if dtype is not None:
            array = array.astype(dtype)

        if normalize:
            array = norm(clip_percentiles(array))
        return array


class SampleCycle:
    """SampleCycle object groups SampleImage objects by imaging cycle.
    Contains import functions for cycle-cycle registration.

    Args:
        name (str): cycle name.
        images (Dict): dictionary of channel_name: SampleImage. All channels in
        same imaging cycle.

    Attributes:
        channels (list): channel name.
        name (str): cycle name.
        images (Dict): dictionary of channel_name: SampleImage
    """

    def __init__(self,
                 name: str,
                 images: Dict[str, SampleImage]):
        self.name = name
        self.images = images
        self.channels = sorted(list(self.images.keys()))

    def get_array(self, normalize=True, maxp=True, dtype=None) -> np.ndarray:
        """Open image file(s) in cycle as numpy array. Compression by max
        projection optional.

        Returns:
            np.ndarray: image file() as array.
        """
        # Get normalized image arrays from SampleImage objects
        im_arr = [self.images[ch].get_array(normalize=normalize, dtype=dtype)
                  for ch in self.channels]
        im_arr = np.stack(im_arr, axis=2)
        if maxp:
            im_arr = im_arr.max(axis=2)
        return im_arr

    @staticmethod
    def make_cutoff_fn(thresh_fun, scale):
        def get_cutoff(input):
            p25 = np.percentile(input, 25, axis=(0, 1))
            if input.ndim == 2:
                cutoff = scale * thresh_fun(input)
            else:
                N = input.shape[-1]
                cutoff = [scale * thresh_fun(input[..., i]) for i in range(N)]
            return np.maximum(p25, cutoff)
        return get_cutoff

    @staticmethod
    def get_masks(cd45, dapi,
                  cd45_f: float = 1.0,
                  dapi_f: float = 1.0,
                  cd45_dil: int = 1,
                  dapi_dil: int = 3,
                  cd45_cutoffs: list = [500, 70000],
                  dapi_cutoffs: list = [300, 7000],
                  parallel=True):
        cd45_thresh = threshold_triangle(cd45[cd45 > 0])
        dapi_thresh = threshold_triangle(dapi[dapi > 0])

        cd45_thresh = cd45_f * cd45_thresh
        dapi_thresh = dapi_f * dapi_thresh

        dapi = dapi > dapi_thresh
        cd45 = cd45 > cd45_thresh

        cd45 = dapi_processing(cd45, cd45_dil, 0, cd45_cutoffs)[0]
        dapi = dapi_processing(dapi, dapi_dil, 0, dapi_cutoffs)[0]
        return cd45, dapi

    @staticmethod
    def normalize_unmasked(im):
        """ standard normalization, inplace array manipulation"""
        np.subtract(im, im.min(), out=im)
        np.multiply(im, 1.0/im.max(), out=im)

    @staticmethod
    def normalize_masked(im, mask, pos_min):
        """Normalize masked image pixel values from pos_min to 1.0.

        Foreground pixels defined by mask are normalized to a range between
        pos_min and 1.0."""
        im[~mask] = 0
        np.subtract(im, im[mask].min(), where=mask, out=im)

        scale_factor = (1 - pos_min) / im[mask].max()
        np.multiply(im, scale_factor, where=mask, out=im)
        np.add(im, pos_min, where=mask, out=im)

    @classmethod
    def clean_objects(cls, mask: np.ndarray, size_cutoffs: Iterable):
        """Remove noise objects from mask array. mask.ndim must be 2 or 3.

        Remove salt noise with binary_opening(), then labels individual
        objects. Objects whose area are too small or too large are removed from
        the mask. This helps focus alignment on cell-sized objects.
        Args:
            mask (np.ndarray[Bool]): Mask array.
            size_cutoffs (Iterable): [MIN, MAX] cutoffs for object areas.
        """
        if mask.ndim == 2:
            cls._clean_objects(mask, size_cutoffs)
        elif mask.ndim == 3:
            for i in range(mask.shape[2]):
                cls._clean_objects(mask[..., i], size_cutoffs=size_cutoffs)

    @staticmethod
    def _clean_objects(mask: np.ndarray, size_cutoffs: Iterable):
        """ Label objects in mask, remove those outside of cutoff range.

        Remove salt noise with binary_opening(), then labels individual
        objects. Objects whose area are too small or too large are removed from
        the mask. This helps focus alignment on cell-sized objects.

        Args:
            mask (np.ndarray[Bool]): Mask array.
            size_cutoffs (Iterable): [MIN, MAX] cutoffs for object areas.
        """
        binary_opening(mask, disk(3), out=mask)
        binary_closing(mask, disk(3), out=mask)

        labels, num = label(mask, return_num=True)
        areas = np.array(
            regionprops_table(labels, properties=('area',))["area"]
        )
        to_remove = np.where(
            (areas < size_cutoffs[0]) | (areas > size_cutoffs[1])
        )
        to_remove = to_remove[0] + 1
        np.logical_not(mask, where=np.isin(labels, to_remove), out=mask)
        binary_opening(mask, disk(5), out=mask)
        binary_dilation(mask.astype(int), out=mask)

    @classmethod
    def preprocess_ims(cls,
                       x: np.ndarray,
                       y: np.ndarray,
                       thresh_fun: Callable,
                       scale: float = 1.0,
                       size_cutoffs=[500, 50000],
                       pos_min: float = 0.5,
                       params=None):
        """Preprocess images prior to alignment.

        Args:
            x (np.ndarray): array of shape [rows, cols] (reference)
            y (np.ndarray): array of shape [rows, cols, n] (channel imgs)
            thresh_fun (Callable): thresholding function for image binarization
            scale (float): threshold scale. Defaults to 0.95.

        Returns:
            Tuple[np.ndarray, np.ndarray]: pre-processed arrays.

        """
        # get_cutoff = cls.make_cutoff_fn(thresh_fun, scale)
        # # Remove background objects and re-scale foreground intensity
        # mask_x = x > get_cutoff(x)
        # cls.clean_objects(mask_x, size_cutoffs)
        if params is None:
            params = MASK_PARAMS.copy()

        params.pop("cutoff")
        mask_x, mask_y = cls.get_masks(x, y, **params)
        cls.normalize_unmasked(x)

        # Create mask for cycle images and remove noise as with x
        # mask_y = y > get_cutoff(y)
        # cls.clean_objects(mask_y, size_cutoffs)
        cls.normalize_unmasked(y)
        return mask_x, mask_y

    @staticmethod
    def check_if_empty(x: np.ndarray,
                       y: np.ndarray) -> bool:
        """Check if a preprocessed image has any cell-sized objects.
        Args:
            x (np.ndarray): preprocessed image, highlighting cell-sized objects
            y (np.ndarray): preprocessed image, highlighting cell-sized objects
        Returns:
            bool: Description of returned object.
        """
        return ((x.sum() < 2000) or (y.sum() < 2000))

    @staticmethod
    def get_random_crop(ref_shape, im_shape, frac):
        # For now, assume len(ref_shape) = len(im_shape) = 2
        h, w = ref_shape
        if frac < 1.0:
            sh, sw = int((frac * h)), int((frac * w))
            idy = np.random.randint(0, h - sh)
            idx = np.random.randint(0, w - sw)
        elif frac == 1.0:
            idy, idx = 0, 0
            sh, sw = h, w
        return idy, sh, idx, sw

    @staticmethod
    def crop(ref: np.ndarray,
             im: np.ndarray,
             crop_params) -> Tuple[np.ndarray, np.ndarray]:
        idy, sh, idx, sw = crop_params
        refCrop = ref[idy:idy + sh, idx:idx + sw]
        imCrop = im[idy:idy + sh, idx:idx + sw]
        return refCrop, imCrop

    @staticmethod
    def shift_array(image, shiftVals, min_val=None):
        """Shifts image from top-left corner in the first two axes.

        Args:
            image (np.ndarray): Array to shift.
            shiftVals (Iterable): iterable of length 2 that describes the
                image shift as (offset in rows, offset in cols)
        Returns:
            np.ndarray: shifted image as numpy array.
        """
        if np.array_equal(shiftVals, [0, 0]):
            return np.array(image)

        # Convert shift values to padding tuples
        shiftr, shiftc = int(shiftVals[0]), int(shiftVals[1])
        padr = (shiftr, 0) if np.sign(shiftr) >= 0 else (0, -shiftr)
        padc = (shiftc, 0) if np.sign(shiftc) >= 0 else (0, -shiftc)

        v = 0
        if min_val is not None:
            v = min_val

        # May have change padding in new numpy version
        if image.ndim > 2:
            shifted_im = np.pad(image,
                                (padr, padc, (0, 0)),
                                constant_values=v)
        else:
            shifted_im = np.pad(image,
                                (padr, padc),
                                constant_values=v)

        # Crop to original size
        h, w = shifted_im.shape[:2]
        shifted_im = shifted_im[padr[1]:h-padr[0], padc[1]:w-padc[0]]
        return shifted_im

    @classmethod
    def get_shift_acc(cls, ref: np.ndarray, im: np.ndarray, shift: List):
        shifted_im = cls.shift_array(im, shift)
        return get_percentage(ref, shifted_im, 0.4)
        # total = (ref > 0).sum()
        # accs = ((shifted_im > 0)[ref > 0].sum(axis=0)) / total
        # return np.mean(accs)

    @staticmethod
    def variance_2d(points: np.ndarray) -> float:
        """Find variance in datapoints by taking mean distance to centroid.
        Args:
            points (np.ndarray): array of points of shape [N, 2].

        Returns:
            float: variance.
        """
        mean = np.mean(points, axis=0)
        d = [distance.euclidean(i, mean) for i in points]
        return np.mean(d)

    @classmethod
    def remove_outlier(cls,
                       shifts: np.ndarray,
                       remove_n: int = 1) -> np.ndarray:
        """Remove shift outliers that contribute to greater trial variance.

        Variance is calculated after removal of a single trial. If variance
        improves it is removed from the final list of trials.

        Args:
            shift_trials (np.ndarray): array of shape [N, 2] of shift trials.
            remove_n (int): Number of trials to remove. Defaults to 1.

        Returns:
            np.ndarray: updated shift trials of size [n, 2] where n >= N-2.
        """
        for removal in range(remove_n):
            N = len(shifts)
            if N > 2:
                best_var = cls.variance_2d(shifts)
                to_remove = None
                for i in range(N):
                    remaining = np.delete(shifts, i, axis=0)
                    var = cls.variance_2d(remaining)
                    if var < best_var:
                        to_remove = i
                        best_var = var
                if to_remove is not None:
                    shifts = np.delete(shifts, to_remove, axis=0)
                else:
                    return shifts
        return shifts

    def align(self,
              im: np.ndarray,
              scale: float = 1.0,
              trials: int = 3,
              maxAttempts: int = 3,
              frac: float = 0.80,
              fracIncrement: float = 0.05,
              minFrac: float = 0.80,
              shiftThresh: int = 150,
              shiftIncrement: int = 25,
              maxShiftThresh: int = 350,
              minAcc: float = 0.8,
              accIncrement: float = 0.02,
              overlap_ratio: float = 0.6,
              returnNan: bool = True,
              thresh_fun: Callable = threshold_triangle) -> List[int]:

        # Get normalized projected array from all images in cycle
        ref = self.get_array(maxp=True, dtype=float)
        refh, refw = ref.shape

        # No maximum shift in initial assessment
        if not returnNan:
            maxShiftThresh = 400

        # Preprocessing steps
        refMask, imMask = self.preprocess_ims(ref, im, thresh_fun, scale=scale)

        # If preprocessing detects no cell-sized objects in either image
        # then we cannot accurately determine alignment. [0, 0] alignment
        # should have little influence on overall signal aggregation.
        if self.check_if_empty(refMask, imMask):
            return np.array([0, 0]), 1.0

        # Ask for a reasonable accuracy of TP / (TP + FN) on foreground
        maxAcc = min(len(im[im > 0]) / len(ref[ref > 0]), 1.0)
        minAcc = min(minAcc, maxAcc)

        t = 0
        finalShifts = []
        accs = []
        # Store for cycling through attempts w different thresholds
        frac_init, shiftThresh_init = frac,  shiftThresh
        while t < trials:
            t += 1
            successfulTrial = False
            trial_attempts = 0
            while not successfulTrial:
                # Crop a sub-image
                trial_attempts += 1
                p = self.get_random_crop(ref.shape, im.shape, frac)
                refCrop, imCrop = self.crop(ref, im, p)
                refCrop_mask, imCrop_mask = self.crop(refMask, imMask, p)

                # From Sci-kit image
                # fft(refCrop) * fft(imCrop).conj()
                # Normalized cross-correlaiton much slower, pre-processing
                # steps ensure accuracy of the registration by normalizing
                # intensities and focusing on cell-sized objects
                results = phase_cross_correlation(refCrop, imCrop,
                                                  reference_mask=refCrop_mask,
                                                  moving_mask=imCrop_mask,
                                                  return_error=True,
                                                  overlap_ratio=overlap_ratio)
                trialShift = results[:2]

                # Accuracy measured by TP / (TP + FN) on foreground (ie. cells)
                # Is 1.0 if all reference cells are covered by im cells
                # when shifted by trialShift
                acc = self.get_shift_acc(refMask, imMask, trialShift)
                # Sucessful trials fall below maximum shift
                if np.all(np.abs(trialShift) < shiftThresh) and acc >= minAcc:
                    successfulTrial = True
                else:
                    if np.any(np.abs(trialShift) >= shiftThresh):
                        if overlap_ratio == 0.999:
                            return np.array([0, 0]), 1.0
                        elif overlap_ratio >= 0.9:
                            overlap_ratio = min(overlap_ratio + 0.02, 0.999)
                        else:
                            overlap_ratio = min(overlap_ratio + 0.2, 0.9)
                    elif not returnNan:
                        minAcc = max(minAcc - accIncrement, 0.00)

                    # After we attempt maxAttempts times
                    if trial_attempts >= maxAttempts:
                        # First, we iterate through smaller sub-image fractions
                        if frac < 1.0:
                            frac = min(frac + fracIncrement, 1.0)
                        else:
                            # If we have maximum shift idea, rather output NaN
                            # for values that reach a maximum deviation from
                            # the initial threshold, shiftThresh
                            if shiftThresh < maxShiftThresh:
                                frac = frac_init
                                shiftThresh = min(shiftThresh + shiftIncrement,
                                                  maxShiftThresh)
                                # overlap_ratio = 0.5
                            else:
                                if returnNan:
                                    return [np.nan, np.nan], np.nan
                                else:
                                    minAcc = max(minAcc - accIncrement, 0.00)
                                    frac = frac_init
                                    shiftThresh = shiftThresh_init
                        trial_attempts = 0
            finalShifts.append(trialShift)
            accs.append(acc)

        # remove outliers to use an average of 3 or less calculations
        to_remove = max(0, trials-3)
        finalShifts = self.remove_outlier(finalShifts, remove_n=to_remove)

        finalShifts = np.mean(finalShifts, axis=0).astype(int)
        return list(finalShifts), np.mean(accs)

    def shift(self, shiftVals: Iterable) -> np.ndarray:
        """Reads image file and applies a shift from the top left corner.

        Args:
            shiftVals (Iterable): iterable of length 2 that describes the
                image shift as (offset in rows, offset in cols)
        Returns:
            np.ndarray: shifted image as numpy array.
        """
        im = self.get_array(normalize=True, maxp=False)  # ndim = 3
        if np.array_equal(shiftVals, [0, 0]):
            return np.array(im)
        return self.shift_array(im, shiftVals)


class Sample:
    """ Data structure for 1 FOV in complete patient sample.
    Individual channels stored as SampleCycle objects, grouped by imaging
    cycle. Defines operations to be executed on whole-samples.

    Args:
        name (str): patient sample name
        images (dict): channel names and corresponding SampleImage objects.
        cycles (set): cycle names. each cycle a substring of >=1 channel name
        mask (SampleImage): contains file info for segmentaiton mask
        dapi (SampleImage): contains file info for chosen DAPI file

    Attributes:
        name (str): patient (FOV) sample name.
        channels (list): list of channel names
        cycles (dict): cycle names and corresponding SampleCycle objects
        mask (SampleImage): SampleImage object for mask info
        dapi (SampleImage): SampleImage object for DAPI info

    """

    def __init__(self,
                 name: str,
                 images: Dict[str, SampleImage],
                 cycles: Set[str],
                 mask: SampleImage,
                 dapi: SampleImage):
        self.name = name
        self.mask = mask
        self.dapi = dapi
        self.channels = sorted(list(images.keys()))
        self.make_SampleCycles(cycles, images)
        self.set_reference()

    def make_SampleCycles(self, cycles, images):
        """ Take cycle names, and image dictionary to produce a dictionary of
        cycle_name: SampleCycle"""

        self.cycles = {}
        for c in cycles:
            cycle_images = {ch: images[ch] for ch in self.channels if c in ch}
            self.cycles.update({c: SampleCycle(c, cycle_images)})

        # Add self.dapi as a single-SampleImage cycle
        self.cycles.update({"dapi": SampleCycle(c, {"dapi": self.dapi})})

    def set_reference(self):
        """ Set reference cycle, chosen by MASK file """
        self.shifts = {}
        maskName = self.mask.imFile.split("/")[-1]
        refCycle = [i in maskName for i in self.cycles].index(True)
        refCycle = list(self.cycles)[refCycle]
        self.refCycleName = refCycle

        ch_idx = [i in refCycle for i in maskName.split("_")].index(True)
        self.refChannel = "_".join(maskName.split("_")[ch_idx:ch_idx+2])
        self.shifts[self.refCycleName] = [0, 0]

    def align(self, align_kws=None, parallel=False):
        """Calculates translations between cycles. Parallel processing optional

        All translations stored in self.shift dictionary. If image registration
        is difficult, SampleCycle.align() function may return [np.nan, np.nan],
        depending on align_kws.

        Args:
            align_kwargs (dict): aligment parameters used in SampleCycle.align
            parallel (bool): use parallel CPU processing"""
        # Alignment kwargs prepared on per-cycle basis. (DAPI typically more
        # stringer parameters)
        align_kws = self.setup_align_params(align_kws)

        # Set up generator for _align_parallel helper funciton
        cycle_order = sorted(self.cycles.keys())
        cycle_order.pop(cycle_order.index(self.refCycleName))
        args = ((self.cycles[self.refCycleName],
                 self.cycles[c].get_array(maxp=True, dtype=float),
                 c,
                 align_kws[c]) for c in cycle_order)
        map_fun = itertools.starmap
        if parallel:
            pool = multiprocessing.Pool()
            map_fun = pool.starmap

        # Store translations, accuracies in calculated_shifts, shift_accs
        calculated_shifts = {}
        shift_accs = {self.refCycleName: 0.80}
        for new_shift, shift_acc in map_fun(_align_parallel, args):
            calculated_shifts.update(new_shift)
            shift_accs.update(shift_acc)
        if parallel:
            pool.close()
            pool.join()

        self.update_alignments(calculated_shifts)
        return (self.get_shifts(), shift_accs)

    def get_channel_im(self, channel: str,
                       normalize=False, dtype=float) -> np.ndarray:
        """Return image for indicated channel name.
        Args:
            channel (str): channel name, must be in self.channels.
        Returns:
            np.ndarray: image array of size [H, W]
        """
        cycle = channel.split("_")[0]
        im = self.cycles[cycle].images.get(channel, None)
        if im is not None:
            return im.get_array(normalize=normalize, dtype=dtype)
        return

    def get_cycle_ims(self, cycle: str) -> np.ndarray:
        """Return images for indicated cycle name, stacked on axis=2.
        Args:
            cycle (str): cycle name, must be in self.cycles.
        Returns:
            np.ndarray: image array of size [H, W, n]. where n is the number
                of channels belonging to the indicated cycle.
        """
        cycleChannels = [ch for ch in self.channels if cycle in ch]
        im = [self.images[ch].get_array(dtype=float) for ch in cycleChannels]
        return np.stack(im, axis=2)

    def update_alignments(self, shifts: Dict[str, List]):
        """Add shifts by cycle to current shifts dictionary.
        Args:
            shifts (Dict): new shifts
        """
        self.shifts.update(shifts)

    def fix_shifts(self, replacements: Dict[str, List]):
        """Replace NaN shift values in cycle too difficult to register.

        Args:
            replacements (Dict[str, List]): Dictionary of whole-sample mean
                shifts to use as replacement for NaN values
        """
        for cycle in self.cycles:
            if np.isnan(self.shifts[cycle]).any():
                self.shifts[cycle] = replacements[cycle]

    def get_array(self, channel, dtype=None):
        return self.images[channel].get_array(dtype=dtype)

    def get_aligned_array(self) -> np.ndarray:
        """return aligned images as a single numpy array.

        Array will have shape (ref.shape[0], ref.shape[1], n_channels),
        where ref is the refernece channel, and n_channels is the number of
        imaged channels.
        """
        alignedData = []

        for cycle, samplecycle in self.cycles.items():
            shift_vals = self.shifts.get(cycle, [0, 0])
            shifted_arr = samplecycle.shift(shift_vals)
            alignedData.append(shifted_arr)
        alignedData = np.concatenate(alignedData, axis=2)
        return alignedData

    def get_shifts(self, by_channel: bool = False) -> Dict[str, List]:
        """Returns a dictionary with shift values for each channel.
        Args:
            by_cycle (bool): return shifts by cycle. Defaults to False.
        Returns:
            Dict[str, List]: dictionary of shifts per channel or cycle.
        """
        if not by_channel:
            return self.shifts
        return {ch: self.shifts[self.images[ch].cycle] for ch in self.channels}

    def set_shifts(self, shifts):
        self.shifts = shifts

    def setup_align_params(self, align_kwargs):
        # Create empty dictionaries to use default SampleImage.align() params
        if align_kwargs is None or align_kwargs == {}:
            align_kwargs = {c: {} for c in self.cycles}

        # If only one set of params given, copy to use same set of params
        # for every cycle
        elif not any([c in align_kwargs for c in self.cycles]):
            align_kwargs = {c: align_kwargs.copy() for c in self.cycles}

        # If params are given for some cycles, fill rest with defaults
        else:
            align_kwargs.update({c: {} for c in self.cycles
                                 if c not in align_kwargs})
        align_kwargs["dapi"]["returnNan"] = False
        return align_kwargs

    def check_dapi_empty(self):
        dapi_im = self.cycles["dapi"].get_array(maxp=True, dtype=float)
        im_mask, _ = self.cycles["dapi"].preprocess_ims(dapi_im, dapi_im,
                                                        threshold_triangle,
                                                        scale=0.8)
        return self.cycles["dapi"].check_if_empty(im_mask, im_mask)


class WholeSample:
    """ Data structure for complete patient sample. Mainly handles alignment
    among all fields of view in a sample. """

    def __init__(self,
                 name,
                 samples: List[Sample]):
        self.name = name
        self.samples = samples
        self.size = len(samples)
        self.cycleNames = sorted(self.samples[0].cycles.keys())

    def filter_cycles(self, new_list):
        new_list = [c for c in new_list if c in self.cycleNames]
        for i in range(self.size):
            older = self.samples[i].cycles
            refName = self.samples[i].refCycleName

            self.samples[i].cycles = {refName: older[refName]}
            self.samples[i].cycles.update({c: older[c] for c in new_list})

        self.cycleNames = sorted(self.samples[0].cycles.keys())

    def set_shifts(self, alignment_dict):
        for i in range(self.size):
            name = self.samples[i].name
            self.samples[i].set_shifts(alignment_dict[name])

    def clean_samples(self):
        """ Iterates through all DAPI files, and removes samples (FOVs) where
        there are too few DAPI (nucleus-sized) objects.

        Helps minimize computation time and removes samples with little
        potential to inform diagnosis/segmentation training"""
        for i in range(self.size-1, -1, -1):
            if self.samples[i].check_dapi_empty():
                logging.info("Skipped {} in {}".format(self.samples[i].name,
                                                       self.name))
                del(self.samples[i])
        self.size = len(self.samples)

    def align(self, params, parallel=True):
        """ Whole-sample alignment.
        Takes a random field of view from self.samples and produces alignment
        calculations. These determine alignment limits in all other FOVs.

        Args:
            params (dict): dictionary with alignment parameters used in
                SampleCycle.align()
        """
        self.clean_samples()

        test_i = np.random.randint(self.size)
        test_i = 3
        test_name = self.samples[test_i].name
        logging.info("Initial Alignment on: {}".format(test_name))

        st0 = time()
        print("(" + self.samples[test_i].name.split("_")[-1], end=", ")
        test_results = self.samples[test_i].align(params, parallel=parallel)
        print("{:.0f}s".format(time() - st0) + ")", end="\t")

        logging.info("{:.1f}s".format(time() - st0))
        logging.info(self.samples[test_i].get_shifts())

        new_defaults = {"trials": 2,
                        "maxAttempts": 8,
                        "fracIncrement": 0.2,
                        "accIncrement": 0.05}
        params_final = self.update_parameters(test_results, new_defaults)

        # Log results for evaluation
        str_params = [("{}:\t{}\n").format(c,
                                           "\n\t".join(["{}-> {}".format(k, v)
                                                        for k, v in kw.items()]
                                                       ))
                      for c, kw in params_final.items()]
        logging.info("".join(str_params))
        logging.info("Aligning remaining FOVs.")
        # Align the rest of the samples with updated thresholds for maximum
        # shift allowances. fracIncrement updated for speed
        for i in range(self.size):
            if i != test_i:
                st = time()
                logging.info("\t {}".format(self.samples[i].name))
                print("(" + self.samples[i].name.split("_")[-1], end=", ")
                self.samples[i].align(params_final, parallel=parallel)
                if time() - st0 > 120:
                    print("{:.1f}m".format((time() - st0)/60) + ")", end="\t")
                else:
                    print("{:.0f}s".format(time() - st0) + ")", end="\t")
                logging.info("{:.1f}s".format(time() - st))
                logging.info(self.samples[i].get_shifts())

    def update_NaNs(self):
        """ Fixes any NaN results produced in self.align(). Uses mean of all
        non-NaN fields of view """
        nanMeans = {}
        im_shifts = [self.samples[i].get_shifts() for i in range(self.size)]
        for cycle in self.cycleNames:
            nanMeans.update({cycle: self.get_sample_mean(im_shifts, cycle)})
        self.cycleMeans = nanMeans
        nanMeansTest = [np.isnan(nanMeans[c]).any() for c in self.cycleNames]

        if any(nanMeansTest):
            self.error = True
        else:
            self.error = False

    def fix_shifts(self):
        """ Fix NaN shifts in any sample using the calculated means """
        for i in range(self.size):
            self.samples[i].fix_shifts(self.cycleMeans)

    def update_alignments(self, old_alignments):
        for sample in self.samples:
            old_alignments[sample.name] = sample.get_shifts()
        return old_alignments

    def save_images(self, directory, return_data=True, maxp=False):
        """" Save samples to specified directory as numpy array """
        all_data = {}
        for sample in self.samples:
            # Segmentation Mask
            y = sample.mask.get_array(normalize=False, dtype=int)
            y -= y.min()  # Correct for single-valued image
            y = self.makelabel(y, border_size=3)

            # Normalized Max Projection and DAPI
            # get_aligned_array() will always return dapi as last channel
            x = sample.get_aligned_array()
            ordered_channels = [sc.channels for sc in sample.cycles.values()]
            ordered_channels = [ch for chs in ordered_channels for ch in chs]
            refIdx = ordered_channels.index(sample.refChannel)
            dapiIdx = ordered_channels.index(sample.dapi.name)
            if maxp:
                refIdx = ~dapiIdx

            cd45, dapi = x[..., refIdx], x[..., dapiIdx]
            dapi = (norm(dapi) * 2) - 1

            # Clip 99th and 25th percentiles, then sum across channels
            cd45 = norm(clip_percentiles(cd45))
            cd45 = (cd45 * 2) - 1
            x = np.stack([cd45, dapi], axis=2)

            temp = "\t{}\tGathered aligned projection and mask"
            logging.info(temp.format(sample.name))

            data = np.concatenate([x, y[..., np.newaxis]], axis=2)
            if return_data:
                all_data[sample.name] = data
            else:
                np.save(join(directory, sample.name) + ".npy", data)
        return all_data

    @staticmethod
    def update_parameters(results, defaults, shift_dev=10, acc_dev=0.05):
        shifts, accs = results
        cycles = sorted(shifts.keys())

        # We only update shift threshholds bc this parameter should be
        # independent, of image quality / sparsity / intensity
        params_updated = {}
        for c in cycles:
            params_updated[c] = defaults.copy()
            shift_max = np.max(np.abs(shifts[c]))
            params_updated[c]["shiftThresh"] = shift_max + 2 * shift_dev
            params_updated[c]["maxShiftThresh"] = shift_max + 2 * shift_dev
            params_updated[c]["shiftIncrement"] = 2 * shift_dev
            params_updated[c]["minAcc"] = accs[c] - acc_dev
        return params_updated

    @staticmethod
    def get_sample_mean(im_shifts: List[Dict], cycle: str) -> np.ndarray:
        """Using a shift values for all samples, return mean for given cycle.

        Args:
            im_shifts (List[Dict]): List of dictionaries.
            cycle (str): Cycle name.  must be a key in all im_shifts
        Returns:
            np.ndarray: mean shift value for cycle
        """
        shifts = np.stack([s[cycle] for s in im_shifts], axis=0)
        shifts_mean = np.nanmean(shifts, axis=0)
        return shifts_mean

    @staticmethod
    def makelabel(y, border_size=2):
        """ Create 3-class label for NN training purposes"""
        numCells = int(y.max())
        new_y = np.zeros_like(y).astype(np.int)

        for i in range(1, 1 + numCells):
            interior = (y == i)
            new_y[interior] = 1

            boundary = interior & ~binary_erosion(interior, disk(border_size))
            boundary = binary_closing(boundary, disk(3))
            new_y[boundary] = 2
        return new_y


def _align_parallel(refCycle: SampleCycle,
                    im: np.ndarray,
                    cycle: str,
                    align_kwargs: Dict) -> Dict[str, List]:
    """Helper function for parallel process alignment.
    Args:
        args (Tuple[SampleImage, np.ndarray, str]): function needs reference
            SampleImage, image array to align, and cycle name.
    Returns:
        Dict[str, List]: Dictionary of cycle name to shift values.
    """
    shift, acc = refCycle.align(im, **align_kwargs)
    return {cycle: shift}, {cycle: acc}


def norm(arr, axis=None):
    arr_min, arr_max = arr.min(axis=axis), arr.max(axis=axis)
    return (arr - arr_min) / (arr_max - arr_min)


def clip_percentiles(arr, low=0.25, high=0.85):
    p25 = np.percentile(arr, int(low*100), axis=(0, 1), keepdims=True)
    arr = np.clip(arr, p25, arr.max(axis=(0, 1), keepdims=True) * high)
    return arr


def load_cd45_dapi(filename):
    return np.load(filename).astype(np.float32)[..., :2]

###
# Code below used largely for alignment assessment.
# Will eventually be added to WholeSample.align() to automatically avoid saving
# imporoperly aligned images.
###


def load_ims(names, dataDir, parallel=True):
    H, W = np.load(join(dataDir, names[0]))[..., 0].shape

    file_gen = (join(dataDir, s) for s in names)

    if parallel:
        pool = multiprocessing.Pool(12)
        all_images = np.stack(pool.map(load_cd45_dapi, file_gen), axis=0)
        pool.close()
        pool.join()
    else:
        all_images = np.stack(map(load_cd45_dapi, file_gen), axis=0)
    all_images = (all_images + 1) / 2
    return all_images


def dapi_processing(dapi_pos, dapi_dil, slice_i, cutoff):
    dapi_pos = binary_opening(dapi_pos, disk(3))
    dapi_pos = binary_closing(dapi_pos, disk(3))
    dapi_pos = binary_opening(dapi_pos, disk(2))
    dapi_pos = binary_dilation(dapi_pos, disk(dapi_dil))

    areas = regionprops_table(label(dapi_pos), properties=('area',))["area"]
    areas = np.array(areas)
    to_remove = np.where((areas < cutoff[0]) | (areas > cutoff[1]))
    to_remove = to_remove[0] + 1
    np.logical_not(dapi_pos,
                   where=np.isin(label(dapi_pos), to_remove),
                   out=dapi_pos)
    return (dapi_pos, slice_i)


def cd45_processing(cd45_pos, cd45_dil, slice_i, cutoff):
    cd45_pos = binary_opening(cd45_pos, disk(3))
    cd45_pos = binary_closing(cd45_pos, disk(3))
    cd45_pos = binary_opening(cd45_pos, disk(2))
    if cd45_dil is not None:
        cd45_pos = binary_dilation(cd45_pos, disk(cd45_dil))

    areas = regionprops_table(label(cd45_pos), properties=('area',))["area"]
    areas = np.array(areas)
    to_remove = np.where((areas < cutoff[0]) | (areas > cutoff[1]))
    to_remove = to_remove[0] + 1
    np.logical_not(cd45_pos,
                   where=np.isin(label(cd45_pos), to_remove),
                   out=cd45_pos)
    return (cd45_pos, slice_i)


def get_quadrants(inputs,
                  cd45_f: float = 1.0,
                  dapi_f: float = 1.0,
                  cd45_dil: int = 1,
                  dapi_dil: int = 3,
                  cd45_cutoffs: list = [500, 70000],
                  dapi_cutoffs: list = [300, 7000],
                  parallel=True):
    cd45, dapi = inputs[..., 0], inputs[..., 1]

    if parallel:
        N = cd45.shape[0]
        pool = multiprocessing.Pool(12)
        gen = (cd45[i, ...][cd45[i, ...] > 0] for i in range(N))
        cd45_thresh = np.array(pool.map(threshold_triangle, gen))
        cd45_thresh = cd45_thresh[..., None, None]

        gen = (dapi[i, ...][dapi[i, ...] > 0] for i in range(N))
        dapi_thresh = np.array(pool.map(threshold_triangle, gen))
        dapi_thresh = dapi_thresh[..., None, None]
    else:
        cd45_thresh = threshold_triangle(cd45[cd45 > 0])
        dapi_thresh = threshold_triangle(dapi[dapi > 0])

    cd45_thresh = cd45_f * cd45_thresh
    dapi_thresh = dapi_f * dapi_thresh

    dapi = dapi > dapi_thresh
    cd45 = cd45 > cd45_thresh

    if parallel:
        dapi_pos_gen = ((dapi[i, ...], dapi_dil, i) for i in range(N))
        for result in pool.starmap(dapi_processing, dapi_pos_gen):
            dapi[result[1], ...] = result[0]

        cd45_pos_gen = ((cd45[i, ...], cd45_dil, i) for i in range(N))
        for result in pool.starmap(cd45_processing, cd45_pos_gen):
            cd45[result[1], ...] = result[0]
        pool.close()
        pool.join()
    else:
        cd45 = dapi_processing(cd45, cd45_dil, 0, cd45_cutoffs)[0]
        dapi = dapi_processing(dapi, dapi_dil, 0, dapi_cutoffs)[0]

    return (np.logical_and(dapi, cd45), np.logical_and(~dapi, cd45),
            np.logical_and(dapi, ~cd45), np.logical_and(~dapi, ~cd45))


def process_percentages(dapi_pos, cd45_pos, cutoff):
    dapi_pos = label(dapi_pos)
    N = dapi_pos.max()
    if N == 0:
        return (0.0, 0.0)
    intensities = np.zeros(N)
    for j in range(1, N + 1):
        object = cd45_pos[dapi_pos == j]
        intensities[j-1] = (object.astype(float).sum() / len(object)) > cutoff

    return (intensities.sum()/N, (dapi_pos > 0).astype(float).sum() * N)


def get_metrics(names, dataDir, kwargs, parallel=True):
    """ Get Metrics required for assessing alignment success"""

    cutoff = kwargs.pop("cutoff")
    if parallel:
        quadrants = get_quadrants(load_ims(names, dataDir, parallel=parallel),
                                  parallel=parallel,
                                  **kwargs)
        N = quadrants[0].shape[0]
        N_pixels = np.multiply(*quadrants[0].shape[1:])
        GT_gen = ((
                   np.logical_or(quadrants[0][i, ...], quadrants[2][i, ...]),
                   np.logical_or(quadrants[0][i, ...], quadrants[1][i, ...]),
                   cutoff
                   )
                  for i in range(N))
        pool = multiprocessing.Pool(12)
        results = pool.starmap(process_percentages, GT_gen)
        pool.close()
        pool.join()
        percentages = [r[0] for r in results]
        n_obj = [r[1] for r in results]

        metrics = [q.sum(axis=(1, 2)) for q in quadrants]
        metrics.append(np.array(percentages))
        metrics.append(np.array(n_obj))
        metrics = np.stack(metrics, axis=1)
    else:
        metrics = []
        for i in range(len(names)):
            s = names[i]
            print("{}/{}..".format(i+1, len(names)), end="\t")

            im_arr = (load_cd45_dapi(join(dataDir, s)) + 1)/2
            quadrants_i = get_quadrants(im_arr, parallel=parallel, **kwargs)
            results = process_percentages(
               np.logical_or(quadrants_i[0], quadrants_i[2]),
               np.logical_or(quadrants_i[0], quadrants_i[1]),
               cutoff
            )

            metrics_i = [q.sum() for q in quadrants_i]
            metrics_i.append(results[0])
            metrics_i.append(results[1])
            metrics.append(metrics_i)
        N_pixels = np.multiply(*quadrants_i[0].shape)
        metrics = np.stack(metrics, axis=0)

    # cd45_pos_dapi_pos, cd45_pos_dapi_neg,cd45_neg_dapi_pos,cd45_neg_dapi_neg
    metrics[:, :4] = metrics[:, :4] / (metrics[:, :4].max(axis=0) + 1e-9)
    metrics[:, 0] = 1 - metrics[:, 0]

    normalized_avg = metrics[:, :3].mean(axis=1) / metrics[:, 3]
    normalized_avg /= normalized_avg.max()

    # Final metric combines total "DAPI + space", and a normalized avg of
    # different quadrant scores.
    return metrics[:, 4], (metrics[:, 5]/N_pixels) + normalized_avg


def get_percentage(cd45_pos, dapi_pos, cutoff):
    """ Get Metrics required for assessing alignment success.
    im1, im2 should already be binary masks generated at start of alignment"""
    results = process_percentages(dapi_pos, cd45_pos, cutoff)
    return results[0]
