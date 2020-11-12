from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import cv2
import io
from skimage import exposure
from scipy.ndimage.morphology import binary_fill_holes
import copy

class PPImage:

    def __init__(self, image_path = ""):
        if image_path != "":
            self.open(image_path)

    def from_zip(self, zip_file, file, folder = ''):
        archive = zipfile.ZipFile(zip_file, 'r')
        imgdata = archive.read(folder+file)
        imgdata = io.BytesIO(imgdata)

        self.open(imgdata, True)

    def open(self, path, resize_im=False):
        self.image = Image.open(path)

        if resize_im:
            basewidth = 1000
            img = self.image
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            self.image = img.resize((basewidth, hsize))

        self.data = np.array(self.image)
        # self.get_stats()

    def show(self, img = ""):
        if isinstance(img, str): img = self.data
        if len(img.shape) == 2: plt.imshow(img, plt.cm.gray)
        else: plt.imshow(img)
        plt.show()

    def mean_subtract(self):
        pass

    def plot_hist(self):
        ch = self.data.shape[2]

        # fig = plt.figure("Comparison")
        for i in range(ch):
            dt = self.data[:,:,i]
            # ax = fig.add_subplot(ch, 1, i+1)
            plt.hist(dt, bins=10)
            plt.show()

    def export(self, path, img='', quality='avg'):
        if isinstance(img, str): img = self.data
        if quality == 'good':
            Image.fromarray(img).save(path, subsampling=0, quality=100)
        else: Image.fromarray(img).save(path)

    def grayscale(self, img='', type='avg'):
        if isinstance(img, str): img = self.data
        if type =='max':
            return np.max(img, axis=2)
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def thresh(self, thresh=20):
        gray = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
        black_loc = gray < thresh
        self.data[black_loc] = 0

    def pad(self):
        self.data = cv2.copyMakeBorder(self.data, 350, 350, 350, 350, cv2.BORDER_CONSTANT, value=0)

    def white_balance(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def compare(self, img1 = None, img2 = None):
        if (img1 is None): img1 = np.array(self.image)
        if (img2 is None): img2 = self.data
        fig = plt.figure("Comparison")

        ax = fig.add_subplot(1, 3, 1)
        if len(img1.shape)==2: plt.imshow(np.array(self.image), plt.cm.gray)
        else: plt.imshow(np.array(self.image))

        ax = fig.add_subplot(1, 3, 2)
        if len(img1.shape)==2: plt.imshow(img1, plt.cm.gray)
        else: plt.imshow(img1)

        ax = fig.add_subplot(1, 3, 3)
        if len(img2.shape)==2: plt.imshow(img2, plt.cm.gray)
        else: plt.imshow(img2)


        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig.canvas.draw()
        fig.tight_layout()

        plt.show()

    def get_stats(self):
        ch = self.data.shape[2]
        self.stats = []
        for i in range(ch):
            st = {}
            st['mean'] = np.mean(self.data[:,:,i])
            self.stats.append(st)

    def color_correct(self):
        p2, p98 = np.percentile(self.data, (2, 98))
        self.data = exposure.rescale_intensity(self.data, in_range=(p2, p98))

    def hist_equalize(self, img=''):
        if isinstance(img, str): img = self.data
        img = copy.copy(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ch = img.shape[2]
        for i in range(ch):
            img[:, :, i] = clahe.apply(img[:, :, i])
        return img

    def hist_match_rgb(self, target, exclude_backgroud = True, background_tol = 15):
        ch = self.data.shape[2]
        thresh = -1
        if exclude_backgroud:
            thresh = background_tol

        source_bin = self.binary(self.data, thresh)
        source_targ = self.binary(target, thresh)
        # self.compare(source_bin, source_targ)


        for i in range(ch):
            source = self.data[:,:,i][source_bin > 0]
            template = target[:,:, i][source_targ > 0]
            oldshape = source.shape
            source = source.ravel()
            template = template.ravel()

            # get the set of unique pixel values and their corresponding indices and
            # counts
            s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                    return_counts=True)
            t_values, t_counts = np.unique(template, return_counts=True)

            # take the cumsum of the counts and normalize by the number of pixels to
            # get the empirical cumulative distribution functions for the source and
            # template images (maps pixel value --> quantile)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]

            # interpolate linearly to find the pixel values in the template image
            # that correspond most closely to the quantiles in the source image
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
            self.data[:, :, i][source_bin > 0] = interp_t_values[bin_idx].reshape(oldshape)
        print()

    def crop_image_only_outside(self, img, mask_im='', tol=10):
        # img is 2D or 3D image data
        # tol  is tolerance
        bin = self.binary(img, tol=tol)
        # bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2RGB)
        col = bin < tol
        img[col] = 0
        # self.show(bin)
        # exit()
        mask = img > 0
        if img.ndim == 3:
            mask = mask.all(2)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax() - 1
        row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax() - 1
        img = img[row_start:row_end, col_start:col_end]
        if isinstance(mask_im, str) == False:
            mask_im = mask_im[row_start:row_end, col_start:col_end]
            return img, mask_im
        else:
            return img


    def binary(self, img="", tol=15):
        if isinstance(img, str): img = self.data
        gray = self.grayscale(img, type='max')
        gray[gray < tol] = 0
        gray[gray > tol] = 255
        gray = binary_fill_holes(gray).astype(np.uint8)*255
        return gray

    def resize(self, img="", dim=""):
        if isinstance(img, str): img = self.data
        if isinstance(dim, str):
            return img
        new_im = np.array(Image.fromarray(img).resize(dim))
        return new_im
