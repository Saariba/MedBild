from __future__ import print_function
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols = X[0].shape
        self.slices = len(X)
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def data_path(x):
    if len(x) == 1:
        x = '0' + x.lstrip()
    path = "./P" + x.lstrip() + "/"
    return path


output_path = working_path = "./Docs"


def load_scan(path):
    files = os.listdir(path)
    idxSlice = len(files)  # get the Number of Files in a Path (one Dicom Series)

    reader = sitk.ImageSeriesReader()  # Loads the Dicom Reader
    filenamesDICOM = reader.GetGDCMSeriesFileNames(path)  # Load the Dicom files into the Reader
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()
    imgarray = []
    for x in range(0, idxSlice):
        pic = imgOriginal[:, :, x]
        imgarray.append(pic)
    return imgarray


def give_histogramm(id):
    file_used = output_path + "fullimages_%d.npy" % id
    img_to_process = np.load(file_used).astype(np.float64)

    plt.hist(img_to_process.flatten(), bins=50, color='c')
    plt.xlabel("Houndsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


def display_image_stack(stack, rows=6, cols=6, start_with=10, show_every=2):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every - 1
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def display_image(img, title=None, margin=0.05, dpi=40):  # Shows the Image in a Plot
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)
    print("max Value: " + str(np.max(nda)) + "min Value: " + str(np.min(nda)))
    if title:
        plt.title(title)
    plt.show()


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    # spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    print("Resampling image, putting the Layers Together")
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]],
                       dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    print("done")

    return image


def sample3d(imgstack):
    # res = np.dstack((A, B))
    # res = np.dstack((res,C))
    # np.save(output_path + "fullimage3d_%d.npy" % (1), res)
    res = np.load(output_path + 'fullimage3d_1.npy')
    # res = np.dstack(imgstack)
    # for element in range(2, len(imgstack)):
    #     res = np.dstack((res, imgstack[element]))
    #     print("Finished" + str(element))
    # np.swapaxes(res, 0, 1)
    return res


def sample3d_save():
    start = time.clock()
    for x in range(1, 12):
        patient = load_scan(data_path(str(x)))
        res = np.dstack(patient)
        np.save(output_path + "fullimage3d_%d.npy" % x, res)
        print("Done " + str(x))
    print(time.clock() - start, "Seconds")


def imtoArray(files):
    files[0] = sitk.GetArrayFromImage(files[0])
    for x in range(1, len(files)):
        files[x] = sitk.GetArrayFromImage(files[x])
    return files


if __name__ == "__main__":
    id = input("Input the Patient number and press enter")  # number of the patient
    files = load_scan(data_path(id))
    # display_image_stack(np.load(output_path + 'fullimages_{}.npy'.format(int(id))))
    # display_image(files[int(input("Which Image do you want to see? 1-80"))])
    # # imgs_after_resamp = resample(imgs_to_process, patient, [1, 1, 1])
    # test = sample3d(files)
    imtoArray(files)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, files)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
