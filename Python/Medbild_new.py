from __future__ import print_function
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import cv2 as cv
import RegionGrowing as rg
import scipy.ndimage


class IndexTracker2(object):
    def __init__(self, X, Y, ax1, ax2, cmap="gray"):
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title('use scroll wheel to navigate')
        ax2.set_title('use scroll wheel to navigate')
        self.cmap = cmap
        self.X = X
        self.Y = Y
        self.slices = len(X)
        self.ind = self.slices // 2
        self.points = []
        self.im1 = ax1.imshow(self.X[self.ind], cmap=plt.get_cmap(cmap))
        self.im2 = ax2.imshow(self.Y[self.ind], cmap=plt.get_cmap(cmap))
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im2.set_data(self.Y[self.ind])
        self.im1.set_data(self.X[self.ind])
        ax1.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()


class IndexTracker(object):

    def __init__(self, ax, X, cmap='gray'):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.cmap = cmap
        self.X = X
        self.slices = len(X)
        self.ind = self.slices // 2
        self.points = []
        self.im = ax.imshow(self.X[self.ind], cmap=plt.get_cmap('gray'))
        self.update()
        self.Click = [0, 0]
        self.inx = 0
        self.value = 0
        print("Please click on the Center of the Liver")

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def onclick(self, event):
        print(math.floor(event.xdata), math.floor(event.ydata))
        self.Click = (math.floor(event.xdata), math.floor(event.ydata))
        self.inx = self.ind
        help = self.X[self.inx]
        # rect = help[self.Click[1]:self.Click[1] + 10, self.Click[0]:self.Click[0] + 10]
        self.value = self.X[self.inx][math.floor(event.xdata), math.floor(event.ydata)]  # math.floor(rect.mean())
        print(self.value)
        print(self.Click)
        self.points.append(self.Click)

    def update(self):
        help = scipy.ndimage.rotate(self.X[self.ind], -135)
        self.im.set_data(help)
        ax.set_ylabel('slice %s' % self.ind)
        # img = scipy.misc.lena()
        # tr = scipy.ndimage.rotate(img, 45)
        self.im.axes.figure.canvas.draw()


def data_path(x, seg=False):
    if seg:
        if len(x) == 1:
            x = '0' + x.lstrip()
        path = "./Ps" + x.lstrip() + '/'
        print(path)
        return path
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
    reader.SetOutputPixelType(sitk.sitkFloat32)
    imgOriginal = reader.Execute()
    imgarray = []
    for x in range(0, idxSlice):
        pic = imgOriginal[:, :, x]
        imgarray.append(pic)
    return imgarray


# def resize_image(img, x, y):
#     return cv.resize(img, )


def give_histogramm(id):
    file_used = output_path + "fullimages_%d.npy" % id
    img_to_process = np.load(file_used).astype(np.float64)

    plt.hist(img_to_process.flatten(), bins=50, color='c')
    plt.xlabel("Houndsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


def display_image_stack(stack, rows=6, cols=6, start_with=10, show_every=1):
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


def imtoArray(files):
    # files[0] = sitk.GetArrayFromImage(files[0])
    for x in range(0, len(files)):
        files[x] = sitk.GetArrayFromImage(files[x])
    return files


def get_Click_values(tracker):
    ClickCOx = tracker.Click[0]
    ClickCOy = tracker.Click[1]
    value = tracker.value
    index = tracker.inx
    points = tracker.points
    return points, index, value


# Performs a Median Filter on the image
# input: image as array; output: median filtered image as array
def MedianFilterImage(imag, kernel=9):
    img = sitk.GetImageFromArray(imag)
    medianFi = sitk.MedianImageFilter()
    medianFi.SetRadius(kernel)
    slide = medianFi.Execute(img)
    return sitk.GetArrayFromImage(slide)


def dif_help(img, niter=1, kappa=80, gamma=0.25, step=(0.5, 0.5), option=1, ploton=False):
    """
        Anisotropic diffusion.

        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)

        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration

        Returns:
                imgout   - diffused image.
    """

    # ...you could always diffuse each color channel independently if you
    # really want

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(0, niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout


def diffilter(imgStack, numberOfIterations=5):
    result = []
    for index in range(0, len(imgStack)):
        curr = imgStack[index]
        filterd = dif_help(curr, niter=numberOfIterations)
        result.append(filterd)
    return result


def imgscroll(files, fig, ax):
    # fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, files)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)
    plt.title("Patient" + str(id))
    plt.show()
    return tracker


def morph_close(imgStack, numberofiterations=1):  # first dilatate then erode
    kernel = np.ones((10, 10), np.uint8)
    result = imgStack
    for j in range(0, numberofiterations):
        for index in range(0, len(imgStack)):
            curr = result[index]
            filterd = cv.morphologyEx(curr, cv.MORPH_CLOSE, kernel)
            result[index] = filterd
    return result


def lower_res(imgStack, solutiondec=2):
    for x in range(0, len(imgStack)):
        imgStack[x] = imgStack[x].reshape(-1, solutiondec).mean(axis=1)
        # npi.group_by(imgStack[x][:, 1]).mean(imgStack[x])


def diff(imgstack, iterations=40, conductance=5):
    print("diffusion filtering:")
    for x in range(0, len(imgstack)):
        print(str(len(imgstack) - x) + " to go")
        image = imgstack[x]
        diff = sitk.CurvatureAnisotropicDiffusionImageFilter()
        diff.SetNumberOfIterations(iterations)
        timestep = diff.EstimateOptimalTimeStep(imgstack[x])
        diff.SetTimeStep(timestep)
        print(diff.GetConductanceScalingUpdateInterval())
        diff.SetConductanceScalingUpdateInterval(5)
        # print("timestep " + str(timestep))
        diff.SetConductanceParameter(conductance)
        imgstack[x] = diff.Execute(image)
        # print((int(len(imgstack)) - x))


def median(imgstack):
    for x in range(0, len(imgstack)):
        img = imgstack[x]
        med = sitk.MeanImageFilter()
        print(med.GetRadius())
        med.SetRadius((3, 3, 3))
        print(med.GetNumberOfThreads())
        imgstack[x] = med.Execute(img)


def dice_test(imgstack, segstack):
    dice = 0
    for x in range(0, len(imgstack)):
        img = imgstack[x]
        seg = segstack[x]

        dice += round((np.sum(np.where(img == seg))) * 2.0 / ((np.sum(img) + np.sum(seg))), 3)

        print
        'Dice similarity score is {}'.format(dice)
    dice = dice / len(imgstack)
    print("dice for the whole stack: " + str(dice))


# def myFilter(imgStack, kernel=3):
#     for x in range(imgStack):
#         img = imgStack[x]


def map_down(imgstack):
    for x in range(0, len(imgstack)):
        imgstack[x] = cv.normalize(imgstack[x], imgstack[x], 0, 3, cv.NORM_MINMAX)


def resize_image(imgstack):
    for x in range(0, len(imgstack)):
        img = imgstack[x]
        imgstack[x] = cv.resize(img, dsize=(math.trunc(img.shape[0] / 2), math.trunc(img.shape[1] / 2)),
                                interpolation=cv.INTER_LINEAR)


def arraytoIm(imgStack):
    for x in range(0, len(imgStack)):
        imgStack[x] = sitk.GetImageFromArray(imgStack[x])


def window_image(imgStack, lower=50, upper=200):
    for x in range(0, len(imgStack)):
        img = imgStack[x]
        img[img < lower] = 0
        img[img > upper] = 400
        imgStack[x] = img


if __name__ == "__main__":
    id = input("Input the Patient number and press enter")  # number of the patient
    files = load_scan(data_path(id))
    filesseg = load_scan(data_path(id, True))

    imtoArray(files)

    resize_image(files)
    arraytoIm(files)
    diff(files, iterations=100, conductance=10000)

    # median(files)
    imtoArray(files)
    window_image(files)
    for x in range(0, len(files)):
        print(str(len(files) - x) + " to go")
        files[x] = MedianFilterImage(files[x])
    imtoArray(filesseg)
    resize_image(filesseg)
    # morph_close(files)
    print(files[0].size)
    # resize_image(files)
    print(files[0].size)
    img = files[37]
    print(img[112, 39])
    seeds = []

    # res = []
    # print("median filtering: ")
    # for x in range(0, len(files)):
    #     print(str(len(files)-x)+ " to go")
    #     files[x] = MedianFilterImage(files[x])
    display_image_stack(files)
    fig, ax = plt.subplots(1, 1)
    tracker = imgscroll(files, fig, ax)
    points, index, value = get_Click_values(tracker)  # points = array with clicked points
    for x in range(0, len(points)):  # appending points
        seeds.append(points[x])
        # print(files[index][points[x[0]]][points[x[1]]])
    seeds.append((86, 35))
    seeds.append((125, 84))
    slices = rg.RGHandler(files, index, seeds, maxdif=40)
    slices = morph_close(slices)
    map_down(slices)  # used for dice calculation
    dice_test(slices, filesseg)  # dice calculation , not sure if works
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # show our segmentation next to GT data
    tracker = IndexTracker2(slices, filesseg, ax1, ax2)
    f.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.title("Patient" + str(id))
    plt.show()
