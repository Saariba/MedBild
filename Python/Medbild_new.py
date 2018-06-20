from __future__ import print_function
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import cv2 as cv
import RegionGrowing as rg
import scipy.ndimage


class IndexTracker3(object):
    def __init__(self, X, Y, Z, ax1, ax2, ax3, cmap="gray"):
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title('use scroll wheel to navigate')
        ax2.set_title('use scroll wheel to navigate')
        ax3.set_title('use scroll wheel to navigate')
        self.cmap = cmap
        self.X = X
        self.Y = Y
        self.Z = Z
        self.slices = len(X)
        self.ind = self.slices // 2
        self.points = []
        self.im1 = ax1.imshow(self.X[self.ind], cmap=plt.get_cmap(cmap))
        self.im2 = ax2.imshow(self.Y[self.ind], cmap=plt.get_cmap(cmap))
        self.im3 = ax3.imshow(self.Z[self.ind], cmap=plt.get_cmap(cmap))
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
        self.im3.set_data(self.Z[self.ind])
        ax1.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()
        self.im3.axes.figure.canvas.draw()


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
        self.Click = (math.floor(event.ydata), math.floor(event.xdata))
        self.inx = self.ind
        help = self.X[self.inx]
        # rect = help[self.Click[1]:self.Click[1] + 10, self.Click[0]:self.Click[0] + 10]
        self.value = self.X[self.inx][math.floor(event.ydata), math.floor(event.xdata)]  # math.floor(rect.mean())
        print(self.value)
        print(self.Click)
        self.points.append(self.Click)

    def update(self):
        help = scipy.ndimage.rotate(self.X[self.ind], -135)
        self.im.set_data(self.X[self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        # img = scipy.misc.lena()
        # tr = scipy.ndimage.rotate(img, 45)
        self.im.axes.figure.canvas.draw()


def data_path(x, seg=False):
    if seg:
        # if int(x) > 6:
        #     (x) = x + 1
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


def imgscroll(data, fig, ax):
    # fig, ax = plt.subplots(1, 1)
    track = IndexTracker(ax, data)
    fig.canvas.mpl_connect('scroll_event', track.onscroll)
    fig.canvas.mpl_connect('button_press_event', track.onclick)
    plt.title("Patient" + str(id))
    # plt.gca().invert_yaxis()
    plt.show()
    return track


def morph_close(imgStack, numberofiterations=1, kernelsize=10):  # first dilatate then erode
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
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
        diff.SetConductanceScalingUpdateInterval(1)
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
        img[img < lower - 200] = 0
        img[img > upper - 90] = img[img > upper - 90] + 100
        imgStack[x] = img


def window_image_seg(imgStack, lower=3, upper=200):
    for x in range(0, len(imgStack)):
        img = imgStack[x]
        img[img < lower] = 0
        imgStack[x] = img


def get_dice(imgStack, segStack):
    arraytoIm(imgStack)
    arraytoIm(segStack)
    cast_image(imgStack)
    cast_image(segStack)
    respic = []
    res = 0
    x = 0
    for i in range(len(imgStack)):
        print('do stuff: ' + str(i))
        img = sitk.Cast(segStack[i], sitk.sitkLabelUInt8)
        segimg = sitk.Cast(imgStack[i], sitk.sitkUInt8)

        overlayimg = sitk.LabelMapContourOverlay(img, segimg)
        respic.append(overlayimg)

        overlapfilter = sitk.LabelOverlapMeasuresImageFilter()
        overlapfilter.Execute(sitk.Cast(img, sitk.sitkUInt8), segimg)
        result = overlapfilter.GetDiceCoefficient()
        if 0 < result < 2:
            print(result)
            res = res + result
            x = x + 1
    print('can not evaluate dice in: ' + str(len(imgStack) - x) + ' slices')
    print(res / x)
    return respic


def cast_image(imgstack):
    for i in range(len(imgstack)):
        cast = sitk.CastImageFilter()
        cast.SetOutputPixelType(sitk.sitkInt16)
        imgstack[i] = cast.Execute(imgstack[i])


def mydice(imgstack, segstack):
    for i in range(len(imgstack)):
        img = imgstack[i]
        seg = segstack[i]
        imgs = np.isin(img, 3)
        print(imgs)
        segs = np.isin(seg, 3)
        print(segs)
        imgsize = list(imgs.flatten()).count(True)
        segsize = list(segs.flatten()).count(True)
        a = np.where(np.isclose(img,3))
        print(a)
        b = np.where(np.isclose(seg,3))
        print(b)
        a = a[0]
        b = b[0]
        print(a)
        print(b)
        union = np.sum(a,b)
        dice = union / (imgsize + segsize)
        print(dice)


if __name__ == "__main__":
    a = [5, 8, 9, 9, 5, 5]
    index = np.where(np.isclose(a,9))
    print(index)
    id = input("Input the Patient number and press enter")  # number of the patient
    files = load_scan(data_path(id))
    filesseg = load_scan(data_path(id, True))
    imtoArray(files)
    resize_image(files)
    orig = sitk.GetImageFromArray(files)
    orig = sitk.GetArrayFromImage(orig)
    arraytoIm(files)
    diff(files, iterations=30, conductance=25)
    imtoArray(files)
    imtoArray(filesseg)
    resize_image(filesseg)
    # get_dice(files, filesseg)
    seeds = []

    display_image_stack(files)
    fig, ax = plt.subplots(1, 1)
    tracker = imgscroll(files, fig, ax)
    points, index, value = get_Click_values(tracker)  # points = array with clicked points
    x = points[0][0]
    y = points[0][1]
    window_image(files, files[index][x][y], files[index][x][y])
    fig, ax = plt.subplots(1, 1)
    tracker = imgscroll(files, fig, ax)
    for x in range(0, len(points)):  # appending points
        seeds.append(points[x])
    slices = rg.RGHandler(files, index, seeds, maxdif=100, takeborders=False)
    slices = morph_close(slices, 1, 10)

    window_image_seg(filesseg)
    for i in range(len(slices)):
        img = slices[i]
    img[img > 20] = 3
    slices[i] = img
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)  # show our segmentation next to GT data
    tracker = IndexTracker3(orig, slices, filesseg, ax1, ax2, ax3)
    f.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.title("Patient" + str(id))
    plt.show()
    # mydice(slices, filesseg)
    respic = get_dice(slices, filesseg)
    imtoArray(respic)
    fig, ax = plt.subplots(1, 1)
    tracker = imgscroll(respic, fig, ax)
