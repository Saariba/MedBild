from __future__ import print_function
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
import math
import cv2 as cv
import numpy_indexed as npi


class IndexTracker(object):

    def __init__(self, ax, X, cmap='gray'):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.cmap = cmap
        self.X = X
        self.slices = len(X)
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind], cmap=plt.get_cmap(cmap))
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
        self.Click = [math.floor(event.xdata), math.floor(event.ydata)]
        self.inx = self.ind
        help = self.X[self.inx]
        # rect = help[self.Click[1]:self.Click[1] + 10, self.Click[0]:self.Click[0] + 10]
        self.value = self.X[self.inx][math.floor(event.xdata), math.floor(event.ydata)]  # math.floor(rect.mean())
        print(self.value)
        print(self.Click)

    def update(self):
        self.im.set_data(self.X[self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


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


def dif(x, y):
    res = x - y
    if res < 0:
        res *= -1
    return res


# Handles everything around RegionGrowing as Validation and multi layer
# maxdif - maximal difference every pixel is allowed to differ from the original seed
# returns the segmented image stack
def RGHandler(imgstack, startindex, startseed, maxdif):
    seed = startseed
    index = startindex
    slidestack = imgstack.copy()  # to store the results
    filterstack = imgstack.copy()  # to store the median filtered images
    i = 0
    while (i < imgstack.__len__()):
        slidestack[i] = np.zeros_like(imgstack[i])
        i += 1

    # provide a sample of the median filtered image
    slidestack[0] = MedianFilterImage(imgstack[index], 8)

    # start with the first slide and the provided user input
    # compute the lowest point and the Average around the original seed on the original image to guess the spread better
    originalAV, rpoint = AverageMinValueEightNeigh(imgstack[index], seed)
    tresh = GetTreshholdRG(imgstack[index], seed, originalAV, rpoint)

    # actual compute region
    # apply a median filter on the image first
    filterstack[index] = MedianFilterImage(imgstack[index], 8)
    # perform the region growing on the filtered image and evaluate the time
    starttime = time.localtime(time.time())
    currslide = RegionGrowing(filterstack[index], seed, tresh, maxdif, originalAV)
    endtime = time.localtime(time.time())
    print("Zeit für RG:", endtime[5] - starttime[5], "s")
    # Validate first slide
    while (currslide[0, 0] == -1):
        currslide = RegionGrowing(filterstack[index], seed, tresh - 2, maxdif, originalAV)
    # add to finished slides
    slidestack[index] = currslide

    # Perform the same on the following slides
    cont = 1
    now = index + 9  # HERE: Stellt hier ein wie viele Slides er nach oben gehen soll
    while (index < now and cont):  # imgstack.__len__()-1 and cont):
        index += 1
        # Get the points we will use to compare the regions
        # if we have no points to compare the provided area was too small
        Line = GetComparisonLine(currslide)
        if (Line.__len__() == 0):
            cont = 0
        else:
            filterstack[index] = MedianFilterImage(imgstack[index], 8)  # filter the image
            # compute new seedpoint with the comparison line on the processed filtered image in comparison to the original image
            seed = GetNewSeed(filterstack[index - 1], imgstack[index], Line)
            # compute the average and the tresh on the original image
            originalAV, rpoint = AverageMinValueEightNeigh(imgstack[index], seed)
            tresh = GetTreshholdRG(imgstack[index], seed, originalAV, rpoint)
            print("togo:", (now - index))
            currslide = RegionGrowing(filterstack[index], seed, tresh, maxdif, originalAV)
            # validate the slide
            while (currslide[0, 0] == -1):
                currslide = RegionGrowing(filterstack[index], seed, tresh - 2, maxdif, originalAV)
            slidestack[index] = currslide

    return slidestack


def GetTreshholdRG(img, seed, originalAV, point):
    # compute intial treshhold
    # tresh is either the difference between the average surrounding 7x7 pixel or the difference between the seed and the surrounding minimum
    if (dif(img[seed[0], seed[1]], img[point[0], point[1]])) < dif(img[seed[0], seed[1]], originalAV):
        tresh = dif(img[seed[0], seed[1]],
                    originalAV) * 1  # HERE Ändert den Multiplikator, falls er zu viele/wenige Punkte akzeptiert
    else:
        tresh = dif(img[seed[0], seed[1]], img[
            point[0], point[1]]) * 1  # HERE Ändert den Multiplikator, falls er zu viele/wenige Punkte akzeptiert

    return tresh


# Computes the new Seedpoint
# input: Source Image and TargetImage for comparison along the comparison line
# out: returns the best fitting point as new seed point
def GetNewSeed(imgsource, imgtarget, comparline):
    compars = []
    # compute for every compare point the ratio between a point in the liver and a mirrored point (often) outside the liver
    # compute further the ratio of these ratios between the original image and the next image
    for l in comparline:
        intensS, y = AverageMinValueEightNeigh(imgsource, l)
        intensT, y = AverageMinValueEightNeigh(imgtarget, l)
        z = []
        z.append(l[1])
        z.append(l[0])
        intensSFlip, y = AverageMinValueEightNeigh(imgsource, z)
        intensTFlip, y = AverageMinValueEightNeigh(imgsource, z)

        difSFlip = dif(intensS, intensSFlip)
        difTFlip = dif(intensT, intensTFlip)
        difference = dif(difSFlip, difTFlip)
        compars.append(difference)
    # get the minimum distance between theses ratios == > next seed
    min = 1000
    index = -1
    runner = 0
    for c in compars:
        if (c < min):
            index = runner
            min = c
        runner += 1
    if (min > 100):
        return -1
    return comparline[index]


def get_Click_values(tracker):
    ClickCOx = tracker.Click[0]
    ClickCOy = tracker.Click[1]
    value = tracker.value
    index = tracker.inx
    return ClickCOx, ClickCOy, index, value


# returns a set of points that are nearly in the middle of the segmented liver
def GetComparisonLine(res):
    i = 4
    output = []
    while (i < res.shape[1] - 9):  # 9 to cut out possible points that are too close to the border
        i += 4
        up = 0
        down = res.shape[0] - 1
        cont = 1
        # Find the upper and lowest border point of the segment
        while (res[up, i] != 255 and cont):
            up += 1
            if (up >= res.shape[0] - 1):
                cont = 0
        while (res[down, i] != 255 and cont):
            down -= 1
            if (down == 0):
                cont = 0
        # interpolation
        if (cont):
            if (dif(up, down) > 20):
                inter = int((up + down) / 2)
                # assure interpolation is a valid segment
                flipper = -1
                counter = 1
                while (res[inter, i] != 255):
                    if flipper == -1:
                        flipper = 1
                    else:
                        flipper = -1
                        counter += 1
                    inter += counter * flipper
                if (res[inter, i] == 255):
                    output.append((inter, i))

    return output


# Performs a Median Filter on the image
# input: image as array; output: median filtered image as array
def MedianFilterImage(imag, kernel=5):
    img = sitk.GetImageFromArray(imag)
    medianFi = sitk.MedianImageFilter()
    medianFi.SetRadius(kernel)
    slide = medianFi.Execute(img)
    return sitk.GetArrayFromImage(slide)


# pure Region Growing Algo
# maxdif - maximal difference every pixel is allowed to differ from the original seed
# returns filtered mask
def RegionGrowing(img, seed, tresh, maxdif, seedAV):
    list = []  # points that are in need to be checked
    output = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    cont = 1
    starttime = time.localtime(time.time())
    processed = 0  # counts our pricessed points for failsafe

    while (len(list) > 0 and cont):
        current = list[0]
        output[current[0], current[1]] = 255

        for neighbour in eightNeighbours(current, img.shape):  # perform the search fpr correspondent pixel in 8N
            if (output[neighbour[0], neighbour[1]] == 0):
                dist = dif(img[neighbour[0], neighbour[1]], img[current[0], current[1]])
                distToOrgin = dif(img[neighbour[0], neighbour[1]], seedAV)
                # check whether the new pixel is around our current pixel and in the valuespace around our seed
                if (dist <= tresh and distToOrgin < maxdif):
                    output[neighbour[0], neighbour[1]] = 255
                    list.append(neighbour)
                else:
                    output[neighbour[0], neighbour[1]] = 1
                processed += 1
        list.pop(0)
        # interpolation based on the assumption that if the corners are added to the pixel, the pixel between them are too
        # if (output[current[0]+1,current[1]+1] == 255):
        #     if (output[current[0]+1,current[1]-1] == 255):
        #         output[current[0] + 1, current[1]] = 255
        #     if (output[current[0]-1,current[1]+1] == 255):
        #         output[current[0], current[1] + 1] == 255
        # if (output[current[0] - 1, current[1] - 1] == 255):
        #     if (output[current[0] - 1, current[1] + 1] == 255):
        #         output[current[0] - 1, current[1]] = 255
        #     if (output[current[0] + 1, current[1] - 1] == 255):
        #         output[current[0], current[1] - 1] == 255

        # failsafe case RegionGrowing is taking longer than 10s
        currtime = time.localtime(time.time())
        if (currtime[5] - starttime[5] >= 10):
            cont = 0
            print("RegionGrowing took to long- adjusting")
            output[0, 0] = -1

        # failsafe case RegionGrowing was too big: Liver is not bigger than 1/3 of the image
        if processed > ((img.shape[0] * img.shape[1]) / 3):
            print("tresh too high")
            cont = 0
            output[0, 0] = -1

    return output


# calculate the average value and minimal pixel difference of the pixels of a 7x7 kernel; closer ones are more weighted
def AverageMinValueEightNeigh(img, point):
    aver = 0
    min = 500
    rpoint = point

    for n in eightNeighbours(point, img.shape):
        for n2 in eightNeighbours(n, img.shape):
            for n3 in eightNeighbours(n2, img.shape):
                aver += img[n3[0], n3[1]]
                dist = dif(img[n3[0], n3[1]], img[point[0], point[1]])
                if (dist < min and n3 != point):
                    min = dist
                    rpoint = n3
    aver /= 512

    return aver, rpoint


def fourNeighbours(seed, shape):
    maximum = shape[0] - 1
    minimum = shape[1] - 1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0] + 1), seed[1]))
    List.append(((seed[0] - 1), seed[1]))
    List.append((seed[0], (seed[1] + 1)))
    List.append((seed[0], (seed[1] - 1)))

    return List


def eightNDiagonal(seed, shape):
    maximum = shape[0] - 1
    minimum = shape[1] - 1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0] + 1), seed[1] + 1))
    List.append(((seed[0] + 1), seed[1] - 1))
    List.append((seed[0] - 1, (seed[1] + 1)))
    List.append((seed[0] - 1, (seed[1] - 1)))

    return List


def eightNeighbours(seed, shape):
    maximum = shape[0] - 1
    minimum = shape[1] - 1
    List = []
    if (seed[0] == maximum or seed[1] == minimum):
        List.append(seed)
        return List
    List.append(((seed[0] + 1), seed[1]))
    List.append(((seed[0] - 1), seed[1]))
    List.append((seed[0], (seed[1] + 1)))
    List.append((seed[0], (seed[1] - 1)))
    List.append(((seed[0] + 1), seed[1] + 1))
    List.append(((seed[0] + 1), seed[1] - 1))
    List.append((seed[0] - 1, (seed[1] + 1)))
    List.append((seed[0] - 1, (seed[1] - 1)))

    return List


def dif_help(img, niter=1, kappa=50, gamma=0.125, step=(1., 1.), option=1, ploton=False):
    """


    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

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
    kernel = np.ones((5, 5), np.uint8)
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


if __name__ == "__main__":
    id = input("Input the Patient number and press enter")  # number of the patient
    files = load_scan(data_path(id))
    imtoArray(files)
    print(files[0].size)
    files = diffilter(files, 3)
    # res = []
    # for x in range(0, len(files)):
    #     files[x] = MedianFilterImage(files[x])
    # # files = MedianFilterImage(files)
    # files = morph_close(files)
    display_image_stack(files)
    fig, ax = plt.subplots(1, 1)
    tracker = imgscroll(files, fig, ax)
    x, y, index, value = get_Click_values(tracker)
    slices = RGHandler(files, 30, (180, 80), 100)
    slices = morph_close(slices)
    fig, ax = plt.subplots(1, 1)
    tracker2 = imgscroll(slices, fig, ax)
    display_image_stack(slices)
    plt.imshow(slices[0], cmap="gray")
    plt.show()
    plt.imshow(slices[30])
    plt.show()
    plt.imshow(slices[31])
    plt.show()
    plt.imshow(slices[32])
    plt.show()
