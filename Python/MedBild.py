import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

plt.ion()


def sitk_show(img, title=None, margin=0.05, dpi=40):  # Shows the Image in a Plot
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)
    plt.show()


def give_Dicom():
    path = "./test/"  # reads the path of the Dicom files
    files = os.listdir(path)
    idxSlice = len(files)  # get the Number of Files in a Path (one Dicom Series)

    reader = sitk.ImageSeriesReader()  # Loads the Dicom Reader
    filenamesDICOM = reader.GetGDCMSeriesFileNames(path)  # Load the Dicom files into the Reader
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()  # Starts the Reader

    for x in range(0, idxSlice):                            # iterate through all the files (0-idxSlice)
        imgnew = imgOriginal[:, :, x]                       # Display the current Picture
        sitk_show(imgnew, 'Picture' + str(x + 1))
        plt.pause(0.001)
        user = input("Press [enter] to continue.")  # Pressing enter gives the next image
        if user == 'end':  # Typing end stops the Programm
            break
        if user == 'this':  # Typing this opens the Image to work on it
            return imgnew
        plt.close()
    return imgnew


image = give_Dicom()
plt.ioff()
sitk_show(image)
