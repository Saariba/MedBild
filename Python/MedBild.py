import os
import numpy as np
import SimpleITK as sitk
#import matplotlib.pyplot as plt
import cv2 as cv


# def sitk_show(img, title=None, margin=0.05, dpi=40, normarlize=False):  # Shows the Image in a Plot
#     nda = sitk.GetArrayFromImage(img)
#     cv.normalize(nda, nda, 0, 257, cv.NORM_MINMAX)   # Map Values 0-257
#     spacing = img.GetSpacing()
#     figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
#     extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
#
#     plt.set_cmap("gray")
#     ax.imshow(nda, extent=extent, interpolation=None)
#     print("max Value: " + str(np.max(nda)) + "min Value: " + str(np.min(nda)))
#     if title:
#         plt.title(title)
#     plt.show()


# def give_Dicom():
#     print("Move the Folders with the Patient file into this Directory")
#     patient = input("Enter the Patient Number (1-12)").lstrip()
#     if len(patient) == 1:
#         patient = '0' + patient
#     path = "./P" + patient + "/"  # reads the path of the Dicom files
#     files = os.listdir(path)
#     idxSlice = len(files)  # get the Number of Files in a Path (one Dicom Series)
#
#     reader = sitk.ImageSeriesReader()  # Loads the Dicom Reader
#     filenamesDICOM = reader.GetGDCMSeriesFileNames(path)  # Load the Dicom files into the Reader
#     reader.SetFileNames(filenamesDICOM)
#     imgOriginal = reader.Execute()
#
#     for x in range(0, idxSlice):  # iterate through all the files (0-idxSlice)
#         imgnew = imgOriginal[:, :, x]  # Display the current Picture
#         # sitk_show(imgnew, 'Picture' + str(x + 1))
#         # plt.pause(0.001)
#         user = input("Press [enter] to continue.")  # Pressing enter gives the next image
#         if user == 'end':  # Typing end stops the Programm
#             return imgnew
#         if user == 'this':  # Typing this opens the Image to work on it
#             return imgnew
#         plt.close()
#     return imgnew


def frame_image(image, low=0, high=100):
    image = sitk.GetArrayFromImage(image)

    for ix, iy in np.ndindex(image.shape):  # fensterung unten
        if low < image[ix, iy] < high:
            image[ix, iy] = 0
    return sitk.GetImageFromArray(image)


def give_data():                                #returns an array of arrays witht the Image data
    print("Move the Folders with the Patient file into this Directory")
    patient = input("Enter the Patient Number (1-12)").lstrip()
    if len(patient) == 1:
        patient = '0' + patient
    path = "./P" + patient + "/"  # reads the path of the Dicom files
    files = os.listdir(path)
    idxSlice = len(files)  # get the Number of Files in a Path (one Dicom Series)

    reader = sitk.ImageSeriesReader()  # Loads the Dicom Reader
    filenamesDICOM = reader.GetGDCMSeriesFileNames(path)  # Load the Dicom files into the Reader
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()
    imgarray = []
    for x in range(0, idxSlice):
        pic = sitk.GetArrayFromImage(imgOriginal[:, :, x])
        cv.normalize(pic, pic, 0, 257, cv.NORM_MINMAX)  # Werte zum Mappen/Skalieren (0-257)
        imgarray.append(pic)

    return imgarray


if __name__ == "__main__":
    img = give_data()

