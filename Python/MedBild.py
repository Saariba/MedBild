import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


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


def give_histogramm(image):
    img = sitk.GetArrayFromImage(image).astype(np.float64)
    plt.hist(img.flatten(), bins=50, color='c')
    plt.title("Histogramm")
    plt.xlabel("Units")
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


if __name__ == "__main__":
    id = input("Input the Patient number and press enter")  # number of the patient
    display_image_stack(np.load(output_path + 'fullimages_{}.npy'.format(int(id))))
    image = load_scan(data_path(id))[int(input("Which Image do you want to see? 1-80"))]
    display_image(image)
    give_histogramm(image)
    # imgs_after_resamp = resample(imgs_to_process, patient, [1, 1, 1])
