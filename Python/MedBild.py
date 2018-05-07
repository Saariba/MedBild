import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *


# def give_path():
#     print("Move the Folders with the Patient file into this Directory")
#     patient = input("Enter the Patient Number (1-12)").lstrip()
#     if len(patient) == 1:
#         patient = '0' + patient
#     path = "./P" + patient + "/"
#     return path


def make_mesh(image, threshold=-300, step_size=1):
    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces = measure.marching_cubes_classic(p, threshold)
    print("Done")
    return verts, faces


def plt_3d2(image, verts, faces, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))

    plt.show()
    print("done")


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    plt.show()
    print("Done")


def plot_3d(image, threshold=-300):
    print("Drawing...")
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# def scan_list(path):
#     files = os.listdir(path)
#     idxSlice = len(files)  # get the Number of Files in a Path (one Dicom Series)
#     imgarray = []
#
#     for x in range(0, idxSlice):
#         imgarray.append()

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
        x = '0' + x
    path = "./P" + x + "/"
    return path


output_path = working_path = "./Docs"


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def save_full_data(low, high):
    for x in range(low, high):
        patient = load_scan(data_path(str(x)))
        imgs = get_pixels_hu(patient)
        np.save(output_path + "fullimages_%d.npy" % (x), imgs)


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

    return image, new_spacing


if __name__ == "__main__":
    save_full_data(1, 12)  # EINMAL AUSFÜHREN, DANACH DIE ZEILE AUSKOMMENTIEREN
    # give_histogramm(5)
    id = 1
    imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))
    display_image_stack(imgs_to_process)
    patient = load_scan(data_path(str(1)))
    # give_histogramm()
    imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
    v, f = make_mesh(imgs_to_process, 350)

    # Nur eine von den Funktionen bneutzen, Debug nötig
    # plt_3d(v, f)
    # plotly_3d(v, f)
    # plt_3d2(imgs_after_resamp, v, f)
    # plot_3d(imgs_after_resamp)
