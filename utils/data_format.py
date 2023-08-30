from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np

def transform_into_perspective(img, n_lenses, n_pix):
    '''Transforms a light field image into an image of perspective views'''
    perspective_img = np.zeros((n_lenses * n_pix, n_lenses * n_pix))
    n_lenses = 33
    n_pix = 17
    for lx in range(n_lenses):
        for ly in range(n_lenses):
            for i in range(n_pix):
                for j in range(n_pix):
                    lfx = lx * n_pix + i
                    lfy = ly * n_pix + j
                    psx = i *  n_lenses + lx
                    psy = j *  n_lenses + ly
                    perspective_img[psx, psy] = img[lfx, lfy]
    return perspective_img

def read_plot_vol_tiff(filename):
    '''Reads and plots a slice of the volume tiff'''
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 4 channels
    if image.shape[0] != 4:
        raise ValueError("The image does not have 4 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5))

    channel_names = ['birefringence', 'optic axis 1', 'optic axis 2', 'optic axis 3']
    axial = 23
    for i in range(4):
        axarr[i].imshow(image[i, axial, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()    
    
def read_plot_img_tiff(filename):
    '''Reads and plots a slice of a the image tiff'''
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 2 channels
    if image.shape[0] != 2:
        raise ValueError("The image does not have 2 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

    channel_names = ['retardance', 'orientation']
    for i in range(2):
        axarr[i].imshow(image[i, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()
