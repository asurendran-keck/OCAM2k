#----------Program to unscramble the OCAM2K pixels taken with the EDT VisionLink F4 FG and compute the read noise/dark current---------
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set the number of total frames, fps, gain (EMCCD * amplifier) and total valid pixels
frames = 5000
fps = '100'                                                                             # fps = 'max' if maximum fps, actual number of fps otherwise
gain_total = 20
pixels_total = 57600
folder = '/home/aodev/asurendran/OCAM2k/unbinned/' + fps
descrambler = np.loadtxt('ocam2_descrambling.txt', delimiter = ',').astype(int)         # Source file from FLI for descrambling pixels

# Import data from bmp images taken using EDT program simple_take or take by giving the base filename as 'frame'
# The program first imports the bmp image, converts the 8 bit bmp to 16 bit numpy array and applies the descrambling operation to get the vector of 57600 16-bit pixels
img_unscrambled_vector = np.zeros((pixels_total, frames))
for k in range(0, frames):
    filename_raw = folder + '/frame_' + "{:0>3d}".format(k) + '.bmp'                    # Import bmp file
    im = Image.open(filename_raw)
    img = np.array(im)                                                                  # Convert image to numpy array
    img16 = np.zeros((np.shape(img)[0], int(np.shape(img)[1] / 2)))
    for i in range(0, np.shape(img)[1], 2):
        img16[:, int (i / 2)] = (img[:, i + 1]<<8) + (img[:, i])                        # Convert pixels from 8 bit to 16 bit
    img16_vector = img16.flatten()
    for i in range(0, np.shape(descrambler)[0]):
        img_unscrambled_vector[i, k] = img16_vector[descrambler[i]]                     # Descramble pixels to the 57600 pixel format

# Mean dark fram computation and display
img_unscrambled_vector_mean = np.mean(img_unscrambled_vector, axis = 1)
img_unscrambled_mean = np.reshape(img_unscrambled_vector_mean, (240, 240))
fig1 = plt.figure()
plt.imshow(img_unscrambled_mean)
plt.title('Mean Dark frame for ' + fps + ' fps over ' + str(frames) + ' frames')
plt.colorbar()
plt.show()
filename_out = 'img_darkf' + str(frames) + '_unbinned_' + fps + 'fps.png'
plt.savefig(filename_out, bbox_inches = 'tight')

# Read noise will be computed if fps = 'max', dark current will be computed if fps is a number
if fps == 'max':
    temp = np.zeros(pixels_total)
    read_noise = np.zeros(pixels_total)
    #--------Read noise computation begins-------------#
    for i in range(0, frames):
        temp = temp + ((img_unscrambled_vector[:, i] - img_unscrambled_vector_mean[:]) / gain_total) ** 2
    read_noise[:] = np.sqrt(temp[:] / frames)
    #--------Read noise computation ends-------------#
    read_noise_mean = np.mean(read_noise)
    fig2 = plt.figure()
    plt.imshow(np.reshape(read_noise, (240, 240)))
    plt.title('Read noise for each pixel at ' + fps + ' fps over ' + str(frames) + ' frames, Mean = ' + "{:.3f}".format(read_noise_mean) + 'e-')
    plt.colorbar()
    plt.show()
    filename_out = 'img_readnoisef' + str(frames) + '_unbinned_' + fps + '.png'
    plt.savefig(filename_out, bbox_inches = 'tight')
    np.savetxt('read_noise' + str(frames) + '_unbinned_' + fps + 'fps.txt', read_noise, delimiter = ',')
else:
    temp = np.zeros(pixels_total)
    read_noise = np.loadtxt('read_noise' + str(frames) + '_unbinned_maxfps.txt', delimiter = ',')       # Read noise should be computed already and saved in file for dark current computation
    var_pix = np.zeros(pixels_total)
    dark_pix = np.zeros(pixels_total)
    #--------Dark current computation begins-------------#
    for i in range(0, frames):
        temp = temp + ((img_unscrambled_vector[:, i] - img_unscrambled_vector_mean[:]) / gain_total) ** 2
    var_pix[:] = temp[:] / frames
    #--------Dark current computation ends-------------#
    fps_num = int(fps)
    dark_pix[:] = (var_pix[:] - (read_noise[:] ** 2)) * fps_num
    mean_dark_pix = np.mean(dark_pix)
    fig2 = plt.figure()
    plt.imshow(np.reshape(dark_pix, (240, 240)))
    plt.title('dark current for each pixel at ' + fps + ' fps over ' + str(frames) + ' frames, Mean = ' + "{:.3f}".format(mean_dark_pix))
    plt.colorbar()
    plt.show()
    filename_out = 'img_darkcurrentf' + str(frames) + '_unbinned_' + fps + '.png'
    plt.savefig(filename_out, bbox_inches = 'tight')
    np.savetxt('dark_currentf' + str(frames) + '_unbinned_' + fps + 'fps.txt', read_noise, delimiter = ',')
