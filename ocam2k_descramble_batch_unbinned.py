#----------Program to unscramble the OCAM2K pixels taken with the EDT VisionLink F4 FG and compute the read noise/dark current---------
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from astropy.io import fits
from astropy.io.fits import Header
import os

# Import data from bmp images taken using EDT program simple_take or take by giving the base filename as 'frame'
# The program first imports the bmp image, converts the 8 bit bmp to 16 bit numpy array and applies the descrambling operation to get the vector of 57600 16-bit pixels
# The ADU matrix are divided by the gain given in the test report to get the actual electron count
def unscrambleImage(k, folder, filename_base, gain_total, descrambler, pixels_total):
    img_unscrambled_vector = np.zeros(pixels_total)
    filename_raw = folder + '/' + filename_base + '_' + "{:0>3d}".format(k) + '.bmp'    # Import bmp file
    im = Image.open(filename_raw)
    img = np.array(im).astype('uint16')                                                                  # Convert image to numpy array
    img16 = np.zeros((np.shape(img)[0], int(np.shape(img)[1] / 2)))
    for i in range(0, np.shape(img)[1], 2):
        img16[:, int (i / 2)] = (img[:, i + 1]<<8) + (img[:, i])                        # Convert pixels from 8 bit to 16 bit
    img16_vector = img16.flatten()
    img_unscrambled_vector[:] = img16_vector[descrambler[:]]                            # Descramble pixels to the 57600 pixel format
    return img_unscrambled_vector / gain_total

# Set the number of total frames, fps, gain (EMCCD * amplifier) and total valid pixels
frames = 5000
# Array of FPS values for which the read noise and dark current is computed. Accuracy will be better for more FPS sampling points.
# The folder names of the frames should be the same as that of the FPS value
# fps = ['2067', '1000', '500', '333', '200', '100', '50', '33', '20', '10']
fps = ['2067', '1000', '500', '333', '200', '100']
gain_total = 27.665                                                                     # Total gain derived from OCAM2K test report. Product of EMCCD gain and amplifier gain
pixels_total = 57600
fits_write = 1                                                                          # Set to one if fits datacube of images (for every fps setting) has to be generated
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
descrambler = np.loadtxt('ocam2_descrambling.txt', delimiter = ',').astype(int)         # Source file from FLI for descrambling pixels
num_cores = multiprocessing.cpu_count()                                                 # Number of CPU cores used for processing the descrambling of the frames
var_pix = np.zeros((pixels_total, np.size(fps)))                                        # Variance of pixel count for each pixel at different exposure times
dark_current = np.zeros(pixels_total)                                                   # Dark current for each pixel
read_noise = np.zeros(pixels_total)                                                     # Read noise for each pixel
read_noise_sq = np.zeros(pixels_total)                                                  # Read noise squared for each pixel
read_noise_ch = np.zeros((2, 4))                                                        # Read noise for each of the 2x4 (8 in total) output channels
dark_current_ch = np.zeros((2, 4))                                                      # Dark current for each of the 2x4 (8 in total) output channels
expTime = np.zeros(np.size(fps))                                                        # Exposure time array (one for each FPS value)

# Loops over the different FPS folders, writes to FITS file (optional) and computes variance for each pixel at an FPS setting
for j in range(np.size(fps)):
    # Parsing bmp files acquired with EDT FG
    folder = '/home/aodev/asurendran/OCAM2k/2020-03-17/unbinned/' + fps[j]
    filename_base = 'frame'                                                             # This should be the same name as the base filename given during frame grab through EDT FG
    # Calls the descrambling function in parallel mode
    pixelCount_unscrambled = Parallel(n_jobs=num_cores)(delayed(unscrambleImage)(k, folder, filename_base, gain_total, descrambler, pixels_total) for k in range(0, frames))
    pixelCount_unscramblednp = np.asarray(pixelCount_unscrambled)
    print('Unscrambled ' + fps[j] + ' fps')

    # Convert bmp files into FITS datacube for each FPS setting
    if fits_write == 1:
        img = np.reshape(pixelCount_unscramblednp, (frames, 240, 240)).astype(np.int16)
        hdul = fits.HDUList(fits.PrimaryHDU(data = img))
        filename_out = 'img_darkf' + str(frames) + '_unbinned_' + str(fps[j]) + 'fps.fits'
        if os.path.exists(filename_out):
            os.remove(filename_out)
        hdul.writeto(filename_out)
        hdul.close()
        print('Wrote ' + fps[j] + ' fps FITS file')
    # Variance computation
    var_pix[:, j] = np.var(pixelCount_unscrambled, axis = 0) #CHECK
    expTime[j] = 1 / int(fps[j])

# Linear fitting exposure time with Pixel count variance to compute dark current and read noise
expTime_2d = np.transpose(np.reshape(np.hstack((expTime, np.ones(np.size(fps)))), (2, np.size(fps))))               # Horizontal stacking of exposure time with column of ones to be used in linear fitting
for i in range(0, pixels_total):
    dark_current[i], read_noise_sq[i] = np.linalg.lstsq(expTime_2d, var_pix[i, :], rcond=None)[0]                   # For each pixel
mean_dark_current, mean_read_noise_sq = np.linalg.lstsq(expTime_2d, np.mean(var_pix, axis = 0), rcond=None)[0]      # Mean read noise and dark current over all pixels and frames
read_noise = np.sqrt(read_noise_sq)                                                                                 # Read noise as the square root of the intercept of the linear fitting line
mean_read_noise = np.sqrt(mean_read_noise_sq)
# Dark current and read noise for each channel
for i in range(0, 2):
    for j in range(0, 4):
        read_noise_ch[i, j] = np.mean(np.reshape(read_noise, (240, 240))[i * 120:(i + 1) * 120, j * 60:(j + 1) * 60])
        dark_current_ch[i, j] = np.mean(np.reshape(dark_current, (240, 240))[i * 120:(i + 1) * 120, j * 60:(j + 1) * 60])

# Imshow read noise for all pixels
fig1 = plt.figure()
plt.imshow(np.reshape(read_noise, (240, 240)))
plt.title('Read noise over ' + str(frames) + ' frames, Mean = ' + "{:.3f}".format(mean_read_noise) + 'e-')
plt.colorbar()
for i in range(0, 2):
    for j in range(0, 4):
        plt.text((j * 60) + 30, (i * 120) + 60, "{:.3f}".format(read_noise_ch[i, j], fontdict=font))
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
filename_out = 'img_readnoisef' + str(frames) + '_unbinned.png'
plt.savefig(filename_out, bbox_inches = 'tight')
np.savetxt('read_noisef' + str(frames) + '_unbinned.txt', read_noise, delimiter = ',')

# Imshow dark current for all pixels
fig2 = plt.figure()
plt.imshow(np.reshape(dark_current, (240, 240)))
plt.title('Dark current over ' + str(frames) + ' frames, Mean = ' + "{:.3f}".format(mean_dark_current) + 'e-')
plt.colorbar()
for i in range(0, 2):
    for j in range(0, 4):
        plt.text((j * 60) + 30, (i * 120) + 60, "{:.3f}".format(dark_current_ch[i, j], fontdict=font))
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
filename_out = 'img_darkcurrentf' + str(frames) + '_unbinned.png'
plt.savefig(filename_out, bbox_inches = 'tight')
np.savetxt('dark_currentf' + str(frames) + '_unbinned.txt', dark_current, delimiter = ',')

# Plot data curve for variance vs exposure time and show the read noise^2 (intercept) and dark current (slope)
fig3 = plt.figure()
plt.plot(expTime, np.mean(var_pix, axis = 0), 'o')
plt.plot(np.insert(expTime, 0, 0), np.insert((mean_dark_current * expTime) + mean_read_noise_sq, 0, mean_read_noise_sq))
plt.legend(['OCAM2K data points', 'Fitted linear equation'])
plt.title('Average variance of pixel count vs exposure time, \nIntercept (${RON}^2$) = ' + "{:.3f}".format(mean_read_noise_sq) + '${e-}^2$ \nSlope (Dark current) = ' + "{:.3f}".format(mean_dark_current) + 'e-')
plt.xlabel('Exposure time (seconds)')
plt.ylabel('Variance of pixel count (e-)')
plt.gca().set_xlim(left = 0)
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
plt.show()
filename_out = 'img_variancef' + str(frames) + '_unbinned.png'
plt.savefig(filename_out, bbox_inches = 'tight')
