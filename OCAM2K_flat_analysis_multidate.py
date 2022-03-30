#!/usr/bin/env kpython3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import time
import sys

if __name__ == '__main__':
    dirrd = '/usr/local/aodev/Data/220127/OCAM2K/'   # Location of the flat field raw and median files
    dirflat = '/usr/local/aodev/Data/220127/OCAM2K/'
    suffix = ''
    gain = np.array([1, 10, 100, 300, 600]) # Array of gains at which the FF was taken
    fitswrite = 0                               # 1 for writing new FITS flatmaps and corrected FF, 0 to skip this step.
    channelnoise = 1                # Flag for displaying channel noise in FFC image
    rawindex = 400
    # light_med = np.zeros([np.size(gain), 240, 240])
    # light_snr_raw = np.zeros((500, np.size(gain)))
    # light_snr_postff = np.zeros((500, np.size(gain)))
    ph_noise = np.zeros(np.shape(gain)[0])
    for i in range(np.shape(gain)[0]):
        print('Reading raw fits files at gain ' + str(gain[i]))
        filename_medflat = dirflat + 'light_g' + str(gain[i]) + '_med.fits'
        if suffix == '':
            filename_med = dirrd + 'light_g' + str(gain[i]) + '_med.fits'
            filename_raw = dirrd + 'light_g' + str(gain[i]) + '_raw.fits'
        else:
            filename_med = dirrd + 'light_g' + str(gain[i]) + '_' + suffix + '_med.fits'
            filename_raw = dirrd + 'light_g' + str(gain[i]) + '_' + suffix + '_raw.fits'
        filename_flatmap = dirflat + 'light_g' + str(gain[i]) + '_fmap.fits'
        filename_ffc = dirrd + 'light_g' + str(gain[i]) + '_ffc.fits'
        # Import median flat file
        hdu_medflat = fits.open(filename_medflat)
        light_medflat = hdu_medflat[0].data
        hdu_medflat.close()
        # Import raw file
        hdu_raw = fits.open(filename_raw)
        light_raw = hdu_raw[0].data
        hdu_raw.close()
        # Import median file
        hdu_med = fits.open(filename_med)
        light_med = hdu_med[0].data
        hdu_med.close()
        # Compute normalized flats for each quadrant

        print('Computing FF and FFC images at gain ' + str(gain[i]))
        light_med_max = np.max(np.max(light_medflat))
        light_flat_map = light_medflat / np.max(np.max(light_medflat))
        light_flat_field = np.zeros(np.shape(light_raw))
        light_noise_postff = np.zeros(np.shape(light_raw)[0])
        # Compute flat field and total noise in the FFC image
        for k in range(np.shape(light_raw)[0]):
            light_flat_field[k,:,:] = light_raw[k,:,:] / light_flat_map
            light_noise_postff[k] = np.std(light_flat_field[k,:,:])
        # Compute noise in each channel of the FFC image
        aduffc_chnoise_raw = np.zeros((np.shape(light_raw)[0], 2, 4))
        aduffc_chnoise = np.zeros((2, 4))
        for k in range(0, 2):
            for j in range(0, 4):
                for p in range(0, np.shape(light_raw)[0]):
                    aduffc_chnoise_raw[p, k, j] = np.std(light_flat_field[p, k * 120:(k + 1) * 120, j * 60:(j + 1) * 60])
                aduffc_chnoise[k, j] = np.mean(aduffc_chnoise_raw[:, k, j])

        # Write to fits flatmap and FF corrected files
        if fitswrite == 1:
            print('Writing FF and FFC fits files for gain ' + str(gain[i]))
            hdu_fm = fits.PrimaryHDU(light_flat_map)
            hdu_fm.writeto(filename_flatmap, overwrite=True)
            hdu_ffc = fits.PrimaryHDU(light_flat_field)
            hdu_ffc.writeto(filename_ffc, overwrite=True)

        # Relative channel gain computation
        # 1 - Relative gain of the flats
        adu_flatch = np.zeros((2, 4))
        for k in range(0, 2):
            for j in range(0, 4):
                adu_flatch[k, j] = np.mean(light_medflat[k * 120:(k + 1) * 120, j * 60:(j + 1) * 60])
        relflatgain_ch = adu_flatch / np.max(np.max(adu_flatch))
        # 2 - Relative gain of the median image
        adu_ch = np.zeros((2, 4))
        for k in range(0, 2):
            for j in range(0, 4):
                adu_ch[k, j] = np.mean(light_med[k * 120:(k + 1) * 120, j * 60:(j + 1) * 60])
        relgain_ch = adu_ch / np.max(np.max(adu_ch))
        # 3 - Relative gain of the FFC image
        aduffc_ch = np.zeros((2, 4))
        for k in range(0, 2):
            for j in range(0, 4):
                aduffc_ch[k, j] = np.mean(light_flat_field[rawindex, k * 120:(k + 1) * 120, j * 60:(j + 1) * 60])
        relffcgain_ch = aduffc_ch / np.max(np.max(aduffc_ch))

        # Compute expected photon noise in each channel normalized by the gain (to be compared with measured noise computed as aduffc_chnoise)
        ph_mean_ch = np.zeros((2, 4))
        ph_noise_ch = np.zeros((2, 4))
        for k in range(0, 2):
            for j in range(0, 4):
                ph_mean_ch[k, j] = np.mean(light_med[k * 120:(k + 1) * 120, j * 60:(j + 1) * 60])
                if gain[i] != 1:
                    ph_noise_ch[k, j] = np.sqrt(2 * ph_mean_ch[k, j] * gain[i] * relgain_ch[k, j] / 27.665)
                    # print('Ph Noise: ' + str('{:.2f}'.format(ph_noise[i])))
                else:
                    ph_noise_ch[k, j] = np.sqrt(ph_mean_ch[k, j] * gain[i] * relgain_ch[k, j] / 27.665)
        # print('Mean of photons (before FF) at gain of ' + str(gain[i]) + ' is ' + str('{:.2f}'.format(ph_mean)))
        # print('Mean of photons (after FF) at gain of ' + str(gain[i]) + ' is ' + str('{:.2f}'.format(np.mean(np.mean(light_flat_field[k,:,:])))))
        ph_noise[i] = np.mean(ph_noise_ch)

        # Subplot 1 - Flat map
        plt.subplot(3, np.shape(gain)[0], i + 1)
        plt.imshow(light_medflat)
        plt.title('Flat map at gain = ' + str(gain[i]), fontsize=8)
        for k in range(0, 2):
            for j in range(0, 4):
                plt.text((j * 60) + 15, (k * 120) + 60, "{:.2f}".format(relflatgain_ch[k, j], fontsize=4))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize = 8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Subplot 2 - Median image
        plt.subplot(3, np.shape(gain)[0], np.shape(gain)[0] + i + 1)
        plt.imshow(light_med)
        plt.title('Median image at gain = ' + str(gain[i]), fontsize=8)
        for k in range(0, 2):
            for j in range(0, 4):
                plt.text((j * 60) + 15, (k * 120) + 60, "{:.2f}".format(relgain_ch[k, j], fontsize=4))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize = 8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Subplot 3 - FFC image
        plt.subplot(3, np.shape(gain)[0], (2 * np.shape(gain)[0]) + i + 1)
        plt.imshow(light_flat_field[rawindex,:,:])
        plt.title("FFC image sample at gain = " + str(gain[i]) + "\n Mean of "
            "Total Noise post FF = " + str('{:.2f}'.format(np.mean(light_noise_postff))) + " ADUs\n"
            "Expected photon noise = " + str('{:.2f}'.format(ph_noise[i])) + " ADUs", fontsize=8)
        for k in range(0, 2):
            for j in range(0, 4):
                if channelnoise == 0:
                    plt.text((j * 60), (k * 120) + 60, "{:.2f}".format(relffcgain_ch[k, j], fontsize=4))
                elif channelnoise == 1:
                    plt.text((j * 60), (k * 120) + 60, "{:.2f}".format(aduffc_chnoise[k, j], fontsize=4))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    # for i in range(np.shape(gain)[0]):
    #     mean_snr_raw = np.mean(light_snr_raw, axis=0)[i]
    #     mean_snr_postff = np.mean(light_snr_postff, axis=0)[i]
    #     print('SNR pre-FF at ' + str(gain[i]) + ' is ' + str("{:.2f}".format(mean_snr_raw)))
    #     print('SNR post-FF at ' + str(gain[i]) + ' is ' + str("{:.2f}".format(mean_snr_postff)))

        # plt.figure()
        # plt.plot(light_med[i, 100,:])

        # Subplot 3 - Noise histogram
        # num_bins = 20
        # ax = plt.subplot(3, np.shape(gain)[0], (2 * np.shape(gain)[0]) + i + 1)
        # n, bins, patches = plt.hist(light_flat_field[rawindex,:,:].flatten(), num_bins, facecolor='blue', alpha=0.5)
        # plt.plot(bins)
        # ax.set_xlim([np.min(bins), np.max(bins)])
        # plt.xlabel('ADUs', fontsize=8)
        # plt.ylabel('Number of pixels', fontsize=8)
        # plt.title('ADU distribution for FFC image at gain = ' + str(gain[i]), fontsize=8)
        # plt.xticks(fontsize=8)
        # plt.yticks(fontsize=8)

        # Test division of image by flat field without normalization
        # test_ffc = light_raw[rawindex,:,:] / light_med[i, :, :]
        # print("Mean of non-normalized FF correction is " + str('{:.2f}'.format(np.mean(np.mean(test_ffc)))))

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.show()
