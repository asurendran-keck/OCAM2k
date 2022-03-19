############################## Import Libraries ###############################

## Math Library
import numpy as np
## Subprocess library
import subprocess
## System library
import sys
## Operating system library
import os
## PIL library used to read bitmap files
from PIL import Image as read_bmp
## Library used to plot graphics and show images
import matplotlib.pyplot as plt
## Time library 
import time
## Datatime Library
import datetime as dt
## import library used to manipulate fits files
from astropy.io import fits
## Library used to display np.array into ds9  
import pysao
## To be able to imshow in logscale
from matplotlib.colors import LogNorm
## Library used to over plotpolygone on imshow 
import matplotlib.patches as patches

############################## Local Definitions ##############################

# =============================================================================
def get_path(subdirname = ''):
    ''' -----------------------------------------------------------------------
    This function return the standard path associated to the subdirectory
    provided where FIU data should be saved.	
    Arguments:
     - subdirname[optional]: subdirectory of the main path.
                             -> must be a string.
                             -> / at the beginning and the end not required.
    Returns:
    Nominal case:
     - Path -- Path associated to the sub-directory name provided
     - Name -- Name of for the data based on time.
    In case of error:
     - False
    ----------------------------------------------------------------------- '''
    # Verify if a sub directory name have been provided by the user.
    if not subdirname == '':   
        # if yes verify if sub dir name provided is a string.
        if not isinstance(subdirname,str):
            print('\n\nERROR::: Provided sub directory name must be a string')
            return
        # If it is a string.
        # Check if provided sub directory name is valid. Correct if possible.
        else:
            # Remove blank subdirectories if present.
            while '//' in subdirname: subdirname = subdirname.replace('//','/')
            # if sub dir name provided has '/' at the beginning, strip it.
            if subdirname[0] == '/': subdirname = subdirname[1:]
            # if sub directory name provided has not '/' at the end, add it.
            if not subdirname[-1] == '/': subdirname += '/'
    
    # Format for time-based filename
    FNMTFORMAT = '%H_%M_%S.%f'
    # Format for UTC-named directories
    DIRTFORMAT = '%y%m%d'
    # Default path for saving images
    dfltPath   = '/home/aodev/Data/'
    # --- Check/create UTC date directory for Xenics data
    tmpdate   = dt.datetime.utcnow()
    # UTC date for dir
    timestamp = tmpdate.strftime(DIRTFORMAT)
    # Main path for save
    Path  = dfltPath + timestamp + '/' + subdirname
    # Get path in system 
    ospath    = os.path.dirname(Path)
    if not os.path.exists(ospath):
        # Path did not exist before; create it now
        os.makedirs(ospath)
    
    # Define a name based on time
    Name = tmpdate.strftime(FNMTFORMAT)
    
    return Path, Name

# =============================================================================
def moments(data, width = False):
    ''' -----------------------------------------------------------------------
    Compute the momentum of an np.array.
    Inputs:
     - data: a np.array
     - width: 
    Returns:
     - x 
     - y 
     if width == True:
     - width_x
     - width_y)
    ----------------------------------------------------------------------- '''
    total    = np.sum(data)
    X, Y     = np.indices(np.shape(data))
    cx       = np.sum(Y*data)/total
    cy       = np.sum(X*data)/total
    if width:
        row     = data[int(round(cx)), :]
        tmp     = np.sum(abs((np.arange(row.size)-cx)**2*row))/np.sum(row)
        width_x = 0.0 if tmp < 0. else np.sqrt(tmp)
        col     = data[:, int(round(cy))]
        tmp     = np.sum(abs((np.arange(col.size)-cy)**2*col))/np.sum(col)
        width_y = 0.0 if tmp < 0. else np.sqrt(tmp)
    else:
        width_x = 0
        width_y = 0
    # return
    return cx, cy, width_x, width_y

# =============================================================================
def get_image():
    ''' -----------------------------------------------------------------------
    Function used to get images from the EDT framegrabber. Images are reshaped
    before to be return as a numpy array of 16 bits un-signed integer "u16int".
    Inputs: 
     - None
    Returns:
     - frame    : OCAM2K image return as a numpy array "u16int".
     - time_st  : time stamp associated to the frame.
    ----------------------------------------------------------------------- '''
    frames   = 0
    # Location of the bitmap frames
    filename = '/home/aodev/Data/tmp/OCAM2K_frame.bmp'
    # Prepare command to pull an image
    command  = 'cd /opt/EDTpdv/; ./simple_take -c 0 -N 100 -b ' + filename
    # Pull an image
    trash    = subprocess.call(command, shell = True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) 
    # save a timestamp
    time_st  = time.time()   
    # Open the bitmap file and convert it into a numpy array
    # Each cell of the array can accomodate a 16 bits integer.
    array_08 = np.array(read_bmp.open(filename)).astype('uint16')
    # Remove prescan pixels (1 raw and 48 = 6*8 colums * 2x8 bits values = 96)
    # Reshape as a vector (2 x 8 bits values per pixel)
    #vect_08  = np.reshape(array_08[1:,96:],[240*240,2])
    # Combines 8 bits values into 16 bits values
    #array_16 = np.reshape(vect_08[:,1]*256 + vect_08[:,0],[120,60,8])
    # Reorganize pixel per amplifier
    #amp_1    =                     array_16[:,:,3]
    #amp_2    =           np.fliplr(array_16[:,:,2] )
    #amp_3    =                     array_16[:,:,1]
    #amp_4    =           np.fliplr(array_16[:,:,0] )
    #amp_5    = np.flipud(          array_16[:,:,4])
    #amp_6    = np.flipud(np.fliplr(array_16[:,:,5]))
    #amp_7    = np.flipud(          array_16[:,:,6])
    #amp_8    = np.flipud(np.fliplr(array_16[:,:,7]))
    # Reorganize pixel by side of the detector (upper lower)
    #frame_u  = np.concatenate([amp_4,amp_3,amp_2,amp_1],1)
    #frame_l  = np.concatenate([amp_8,amp_7,amp_6,amp_5],1)
    # Reconstruct the frame
    #frame   = np.concatenate([frame_u,frame_l],0)
    # Return frame
    #Alternate descrambling with txt file
    img16 = np.zeros((np.shape(array_08)[0], int(np.shape(array_08)[1] / 2)))
    for i in range(0, np.shape(array_08)[1], 2):
        img16[:, int (i / 2)] = (array_08[:, i + 1]<<8) + (array_08[:, i])                        # Convert pixels from 8 bit to 16 bit
    
    descrambler = np.loadtxt('ocam2_descrambling.txt', delimiter = ',').astype(int)
    img16_vector = img16.flatten()
    img_unscrambled_vector = np.zeros(np.shape(descrambler)[0])
    for i in range(0, np.shape(descrambler)[0]):
        img_unscrambled_vector[i] = img16_vector[descrambler[i]]
    frame = np.reshape(img_unscrambled_vector, (240, 240))
    return frame, time_st

############################ Image Acquisition Loop ###########################

# =============================================================================
def images_acquisition_loop(max_fps = 25):
    ''' -----------------------------------------------------------------------
    This function read the detector, reformat the images and store them into a
    shared memory.
    Inputs: 
     - max_fps  : maximum number of frame read per second (default value = 25)
    Returns:
     - None
    ----------------------------------------------------------------------- '''
    # Create the shm for the images.
    data     = np.zeros([240,240])
    shm_im   = shmlib.shm('/tmp/OCAM2K_raw_frames.im.shm',data)
    try:
        # Get an image and associated time stamp.
        im, ts = get_image()
        # Update shm.
        shm_raw_im.set_data(im)
        # Limit the fps to value provided by user
        time.sleep(1./max_fps)
    
    except KeyboardInterrupt:
        # Print message
        print('\n Sctipt interupted by user')

    # Print message
    print('\n OCAM2K image acquisition has been interupted.')

    # This function does not return anything
    return

# =============================================================================
def save_images(nb_im = 100, filename = 'tmp.fits', bkg_file = 'bkgd.fits'):
    ''' -----------------------------------------------------------------------
    Temporary function used to save images.
    ----------------------------------------------------------------------- '''
    time.sleep(2)
    # Get path where data must be saved
    path, default_name = get_path('OCAM2K')  
    # Prepare the filename for the images
    if filename == '':
        filename = default_name + '.fits'
    elif filename[-5:] != '.fits':
        filename += '.fits'

    # Try loading the background file (if any)
    bkg_filename = path + bkg_file
    try:
        bkgd = fits.open(bkg_filename)[0].data
        print('bkgd loaded')
    except:
        bkgd = np.zeros([240,240])
        print('bkgd NOT loaded')

    # Prepare a cube to store the images
    cube = np.zeros([nb_im,240,240])
    # Start image acquisition and display
    for i in np.arange(nb_im):
        # Get an image and store it into the cube of images.
        cube[i,:,:] = get_image()[0] - bkgd
        # Print usefull information for the user
        sys.stdout.write('\r Frame = %04.0d/%04.0d' %(i+1,nb_im))
        sys.stdout.flush()
    
    # Save the cube of images
    fullname = path + filename[:-5] + '_raw.fits'
    hdu      = fits.PrimaryHDU(cube)
    hdu.writeto(fullname, overwrite=True)
    print('\nRaw images saved: ' + fullname )
    # Save the median of the cube of image
    fullname = path + filename[:-5] + '_med.fits'
    hdu      = fits.PrimaryHDU(np.median(cube,0))
    hdu.writeto(fullname, overwrite=True)
    print('Cube median saved: ' + fullname + '\n')
    # This function return the path where images has been saved
    return path

# =============================================================================
def SHM(data, bkgd = '',path = '',disp = False, mask = ''):
    ''' -----------------------------------------------------------------------
    Temporary function use to determine PSF positions.
    ----------------------------------------------------------------------- '''
    # Get path if not provided bu user
    if path == '': path, _ = get_path('OCAM2K')   

    ## Open the data fits file    
    # Determine data fullname
    fullname = path + data
    # Check if file type provided
    if fullname[-5:] != '.fits': fullname += '.fits'
    # Open the data  
    data = fits.open(fullname)[0].data
    
    ## Open the bkgd fits file    
    # Check if user provide a file.
    # Case 1: no filename has been provided by user. 
    if bkgd == '': 
        # The background is an np array of zeros.
        bkgd = np.copy(data)*0.
    # Case 2: a filename has been provided by user.
    else:
        # Determine bkgd fullname
        fullname = path + bkgd
        # Check if file type provided
        if fullname[-5:] != '.fits': fullname += '.fits'
        # Open the data  
        bkgd = fits.open(fullname)[0].data    
    
    X,Y = np.meshgrid(np.arange(240),np.arange(240))
    if mask == 'center': 
        X,Y = np.meshgrid(np.arange(240),np.arange(240))
        X -= 120
        Y -= 120
    elif mask == 'BL': 
        X -= 64
        Y -= 64
    elif mask == 'UL': 
        X -= 64
        Y -= 176
    elif mask == 'BR': 
        X -= 180
        Y -= 60
    elif mask == 'UR': 
        X -= 180
        Y -= 180
    mask = np.ceil(np.sqrt(X**2+Y**2)/40)
    mask[np.where(mask != 1)] = 0


    # Compute the difference data - bkgd
    redu     = (data - bkgd)*mask

    limit    = 15
    redu_cp  = np.zeros([np.size(redu,0)+4,np.size(redu,1)+4]) 
    redu_cp[2:-2,2:-2] =  np.copy(redu)
    #redu_cp = np.copy(redu) 
    ite      = 0 
    
    while np.max(redu_cp) > limit:
        # Find the position of the maximum
        tmp_x,tmp_y = np.where(redu_cp == np.max(redu_cp))
        sub_im = redu_cp[tmp_x[0]-2:tmp_x[0]+3,tmp_y[0]-2:tmp_y[0]+3]
        opt_x,opt_y,wx,wy = moments(sub_im,False)
        
        redu_cp[tmp_x[0]-2:tmp_x[0]+3,tmp_y[0]-2:tmp_y[0]+3] = 0
        
        tmp_x = tmp_x[0] - (np.size(sub_im,0) - 1)/2. + opt_x - 2
        tmp_y = tmp_y[0] - (np.size(sub_im,1) - 1)/2. + opt_y - 2
        
        if not 'pts_list' in locals():
            pts_list = np.array([[tmp_x,tmp_y]])
        else:
            pts_list = np.concatenate((pts_list,[[tmp_x,tmp_y]]),0)
        
        ite += 1
        # Print usefull information for the user
        sys.stdout.write('\r nb_pts_found = %04.0d' %(ite))
        sys.stdout.flush()
    # plot images if requested
    if disp:
        # Create a figure
        fig_1 = plt.figure(num = 1)
        # Title of the fig
        plt.title('PSFs position', fontsize = 10)
        plt.subplot(111) 
        # Display redu image
        plt.imshow(redu, origin = 'lower')
        # Modify the axis: one ticks every 120 pixels
        X_ticks = np.arange(0, np.size(redu,1)+1, 120)
        plt.gca().set_xticks(X_ticks)
        plt.xticks(fontsize = 8)
        Y_ticks = np.arange(0, np.size(redu,0)+1, 120)
        plt.gca().set_yticks(Y_ticks)
        plt.yticks(fontsize = 8)
        # Label of the axis
        plt.xlabel('Pixel', fontsize = 8)
        plt.ylabel('Pixel', fontsize = 8)
        
        ax = fig_1.add_subplot(111)
        for k in np.arange(int(np.size(pts_list)/2.)):
            # PSF position in x direction (Standard orientation)
            sx = pts_list[k,1]
            # PSF position y direction (Standard orientation)
            sy = pts_list[k,0]
            # Prepare
            circ_0 = patches.Circle((sx,sy),radius=2. ,linestyle= '-',color='w',fill=False)
            # Draw a circle around the PSF
            ax.add_patch(circ_0)
        
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.85)
        # cax = plt.axes([left, bottom, Width, Hight])
        cax = plt.axes([0.83, 0.115, 0.03, 0.72])
        cbar = plt.colorbar(cax=cax)
        cbar.set_label('Flux (ADU)', rotation=90, fontsize = 8)
        cbar.ax.tick_params(labelsize=8, width=1)
        
        #plt.savefig(path + 'im_bottom_left.png', bbox_inches='tight', pad_inches=0.25, dpi=600)
        plt.show()
        
    return redu, redu_cp, pts_list

# =============================================================================
def dist(pts_list):
    ''' -----------------------------------------------------------------------
    Function use to compute distances between points of a list.
    ----------------------------------------------------------------------- '''
    # Compute the number of points
    nb_pts = np.size(pts_list)/2.
    #
    x = pts_list[:,int(0)]
    y = pts_list[:,int(1)]
    #

    nb_cpl = 0
    for i in np.arange(nb_pts):
        for j in (np.arange(nb_pts-(i+1))+(i+1)):
            nb_cpl += 1

    # Prepare value to return
    distances = np.zeros([int(nb_cpl),6])
    #
    index = 0
    for i in np.arange(nb_pts):
        for j in (np.arange(nb_pts-(i+1))+(i+1)):
            # 
            #print('i = %03d -- j = %03d' %(i,j))
            # Compute distance between the two selected points.
            dist_x = np.round(x[int(j)]-x[int(i)],2)
            dist_y = np.round(y[int(j)]-y[int(i)],2)
            dist_t = np.round(np.sqrt(dist_x**2 + dist_y**2),2)
            #
            distances[index,0] = i
            distances[index,1] = j
            distances[index,2] = dist_x
            distances[index,3] = dist_y
            distances[index,4] = dist_t
            distances[index,5] = dist_t/(j-i)
            #
            index += 1

    return distances
'''
# =============================================================================
def gap_size(bkgd = ''):
    # take images
    #_ = save_images()
    path = '/home/aodev/Data/210401/OCAM2K/'
    # Try to found all psf
    im, cp_im, pts_list = SHM('im_med.fits', 'bkgd_med.fits',path)
    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-120)**2 + (pts_list[:,1]-120)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)
    # Extract the XY coordinate of the points less than 40 pixels away from 
    # the center
    list_a = pts_list[cent_pts,:][0]

    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-40)**2 + (pts_list[:,1]-40)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)
    # Extract the XY coordinate of the points less than 40 pixels away from 
    # the center
    list_b = pts_list[cent_pts,:][0]

    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-200)**2 + (pts_list[:,1]-40)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)
    # Extract the XY coordinate of the points less than 40 pixels away from 
    # the center
    list_c = pts_list[cent_pts,:][0]

    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-200)**2 + (pts_list[:,1]-200)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)
    # Extract the XY coordinate of the points less than 40 pixels away from 
    # the center
    list_d = pts_list[cent_pts,:][0]

    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-40)**2 + (pts_list[:,1]-200)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)
    # Extract the XY coordinate of the points less than 40 pixels away from 
    # the center
    list_e = pts_list[cent_pts,:][0]

    plt.plot(list_a[:,0],list_a[:,1],'.r',list_b[:,0],list_b[:,1],'.g',list_c[:,0],list_c[:,1],'.b',list_d[:,0],list_d[:,1],'.k',list_e[:,0],list_e[:,1],'.m')
    plt.xlim([0,240])
    plt.ylim([0,240])
    plt.show()




    x = np.sort(tmp_list[:,0])
    y = np.sort(tmp_list[:,1])

    nb_pts   = int(np.size(x))

    index    = np.ones([1,1])
    nb_raw   = 0
    raw_nb   = 0
    pts_raw  = np.zeros([1,1])
    y_c      = np.copy(y)*0

    for i in np.arange(nb_pts):
        if np.abs(y[int(i-1)]-y[int(i)]) < 2:
            index[0,0] += 1
            if index > nb_raw:
                nb_raw = np.copy(index)
        else:
            pts_raw = np.concatenate([pts_raw,index],0)
            index = np.ones([1,1])
            raw_nb += 1
        y_c[i] = raw_nb -1

    pts_raw  = np.concatenate([pts_raw[2:],index],0)
    

    index    = np.ones([1,1])
    nb_col   = 0
    col_nb   = 0
    pts_col  = np.zeros([1,1])
    x_c      = np.copy(x)*0

    for i in np.arange(nb_pts):
        if np.abs(x[int(i-1)]-x[int(i)]) < 2:
            index[0,0] += 1
            if index > nb_col:
                nb_col = np.copy(index)
        else:
            pts_col = np.concatenate([pts_col,index],0)
            index = np.ones([1,1])
            col_nb +=1

        x_c[i] = col_nb - 1


    pts_col  = np.concatenate([pts_col[2:],index],0)

    y_1 = np.sort(tmp_list[:,1])
    x_1 = np.zeros(np.size(y_1))
    n_1 = np.zeros(np.size(y_1))

    for i in np.arange(nb_pts):
        for j in np.arange(nb_pts):
            if tmp_list[j,1] == y_1[i]:
                x_1[i] = tmp_list[j,0]
                n_1[i] = j

    x_2 = np.sort(tmp_list[:,0])
    y_2 = np.zeros(np.size(x_2))
    n_2 = np.zeros(np.size(x_2))

    for i in np.arange(nb_pts):
        for j in np.arange(nb_pts):
            if tmp_list[j,0] == x_2[i]:
                y_2[i] = tmp_list[j,1]
                n_2[i] = j

    table = np.zeros([int(nb_raw),int(nb_col),2])
    for i in np.arange(nb_pts):
        tmp_y   = y_2[int(np.where(n_2 == i)[0])]
        tmp_raw = x_c[int(np.where(n_2 == i)[0])]
        tmp_x   = x_1[int(np.where(n_1 == i)[0])]
        tmp_col = y_c[int(np.where(n_1 == i)[0])]
        table[int(tmp_raw),int(tmp_col)] = [tmp_x,tmp_y] 

    nb_dist = int((nb_pts**2-nb_pts)/2.)*2
    gap_size = np.zeros(nb_dist)
    index = 0
    for i in np.arange(int(nb_raw)):
        for j in np.arange(int(nb_col)):
            for k in np.arange(int(nb_raw)):
                for l in np.arange(int(nb_col)):
                    cdt_0 = (table[i,j,0] != 0)
                    cdt_1 = (table[i,j,1] != 0)
                    cdt_2 = (table[int(k),int(l),0] != 0)
                    cdt_3 = (table[int(k),int(l),1] != 0)
                    cdt_4 = (i != k) and (j != l)
                    if cdt_0 and cdt_1 and cdt_2 and cdt_3 and cdt_4:
                        dx  = table[i,j,0] - table[int(k),int(l),0]
                        dy  = table[i,j,1] - table[int(k),int(l),1]
                        pix = np.sqrt(dx**2 + dy**2)
                        dik = i-k
                        djl = j-l 
                        gap = np.sqrt(dik**2 + djl**2)
                        
                        gap_size[index] = pix/gap
                        index += 1

    gap_size = gap_size[np.where(gap_size != 0)[0]]
    gap_size = gap_size[:int(np.size(gap_size)/2.)]
    gsm = np.round(np.median(gap_size),3)

    # Create a figure
    fig_1 = plt.figure(num = 1)
    # Title of the fig
    plt.title('OCAM2K image (reduced)', fontsize = 10)
    # Display histogram
    plt.hist(gap_size,60,[gsm-0.30,gsm+0.30])
    # Modify the axis: one ticks every 128 pixels
    X_ticks = (np.arange(0, 61, 15)-30)*0.01
    plt.gca().set_xticks(X_ticks)
    plt.xticks(fontsize = 8)
    Y_ticks = np.arange(0, np.size(disp_im,0)+1, 30)
    plt.gca().set_yticks(Y_ticks)
    plt.yticks(fontsize = 8)
    # Label of the axis
    plt.xlabel('Pixel', fontsize = 8)
    plt.ylabel('Pixel', fontsize = 8)
    plt.show()
    print('size of the gaps = %5.3f' %(gsm))

    list_a_h = np.copy(list_a)    
    xh = np.sort(list_a[:,0])
    yh = np.zeros(np.size(y))
    for i in np.arange(np.size(x)):
        for j in np.arange(np.size(x)):
            if list_a[j,0] == x[i]:
                y[i] = list_a[j,1]

    list_a_r[:,0] = x
    list_a_r[:,1] = y

    index = 0
    nb_raw = 0
    for i in np.arange(np.size(y)/2.):
        if np.abs(y[int(i)]-y[int(i-1)]) < 3:
            index += 1
            if index > nb_raw:
                nb_raw = np.copy(index)
        else:
            index = 0


    # Compute distance betweel all PSF found and the center of the array
    dist2cent = np.sqrt((pts_list[:,0]-124)**2 + (pts_list[:,1]-124)**2)
    # Get the index of central circle of pts
    cent_pts  = np.where(dist2cent <= 40)

    # Isolate the PSF on the central vertical axis
    central_line = np.where(np.abs(pts_list[:,0]-pts_list[cent_pts,0]) <= 2)

    #redu_list = pts_list[central_line,:][0]
    redu_list = pts_list[cent_pts,:][0]
    # Sort the list of PSF by vertical position
    y = np.sort(redu_list[:,1])
    x = np.zeros(np.size(y))
    for i in np.arange(np.size(y)):
        for j in np.arange(np.size(y)):
            if redu_list[j,1] == y[i]:
                x[i] = redu_list[j,0]
    redu_list[:,0] = x
    redu_list[:,1] = y
    # Compute the distance between all the PSFs of the reduce list divided by 
    # the number of gaps between those PSFs 
    results = dist(redu_list)[5]
    
    sep_dir_V = np.round(np.median(results),2)
    
    plt.hist(results,50,[sep_dir_V-0.25,sep_dir_V+0.25]);plt.show()
    
    return sep_dir_V

#########################################################################################
path = '/home/aodev/Data/210330/OCAM2K/'
data = 'Set_03_med.fits'
bkgd = 'Bkgd_03_med.fits'
im,tmp,pts_list = SHM(data, bkgd, path)


central_line = np.where(np.abs(pts_list[:,0]-124) <= 2)
redu_list = pts_list[central_line,:][0]

y = np.sort(redu_list[:,1])
x = np.zeros(np.size(y))
for i in np.arange(np.size(y)):
    for j in np.arange(np.size(y)):
        if redu_list[j,1] == y[i]:
            x[i] = redu_list[j,0]

redu_list[:,0] = x
redu_list[:,1] = y

A = dist(redu_list)
#sep_dir_1 = np.max(A[:,4])/np.max(A[:,1]) 
sep_dir_1 = np.round(np.median(A[:,5]),2)
plt.hist(A[:,5],50,[sep_dir_1-0.25,sep_dir_1+0.25]);plt.show()


central_line = np.round(np.where(np.abs(pts_list[:,1]-124) <= 2),3)
redu_list = pts_list[central_line,:][0]

x = np.sort(redu_list[:,0])
y = np.zeros(np.size(x))
for i in np.arange(np.size(x)):
    for j in np.arange(np.size(x)):
        if redu_list[j,0] == x[i]:
            y[i] = redu_list[j,1]

redu_list[:,0] = x
redu_list[:,1] = y


B = dist(redu_list)
sep_dir_2 = np.round(np.max(B[:,4])/np.max(B[:,1]),3) 
np.median(B[:,5])
plt.hist(B[:,5],50,[sep_dir_1-0.25,sep_dir_1+0.25]);plt.show()

tmp = ((A[:,4]-(A[:,4]%4))/4)+np.round((A[:,4]%4)/4.)  



tmp = np.sqrt((pts_list[:,0]-10)**2 + (pts_list[:,1]-10)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Ax,Ay = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-10)**2 + (pts_list[:,1]-120)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Bx,By = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-10)**2 + (pts_list[:,1]-230)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Cx,Cy = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-120)**2 + (pts_list[:,1]-10)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Dx,Dy = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-120)**2 + (pts_list[:,1]-120)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Ex,Ey = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-120)**2 + (pts_list[:,1]-230)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Fx,Fy = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-230)**2 + (pts_list[:,1]-10)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Gx,Gy = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-230)**2 + (pts_list[:,1]-120)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Hx,Hy = pts_list[index,0],pts_list[index,1]

tmp = np.sqrt((pts_list[:,0]-230)**2 + (pts_list[:,1]-230)**2)
index = int(np.where(tmp == np.min(tmp))[0][0])
Ix,Iy = pts_list[index,0],pts_list[index,1]

    # Create a figure
    fig_1 = plt.figure(num = 1)
    # Title of the fig
    plt.title('OCAM2K image (reduced)', fontsize = 10)
    plt.subplot(111) 
    # Prepares the image for display
    disp_im = (np.abs(im)+1)
    disp_im /= np.max(disp_im)
    # The image (PSF + Calibration fibers) is show in log scale
    plt.imshow(disp_im,norm=LogNorm(vmin=1e-2, vmax=1), origin = 'lower')
    # Modify the axis: one ticks every 128 pixels
    X_ticks = np.arange(0, np.size(disp_im,1)+1, 30)
    plt.gca().set_xticks(X_ticks)
    plt.xticks(fontsize = 8)
    Y_ticks = np.arange(0, np.size(disp_im,0)+1, 30)
    plt.gca().set_yticks(Y_ticks)
    plt.yticks(fontsize = 8)
    # Label of the axis
    plt.xlabel('Pixel', fontsize = 8)
    plt.ylabel('Pixel', fontsize = 8)

    ax = fig_1.add_subplot(111)
    # Draw a circle around each pupils

    pup    = 'PSF_A'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((10,10),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(10,10+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Ax,Ay),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_B'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((10,120),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(10,120+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Bx,By),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_C'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((10,230),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(10,230+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Cx,Cy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_D'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((120,10),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(120,10+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Dx,Dy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_E'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((120,120),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(120,120+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Ex,Ey),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_F'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((120,230),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(120,230+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Fx,Fy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_G'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((230,10),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(230,10+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Gx,Gy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_H'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((230,120),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(230,120+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Hx,Hy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    pup    = 'PSF_I'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((230,230),radius=4.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(230,230+10,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)
    circ_0 = patches.Circle((Ix,Iy),radius=1.,color='r',fill=True,linewidth = 2,linestyle= '-',)
    ax.add_patch(circ_0)

    # Saves the image in PDF format file
    #plt.savefig(path + 'OCAM2K_Pupils.pdf', bbox_inches='tight', pad_inches=0.25, dpi=600)
    #plt.savefig(path + 'OCAM2K_Pupils.png', bbox_inches='tight', pad_inches=0.25, dpi=600)
    plt.show()


def plot_im():
    # Create a figure
    fig_1 = plt.figure(num = 1)
    # Title of the fig
    plt.title('OCAM2K image (reduced)', fontsize = 10)
    plt.subplot(111) 
    # Prepares the image for display
    disp_im = (np.abs(im)+1)
    disp_im /= np.max(disp_im)
    # The image (PSF + Calibration fibers) is show in log scale
    plt.imshow(disp_im,norm=LogNorm(vmin=1e-2, vmax=1), origin = 'lower')
    # Modify the axis: one ticks every 128 pixels
    X_ticks = np.arange(0, np.size(disp_im,1)+1, 30)
    plt.gca().set_xticks(X_ticks)
    plt.xticks(fontsize = 8)
    Y_ticks = np.arange(0, np.size(disp_im,0)+1, 30)
    plt.gca().set_yticks(Y_ticks)
    plt.yticks(fontsize = 8)
    # Label of the axis
    plt.xlabel('Pixel', fontsize = 8)
    plt.ylabel('Pixel', fontsize = 8)

    plt.show()

    ax = fig_1.add_subplot(111)
    # Draw a circle around each pupils

    cx,cy  = 40,40
    pup    = 'Bottom Left Pupil'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((cx,cy),radius=40.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(cx,cy+50,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)

    cx,cy  = 200,40
    pup    = 'Bottom Right Pupil'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((cx,cy),radius=40.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(cx,cy+50,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)

    cx,cy  = 40,200
    pup    = 'Top Left Pupil'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((cx,cy),radius=40.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(cx,cy-50,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)

    cx,cy  = 200,200
    pup    = 'Top Right Pupil'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((cx,cy),radius=40.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(cx,cy-50,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)

    cx,cy  = 120,120
    pup    = 'Central Pupil'
    bbox   = {'facecolor':'Black','alpha':1,'edgecolor':'none','pad':1}
    circ_0 = patches.Circle((cx,cy),radius=40.,color='w',fill=False,linewidth = 2,linestyle= '-',)
    plt.text(cx,cy+50,pup,color='w',fontsize=9,bbox=bbox, ha='center', va='center')
    ax.add_patch(circ_0)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.85)
    # cax = plt.axes([left, bottom, Width, Hight])
    cax = plt.axes([0.83, 0.115, 0.03, 0.72])
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Normalized Flux (Log scale)', rotation=90, fontsize = 8)
    cbar.ax.tick_params(labelsize=8, width=1)

    # Saves the image in PDF format file
    plt.savefig(path + 'OCAM2K_Pupils.pdf', bbox_inches='tight', pad_inches=0.25, dpi=600)
    plt.savefig(path + 'OCAM2K_Pupils.png', bbox_inches='tight', pad_inches=0.25, dpi=600)
    plt.show()
'''




# =============================================================================
def gap_size(images,background,path,mask):
    # Get path if not provided bu user
    if path == '': path, _ = get_path('OCAM2K')   
    # Try to found all psf
    im, cp_im, pts_list = SHM(images, background,path,mask = mask)
    # Sort the points by coordinates    
    x = np.sort(pts_list[:,0])
    y = np.sort(pts_list[:,1])
    # Compute the number of points
    nb_pts   = int(np.size(x))

    # Prepares values for index
    index    = np.ones([1,1])
    # Prepares a parameter for the number of rows in the image
    nb_row   = 0
    # Prepares a parameter for the row number
    row_nb   = 0
    # Prepares a list of pts per row
    pts_row  = np.zeros([1,1])
    # Parameter contains row nb associate to each point of the list
    y_c      = np.copy(y)*0
    # for each pts of the list of points
    for i in np.arange(nb_pts):
        # Case 1: Current point and previous point on the same row
        if np.abs(y[int(i-1)]-y[int(i)]) < 2:
            # Increment the index
            index[0,0] += 1
            # Update the nb of row if needed
            if index > nb_row:
                nb_row = np.copy(index)
        # Case 2: Current point and previous point not on the same row
        else:
            # Adds the current index (nb_pts in this row) to the list of pts 
            # per row
            pts_row = np.concatenate([pts_row,index],0)
            # Reset the index
            index = np.ones([1,1])
            # increment the number of row
            row_nb += 1
        # Associate a row number to the current point of the list
        y_c[i] = row_nb -1
    # Adds the last index to the number of points per row and remove the two 
    # first one (not valid due to loop construction)
    pts_row  = np.concatenate([pts_row[2:],index],0)
    
    # Prepares values for index
    index    = np.ones([1,1])
    # Prepares a parameter for the number of columns in the image
    nb_col   = 0
    # Prepares a parameter for the column number
    col_nb   = 0
    # Prepares a list of pts per column
    pts_col  = np.zeros([1,1])
    # Parameter contains row nb associate to each point of the list    
    x_c      = np.copy(x)*0

    # for each pts of the list of points
    for i in np.arange(nb_pts):
        # Case 1: Current point and previous point on the same column
        if np.abs(x[int(i-1)]-x[int(i)]) < 2:
            # Increment the index
            index[0,0] += 1
            # Update the nb of column if needed
            if index > nb_col:
                nb_col = np.copy(index)
        # Case 2: Current point and previous point not on the same column
        else:
            # Adds the current index (nb_pts in this column) to the list of pts
            # per column            
            pts_col = np.concatenate([pts_col,index],0)
            # Reset the index
            index = np.ones([1,1])
            # Increment the number of column
            col_nb +=1
        # Associate a column number to the current point of the list
        x_c[i] = col_nb - 1

    # Adds the last index to the number of points per column and remove the two 
    # first one (not valid due to loop construction)
    pts_col  = np.concatenate([pts_col[2:],index],0)

    y_1 = np.sort(pts_list[:,1])
    x_1 = np.zeros(np.size(y_1))
    n_1 = np.zeros(np.size(y_1))

    for i in np.arange(nb_pts):
        for j in np.arange(nb_pts):
            if pts_list[j,1] == y_1[i]:
                x_1[i] = pts_list[j,0]
                n_1[i] = j

    x_2 = np.sort(pts_list[:,0])
    y_2 = np.zeros(np.size(x_2))
    n_2 = np.zeros(np.size(x_2))

    for i in np.arange(nb_pts):
        for j in np.arange(nb_pts):
            if pts_list[j,0] == x_2[i]:
                y_2[i] = pts_list[j,1]
                n_2[i] = j

    table = np.zeros([int(nb_row),int(nb_col),2])
    for i in np.arange(nb_pts):
        try:
            tmp_y   = y_2[int(np.where(n_2 == i)[0])]
            tmp_row = x_c[int(np.where(n_2 == i)[0])]
            tmp_x   = x_1[int(np.where(n_1 == i)[0])]
            tmp_col = y_c[int(np.where(n_1 == i)[0])]
            table[int(tmp_row),int(tmp_col)] = [tmp_x,tmp_y]
        except:
            print('Missing pts') 

    nb_dist = int((nb_pts**2-nb_pts)/2.)*2
    gap_size = np.zeros(nb_dist)
    index = 0
    for i in np.arange(int(nb_row)):
        for j in np.arange(int(nb_col)):
            for k in np.arange(int(nb_row)):
                for l in np.arange(int(nb_col)):
                    cdt_0 = (table[i,j,0] != 0)
                    cdt_1 = (table[i,j,1] != 0)
                    cdt_2 = (table[int(k),int(l),0] != 0)
                    cdt_3 = (table[int(k),int(l),1] != 0)
                    cdt_4 = (i != k) and (j != l)
                    if cdt_0 and cdt_1 and cdt_2 and cdt_3 and cdt_4:
                        dx  = table[i,j,0] - table[int(k),int(l),0]
                        dy  = table[i,j,1] - table[int(k),int(l),1]
                        pix = np.sqrt(dx**2 + dy**2)
                        dik = i-k
                        djl = j-l 
                        gap = np.sqrt(dik**2 + djl**2)
                        
                        gap_size[index] = pix/gap
                        index += 1

    gap_size = gap_size[np.where(gap_size != 0)[0]]
    gap_size = gap_size[:int(np.size(gap_size)/2.)]
    gsm = np.round(np.median(gap_size),3)

    # Create a figure
    fig_1 = plt.figure(num = 1)
    # Title of the fig
    #plt.title('OCAM2K image (reduced)', fontsize = 10)
    # Prepare histogram
    plt.hist(gap_size,60,[gsm-0.30,gsm+0.30])
    # Label of the axis
    plt.xlabel('Average gap size in pixel', fontsize = 8)
    plt.ylabel('Number of spots set', fontsize = 8)
    plt.savefig(path + 'hist_upper_left.png', bbox_inches='tight', pad_inches=0.25, dpi=600)
    # Display histogram
    plt.show()
    # print number of pixel per gap
    print('size of the gaps = %5.3f' %(gsm))

    return gap_size
