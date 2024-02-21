import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from target_detection import simple_target_tracker, multitarget_tracker, find_centroid
from target_detection import CFAR_2D
import pickle
import cv2

c = 299792458
def scale_data(data: tuple, old_shape: tuple, new_shape: tuple) -> tuple:
    '''
    Interp 2D, sort of. 
    Takes old_shape (width, height) and scales data (x,y) to new_shape (width,height)
    '''
    alpha = old_shape[0] / new_shape[0]
    new_x = data[0] / alpha - new_shape[0] / 2

    beta = old_shape[1] / new_shape[1]
    new_y = new_shape[1] - (data[1] / beta)

    return (new_x, new_y)
    


def clean_raw_data(data: np.array, window_percentage = 2)-> np.array:
    print('Cleaning data')
    y,x = data.shape[0:2]
    data_new = data.copy() * 0.5
    height = window_percentage/100 * y
    y_start = int(y / 2 - height)
    y_end = int(y_start + 2 * height)
    x_start = 0
    x_end = x
    if len(data.shape) > 2:
        for z in range(data.shape[2]):
            data_new[y_start:y_end,x_start:x_end,z] = data[:,:,z].mean()
    else:
        data_new[y_start:y_end,x_start:x_end] = data.mean()
    return data_new

def threshold_data(data: np.array) -> np.array:
    binary = np.zeros_like(data)
    print("Applying thresholding")
    for i in range(data.shape[2]):
        _, binary[:,:,i] = cv2.threshold(data[:,:,i],4800,500000,cv2.THRESH_BINARY)
    return binary

def persistence(X, k, hold, decay):
    '''Add persistence (digital phosphor) effect along the time axis of 
    a sequence of range-doppler maps
    
    Parameters: 
    
    X: Input frame stack (NxMxL matrix)
    k: index of frame to acquire
    hold: number of samples to persist
    decay: frame decay rate (should be less than 1)
    
    Returns:

    frame: (NxM matrix) frame k of the original stack with persistence effect'''
    
    frame = np.zeros((X.shape[0], X.shape[1]))

    n_persistence_frames = min(k+1, hold)
    for i in range(n_persistence_frames):
        if k-i >= 0:
            frame = frame + X[:,:,k-i]*decay**i
    return frame


''' Simple Kalman filter based target tracker for a single passive radar
    target. Mostly intended as a simplified demonstration script. For better
    performance, use multitarget_kalman_tracker.py'''

def load_pickle(f: str) -> np.array:
    with open(f,'rb') as fin:
        return pickle.load(fin)

def parse_args():

    parser = argparse.ArgumentParser(
        description="PASSIVE RADAR TARGET TRACKER")
    
    parser.add_argument(
        'data_path',
        type=str,
        help="Path to raw radar data .pkl files")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    data_dir = Path(args.data_path)
    files = os.listdir(str(data_dir))
    files.sort()
    # Allocate memory
    # Open one file to peek the size of the arrays
    tmp = load_pickle(data_dir / files[0])
    raw_data = np.zeros((tmp.shape[0],tmp.shape[1],len(files)))
    print("Loading data files")
    for i,f in enumerate(files):
        fin = data_dir / f
        raw_data[:,:,i] = load_pickle(fin) * 255.0

    Nframes = raw_data.shape[2]

    max_range = 64
    max_speed = 1000
    center_freq = 488.31*10**6
    max_doppler_shift = ((max_speed / 3.6)* center_freq) / c
    print(f"Max doppler shift: {max_doppler_shift}")
    #history = simple_target_tracker(CF, max_range, max_doppler_shift)
    cleaned = clean_raw_data(raw_data,.4)
    print("Applying Kalman Filter...")
    binary = threshold_data(cleaned)
    history = simple_target_tracker(cleaned, max_range, max_doppler_shift)
    video_file_path = "tracker_binary.mp4"
    estimate = history['estimate']
    measurement = history['measurement']
    lockMode = history['lock_mode']

    unlocked = lockMode[:,0].astype(bool)    
    estimate_locked   = estimate.copy()
    estimate_locked[unlocked, 0] = np.nan
    estimate_locked[unlocked, 1] = np.nan
    estimate_unlocked = estimate.copy()
    estimate_unlocked[~unlocked, 0] = np.nan
    estimate_unlocked[~unlocked, 1] = np.nan

    
    framerate=30
    fig_width = 8
    fig_height = 4.5
    savedir = os.path.join(os.getcwd(),  "IMG")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    
    print("Rendering frames...")
    # loop over frames
    figure = plt.figure(figsize = (fig_width, fig_height))
    figure.canvas.draw()
    width, height, _ = np.array(figure.canvas.renderer.buffer_rgba(), dtype=np.uint8).shape
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(
        video_file_path,
        video_codec,
        framerate,
        (height, width),
    )
    video_out2 = cv2.VideoWriter(
        "binary.mp4",
        video_codec,
        framerate,
        (height, width),
    )
    for kk in tqdm(range(Nframes)):
        data = persistence(raw_data, kk, 20, 0.1)
        data = np.fliplr(data.T) # get the orientation right
        svname = os.path.join(savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')


        vmn = np.percentile(data.flatten(), 35)
        vmx = 1.5*np.percentile(data.flatten(),99)

        plt.imshow(data,
            cmap = 'gnuplot2', 
            vmin=vmn,
            vmax=vmx, 
            extent = [-1*max_doppler_shift,max_doppler_shift,0,max_range], 
            aspect='auto')

        if kk>3:
            nr = np.arange(kk)
            decay = np.flip(0.98**nr)
            col = np.ones((kk, 4))
            cc1 = col @ np.diag([0.2, 1.0, 0.7, 1.0])
            cc2 = col @ np.diag([1.0, 0.2, 0.3, 1.0])
            cc1[:,3] = decay
            cc2[:,3] = decay
            plt.scatter(estimate_locked[:kk,1], estimate_locked[:kk,0], 8,  marker='o', color='lime')

            #plt.scatter(estimate_unlocked[:kk,1], estimate_unlocked[:kk,0], 8,  marker='o', color='red')
        if binary[:,:,kk].max() != 0:
            c = find_centroid(np.fliplr(binary[:,:,kk].T))
            old_shape = binary[:,:,kk].shape[0:2]
            new_shape = (max_doppler_shift*2,max_range)
            centroid = scale_data(c,old_shape,new_shape)
            if centroid[0] is not None:
                plt.plot(centroid[0], centroid[1],marker='o',color='red')
        plt.xlim([-1*max_doppler_shift,max_doppler_shift])
        plt.ylim([0,max_range])

        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        plt.tight_layout()
        figure.canvas.draw()
        img_plot = np.array(figure.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        img_plot = img_plot[:,:,0:3]
        cv_frame = img_plot[:,:,::-1]
        video_out.write(cv_frame)
        bin = np.fliplr(binary[:,:,kk].T)
        # video_out2.write(bin)
        cv2.imwrite(svname,cv_frame)
        #print(f"Binary max: {binary[:,:,kk].max()}, Clean max: {cleaned[:,:,kk].max()}")
        figure.clf()
    video_out.release()
    video_out2.release()
    # maxes = []
    # for z in range(cleaned.shape[2]):
    #     maxes.append(cleaned[:,:,z].max())
    # plt.plot(maxes,'-')
    # plt.savefig('max_values.png',dpi=200)
    # plt.show()
