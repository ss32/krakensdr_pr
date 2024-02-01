import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from target_detection import simple_target_tracker
from target_detection import CFAR_2D
import pickle
import cv2

c = 299792458

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
    print("Applying CFAR filter...")
    # CFAR filter each frame using a 2D kernel
    #CF = np.zeros_like(data)
    cutoff = 150
    # CF = np.zeros((raw_data.shape[0],raw_data.shape[1],cutoff+1))
    # for i in tqdm(range(Nframes)):
    #     CF[:,:,i] = CFAR_2D(raw_data[:,:,i], 36, 4)
    #     if i==cutoff:
    #         break

    print("Applying Kalman Filter...")
    max_range = 64
    max_speed = 1000
    center_freq = 488.31*10**6
    max_doppler_shift = ((max_speed / 3.6)* center_freq) / c
    print(f"Max doppler shift: {max_doppler_shift}")
    #history = simple_target_tracker(CF, max_range, max_doppler_shift)
    history = simple_target_tracker(raw_data, max_range, max_doppler_shift)

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

    # if args.output == 'plot':
    #     plt.figure(figsize=(12,8))
    #     plt.plot(estimate_locked[:,1], estimate_locked[:,0], 'b', linewidth=3)
    #     plt.plot(estimate_unlocked[:,1], estimate_unlocked[:,0], c='r', linewidth=1, alpha=0.3)
    #     plt.xlabel('Doppler Shift (Hz)')
    #     plt.ylabel('Bistatic Range (km)')
    #     plt.show()

    # else:
    video_file_path = "TRACKER.mp4"
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
        (width, height),
    )
    for kk in tqdm(range(Nframes)):

        # add a digital phosphor effect to make targets easier to see
        #data = persistence(CF, kk, 20, 0.90)
        data = persistence(raw_data, kk, 20, 0.1)
        data = np.fliplr(data.T) # get the orientation right
        #data = np.flipud(data)
            # get the save name for this frame
        svname = os.path.join(savedir, 'img_' + "{0:0=3d}".format(kk) + '.png')
        # make a figure

    

        # get max and min values for the color map (this is ad hoc, change as
        # u see fit)
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
            plt.scatter(estimate_unlocked[:kk,1], estimate_unlocked[:kk,0], 8,  marker='o', color='mediumspringgreen')

        plt.xlim([-1*max_doppler_shift,max_doppler_shift])
        plt.ylim([0,max_range])

        plt.ylabel('Bistatic Range (km)')
        plt.xlabel('Doppler Shift (Hz)')
        plt.tight_layout()
        # plt.gca().invert_yaxis()
        figure.canvas.draw()
        img_plot = np.array(figure.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        img_plot = img_plot[:,:,0:3]
        video_out.write(img_plot[:,:,::-1])
        plt.savefig(svname, dpi=100)
        # plt.close()
        figure.clf()
        if kk == cutoff:
            break
    video_out.release()

