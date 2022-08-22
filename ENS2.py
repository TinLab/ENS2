# -*- conding: UTF-8 -*-
# version: 0.3.0
# edited by: Zhou Zhanhong 


import os
import sys
import time
import datetime
import glob as glob
import copy
from collections import OrderedDict
from tqdm.auto import trange, tqdm

import argparse
import random
import math

import numpy as np
import scipy
import scipy.signal
import scipy.io as scio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from pytorchtools import EarlyStopping


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


opt = argparse.ArgumentParser()

opt.sampling_rate = 60
opt.smoothing_std = 0.025
opt.smoothing = opt.smoothing_std * opt.sampling_rate

opt.causal_kernel = False
opt.gaussian_kernel = not opt.causal_kernel

opt.signal_len = 96
opt.classes = 6

opt.lr = 0.001
opt.epochs = 5000
opt.patience = 500
opt.batch_size = 1024
opt.sample_interval = opt.epochs

def load_all_ground_truth(sampling_rate):
    ground_truth_folder='./ground_truth/'
    datasets = glob.glob(os.path.join(ground_truth_folder, 'DS*'))
    dataset_trial = dict()
    dataset_neuron = dict()
    dataset_prop = dict()
    for j, dataset in enumerate(datasets):
        files = glob.glob(os.path.join(dataset, '*'))
        recording_trial_list = list()
        recording_neuron_list = list()
        recording_sample_count = 0
        recording_raw_fs = []
        recording_noise_level = []
        recording_firing_rate = []
        for i, file in enumerate(files):
            try:
                recording_neuron, recording_trial, count = load_recordings_from_file(file, sampling_rate)
                recording_neuron_list.extend(recording_neuron)
                recording_trial_list.extend(recording_trial)
                recording_sample_count += count
                recording_raw_fs.append(recording_neuron[0]['frame_rate'])
                recording_noise_level.append(recording_neuron[0]['noise_level'])
                recording_firing_rate.append(recording_neuron[0]['firing_rate'])
            except:
                print('Problem loading file {} from {}'.format(i+1, dataset))
                pass
        dataset_neuron[j+1] = recording_neuron_list
        dataset_trial[j+1] = recording_trial_list
        dataset_prop[j+1] = dict()
        dataset_prop[j+1]['name'] = os.path.basename(dataset)
        dataset_prop[j+1]['dset'] = j+1
        dataset_prop[j+1]['frame_rate'] = np.mean(recording_raw_fs)
        dataset_prop[j+1]['noise_level'] = np.mean(recording_noise_level)
        dataset_prop[j+1]['firing_rate'] = np.mean(recording_firing_rate)
        dataset_prop[j+1]['neuron_number'] = len(recording_neuron_list)
        dataset_prop[j+1]['trial_number'] = len(recording_trial_list)
        dataset_prop[j+1]['duration'] = int(recording_sample_count/sampling_rate/60)

        print(f'[D{j+1:2d}]  Raw Fs: {np.mean(recording_raw_fs):>5.1f}  Noise: {np.mean(recording_noise_level):>4.1f}  FR: {np.mean(recording_firing_rate):>4.1f}  #Neuron: {len(recording_neuron_list):3d}  #Trial: {len(recording_trial_list):3d}  #Sample: {recording_sample_count:7d}  Duration: {int(recording_sample_count/sampling_rate/60):4d}  Name: {os.path.basename(dataset)[:20]}')
    return dataset_neuron, dataset_trial, dataset_prop

def build_causal_kernel(sampling_rate):
    xx = np.arange(0,199)/sampling_rate
    yy = scipy.stats.invgauss.pdf(xx,opt.smoothing/sampling_rate*2.0,101/sampling_rate,1)
    ix = np.argmax(yy)
    yy = np.roll(yy,int((99-ix)/1.5))
    causal_smoothing_kernel = yy/np.nansum(yy)
    return causal_smoothing_kernel
        
def load_recordings_from_file(file_path, sampling_rate):

    data = scio.loadmat(file_path)['CAttached'][0]

    recording_trial = list()

    trace_seq, spike_seq, rate_seq = np.zeros((0, 1),dtype='float32'), np.zeros((0,),dtype='int'), np.zeros((0,),dtype='float32')
    trace_seg, spike_seg, rate_seg = np.zeros((0, opt.signal_len),dtype='float32'), np.zeros((0, opt.signal_len),dtype='int'), np.zeros((0, opt.signal_len),dtype='float32')
    spike_num, rate_num, class_num = np.zeros((0, 1),dtype='int'), np.zeros((0, 1),dtype='float32'), np.zeros((0, 1),dtype='int')
    
    # record sampling rate after processing
    FS, FS_resampled = [],[]
    
    # for calculation of neuron-wise noise level
    concat_traces_mean = np.zeros((0,),dtype='float32')
    
    # for calculation of neuron-wise firing rate
    concat_events = []
    concat_times = []
    
    # for calibration of ER computation
    concat_events_times = []
    cum_times = 0
    
    for i,trial in enumerate(data):
        # find the relevant elements in the data structure
        # (dF/F traces; spike events; time stamps of fluorescence recording)
        keys = trial[0][0].dtype.descr
        keys_unfolded = list(sum(keys, ()))

        try:
            traces_index = int(keys_unfolded.index("fluo_mean")/2)
            fluo_time_index = int(keys_unfolded.index("fluo_time")/2)
            events_index = int(keys_unfolded.index("events_AP")/2)
        except:
            continue

        # spikes
        events = trial[0][0][events_index]
        events = events[~np.isnan(events)]
        ephys_sampling_rate = 1e4
        event_time = events/ephys_sampling_rate
        
        # fluorescence
        fluo_times = np.squeeze(trial[0][0][fluo_time_index])
        traces_mean = np.squeeze(trial[0][0][traces_index])
        traces_mean = traces_mean[:fluo_times.shape[0]]

        traces_mean = traces_mean[~np.isnan(fluo_times)]
        fluo_times = fluo_times[~np.isnan(fluo_times)]

        frame_rate = 1/np.mean(np.diff(fluo_times))

        # concatenate for statistics
        concat_traces_mean = np.concatenate([concat_traces_mean,traces_mean], axis=0)
        concat_events.append(len(events))
        concat_times.append(fluo_times[-1])
        
        # calibrate onset time
        event_time = event_time[np.logical_and(fluo_times[0]<=event_time,event_time<=fluo_times[-1])]
        event_time = event_time - fluo_times[0] + 1/frame_rate
        fluo_times = fluo_times - fluo_times[0] + 1/frame_rate
        if event_time.size==0:
            continue

        # resampling
        num_samples = int(round(traces_mean.shape[0]*sampling_rate/frame_rate))
        (traces_mean,fluo_times_resampled) = scipy.signal.resample(traces_mean,num_samples,np.squeeze(fluo_times),axis=0)
        frame_rate_resampled = 1/np.nanmean(np.diff(fluo_times_resampled))
        
        # calibrate bin size
        fluo_times_resampled = fluo_times_resampled*frame_rate_resampled/sampling_rate
        frame_rate_resampled = 1/np.nanmean(np.diff(fluo_times_resampled))
        
        # cleaning data
        num_samples -= int(np.floor(frame_rate_resampled))
        traces_mean = traces_mean[int(np.ceil(0.5*frame_rate_resampled)):int(np.ceil(0.5*frame_rate_resampled))+num_samples]
        fluo_times_resampled = fluo_times_resampled[int(np.ceil(0.5*frame_rate_resampled)):int(np.ceil(0.5*frame_rate_resampled))+num_samples]
        
        # calibration again
        if event_time.size==0 or fluo_times_resampled.size==0:
            continue
        event_time = event_time[np.logical_and(fluo_times_resampled[0]<=event_time,event_time<=fluo_times_resampled[-1])]
        event_time = event_time - fluo_times_resampled[0] + 1/frame_rate_resampled
        fluo_times_resampled = fluo_times_resampled - fluo_times_resampled[0] + 1/frame_rate_resampled
        if event_time.size==0:
            continue

        # bin the ground truth (spike times) into time bins determined by the resampled calcium recording
        fluo_times_bin_centers = fluo_times_resampled
        fluo_times_bin_edges = np.append(fluo_times_bin_centers,fluo_times_bin_centers[-1]+1/frame_rate_resampled/2) - 1/frame_rate_resampled/2

        [events_binned,event_bins] = np.histogram(event_time, bins=fluo_times_bin_edges)

        # concatenate event_time
        concat_events_times = np.concatenate([concat_events_times, event_time+cum_times])
        cur_times = len(traces_mean)/frame_rate_resampled
        cum_times += cur_times

        # trial-wise firing rate
        firing_rate = sum(events_binned)/(len(event_bins)/frame_rate_resampled)
        
        # trial-wise noise level
        noise_level = np.nanmedian(np.abs(np.diff(traces_mean)))*100/np.sqrt(frame_rate)
        
        # do pre-processing here if needed																										  
        traces_mean = np.expand_dims(traces_mean, 0)																			
		
        # padding incase recording is too short
        traces_mean = np.concatenate([np.zeros((1,opt.signal_len//2)), traces_mean, np.zeros((1,opt.signal_len//2))], axis=1)
        events_binned = np.concatenate([np.zeros((opt.signal_len//2,)), events_binned, np.zeros((opt.signal_len//2,))], axis=0)
        
        # smooth spikes to facilitate gradient descents
        if opt.causal_kernel:
            if 'causal_smoothing_kernel' not in locals():
                causal_smoothing_kernel = build_causal_kernel(sampling_rate)            
            events_binned_smooth = np.convolve(events_binned.astype(float),causal_smoothing_kernel,mode='same')
        if opt.gaussian_kernel:
            events_binned_smooth = scipy.ndimage.filters.gaussian_filter(events_binned.astype(float), sigma=opt.smoothing_std*sampling_rate)
        
        # format data into segments
        data_len = len(events_binned)-opt.signal_len
        before = int(opt.signal_len//2)
        after = int(opt.signal_len//2-1)
        
        X = np.zeros((data_len, opt.signal_len), dtype='float32')
        YY_spike = np.zeros((data_len, opt.signal_len), dtype='int')
        YY_rate = np.zeros((data_len, opt.signal_len), dtype='float32')
        Y_spike = np.zeros((data_len, 1), dtype='int')
        Y_rate = np.zeros((data_len, 1), dtype='float32')
        Y_class = np.zeros((data_len, 1), dtype='int')

        traces_mean = traces_mean.astype('float32')
        events_binned = events_binned.astype('int')
        events_binned_smooth = events_binned_smooth.astype('float32')
        
        for time_point in range(data_len):
            X[time_point,:] = traces_mean[:,time_point:time_point+opt.signal_len]
            YY_spike[time_point,:] = events_binned[time_point:time_point+opt.signal_len]
            YY_rate[time_point,:] = events_binned_smooth[time_point:time_point+opt.signal_len]
            Y_spike[time_point] = events_binned[time_point+before]
            Y_rate[time_point] = events_binned_smooth[time_point+before]
            Y_class[time_point] = events_binned[time_point+before]
        Y_class[Y_class>=opt.classes] = opt.classes-1
        
        recording_trial.append(dict(time_resampled=fluo_times_resampled, 
                                    frame_rate=frame_rate,frame_rate_resampled=frame_rate_resampled,
                                    firing_rate=firing_rate,
                                    noise_level=noise_level,
                                    trace_seq=traces_mean[:,before:before+data_len].T, 
                                    spike_seq=events_binned[before:before+data_len], 
                                    rate_seq=events_binned_smooth[before:before+data_len],
                                    trace_seg=X, spike_seg=YY_spike, rate_seg=YY_rate,
                                    spike_num=Y_spike, rate_num=Y_rate, class_num=Y_class,
                                    events_times=event_time, elapsed_times=cur_times))

        FS.append(frame_rate)
        FS_resampled.append(frame_rate_resampled)
        
        trace_seq = np.concatenate([trace_seq, traces_mean[:,before:before+data_len].T], axis=0)
        spike_seq = np.concatenate([spike_seq, events_binned[before:before+data_len]], axis=0)
        rate_seq = np.concatenate([rate_seq, events_binned_smooth[before:before+data_len]], axis=0)
        trace_seg = np.concatenate([trace_seg, X], axis=0)
        spike_seg = np.concatenate([spike_seg, YY_spike], axis=0)
        rate_seg = np.concatenate([rate_seg, YY_rate], axis=0)
        spike_num = np.concatenate([spike_num, Y_spike], axis=0)
        rate_num = np.concatenate([rate_num, Y_rate], axis=0)
        class_num = np.concatenate([class_num, Y_class], axis=0)
        
    recording_neuron = [dict(frame_rate=np.nanmean(FS), frame_rate_resampled=np.nanmean(FS_resampled), 
                             firing_rate=np.sum(concat_events)/np.sum(concat_times),
                             noise_level=np.nanmedian(np.abs(np.diff(concat_traces_mean)))*100/np.sqrt(np.nanmean(FS)),
                             trace_seq=trace_seq, spike_seq=spike_seq, rate_seq=rate_seq,
                             trace_seg=trace_seg, spike_seg=spike_seg, rate_seg=rate_seg,
                             spike_num=spike_num, rate_num=rate_num, class_num=class_num,
                             events_times=concat_events_times, elapsed_times=cum_times)]
    
    return recording_neuron, recording_trial, trace_seg.shape[0]
    
    
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=9, kernel_size=3, padding=1):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", kernel_size=kernel_size, padding=padding)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.encoder2 = UNet._block(features, features * 2, name="enc2", kernel_size=kernel_size, padding=padding)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", kernel_size=kernel_size, padding=padding)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck", kernel_size=kernel_size, padding=padding)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(1,2), stride=(1,2))
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", kernel_size=kernel_size, padding=padding)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(1,2), stride=(1,2))
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", kernel_size=kernel_size, padding=padding)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(1,2), stride=(1,2))
        self.decoder1 = UNet._block(features * 2, features, name="dec1", kernel_size=kernel_size, padding=padding)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = torch.relu(self.conv(dec1)).squeeze()

        return output

    @staticmethod
    def _block(in_channels, features, kernel_size, padding, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=kernel_size,padding=padding,bias=False,),),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(0.2,inplace=True)),
                    (name + "conv2",nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,bias=False,),),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(0.2,inplace=True)),
                ]
            )
        )

        
        
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        
        
class ENS2(object):
    def __init__(self):
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.DATA = [[]]*27
        self.MODEL = {}
        
        try:
            print(f'Prediction resolution is: {opt.sampling_rate}Hz')
        except:
            opt = argparse.ArgumentParser()

            opt.sampling_rate = 60
            opt.smoothing_std = 0.025
            opt.smoothing = opt.smoothing_std * opt.sampling_rate

            opt.causal_kernel = False
            opt.gaussian_kernel = not opt.causal_kernel

            opt.signal_len = 96
            opt.classes = 6

            opt.lr = 0.001
            opt.epochs = 5000
            opt.patience = 500
            opt.batch_size = 1024
            opt.sample_interval = opt.epochs
        
    
    def train(self, neuron='Exc', inputs='Raw', nets='UNet', losses='MSE', Fs='60', smoothing_std='0.025', smoothing_kernel='gaussian',
              cluster='None', hour='all', lr='0.001', kernel='3', node='150K', seg='96', batch='1024', es='500', verbose=0):
        
        #### define test mode
        self.TEST = 'C'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d')[2:]+'_'+neuron+'_'+Fs+'Hz_'+inputs+'_'+nets+'_'+losses+'_Test'

        if smoothing_std != '0.025':
            self.TEST = self.TEST+'_'+str(int(float(smoothing_std)*1000))+'ms'
        if smoothing_kernel == 'gaussian':
            opt.causal_kernel = False
            opt.gaussian_kernel = not opt.causal_kernel # use Gaussian smoothing kernel
        elif smoothing_kernel == 'causal':
            opt.causal_kernel = True
            opt.gaussian_kernel = not opt.causal_kernel # use causal smoothing kernel
            self.TEST = self.TEST+'_'+'causal'
        else:
            print("Smoothing kernel error...")
            return
            
        if cluster != 'None':
            self.TEST = self.TEST+'_'+cluster
        if hour != 'all':
            self.TEST = self.TEST+'_'+hour+'hr'
        
        if kernel != '3':
            self.TEST = self.TEST+'_'+kernel+'kernel'
        if node != '150K':
            self.TEST = self.TEST+'_'+node+'node'
            
        if lr != '0.001':
            opt.lr = float(lr)
            self.TEST = self.TEST+'_'+lr+'lr'
        if seg != '96':
            opt.signal_len = int(seg)
            self.TEST = self.TEST+'_'+seg+'seg'
        if batch != '1024':
            opt.batch_size = int(batch)
            self.TEST = self.TEST+'_'+batch+'batch'
        if es != '500':
            opt.patience = int(es)
            self.TEST = self.TEST+'_'+es+'es'
            
        print('【'+self.TEST+'】')
        fs_spec = (0,0)
        datasets, datasets_prop = [], []
        
        if neuron == 'Both':
            ds_on, ds_off = 2, 27
        elif neuron == 'Exc':
            ds_on, ds_off = 2, 21
        elif neuron == 'Inh':
            ds_on, ds_off = 22, 27
        else:
            print("Neuron type error...")
            return
        
        for dsets in [0]:#range(ds_on, ds_off+1):
                        
            tqdm.write(f'dataset {dsets}: preparing data...')
            
            #### load or re-compile data
            opt.sampling_rate = float(Fs)
            opt.smoothing_std = float(smoothing_std)
            opt.smoothing = opt.smoothing_std * opt.sampling_rate
                    
            tqdm.write(f'Sampling rate is: {opt.sampling_rate}Hz, smoothing window is: {opt.smoothing_std*1000}ms')
            
            if fs_spec == (opt.sampling_rate, opt.smoothing_std):
                print('Using previous datasets...')
            else:
                print('Re-compiling datasets...')
                del datasets, datasets_prop
                datasets, _, datasets_prop = load_all_ground_truth(opt.sampling_rate)
                fs_spec = (opt.sampling_rate, opt.smoothing_std)

            #### initialize vault
            self.DATA[dsets-1] = {}
            self.DATA[dsets-1]['inputs'] = inputs
            self.DATA[dsets-1]['nets'] = nets
            self.DATA[dsets-1]['losses'] = losses
            self.DATA[dsets-1]['dataset'] = dsets
#             self.DATA[dsets-1]['frame_rate'] = datasets_prop[dsets]['frame_rate']
#             self.DATA[dsets-1]['noise_level'] = datasets_prop[dsets]['noise_level']
#             self.DATA[dsets-1]['firing_rate'] = datasets_prop[dsets]['firing_rate']
#             self.DATA[dsets-1]['neuron_number'] = datasets_prop[dsets]['neuron_number']
            
            self.DATA[dsets-1]['sampling_rate'] = opt.sampling_rate
#             self.DATA[dsets-1]['correlation'] = []
#             self.DATA[dsets-1]['error'] = []
#             self.DATA[dsets-1]['bias'] = []
#             self.DATA[dsets-1]['eucd'] = []
#             self.DATA[dsets-1]['vpd'] = []
#             self.DATA[dsets-1]['er50'] = []
#             self.DATA[dsets-1]['er100'] = []
#             self.DATA[dsets-1]['er500'] = []
#             self.DATA[dsets-1]['gter50'] = []
            self.DATA[dsets-1]['loss'] = []

#             self.DATA[dsets-1]['calcium'] = []
#             self.DATA[dsets-1]['gt_rate'] = []
#             self.DATA[dsets-1]['gt_spike'] = []
#             self.DATA[dsets-1]['pd_rate'] = []
#             self.DATA[dsets-1]['pd_spike'] = []
#             self.DATA[dsets-1]['gt_event'] = []
#             self.DATA[dsets-1]['pd_event'] = []
#             self.DATA[dsets-1]['events_times'] = []

			
            start_time = datetime.datetime.now()
            self.model_ver = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')[2:]+'_dsets'+str(dsets)+'_'+str(opt.sampling_rate)+'Hz_'+inputs+'_'+nets+'_'+losses

            self.DATA[dsets-1]['model_ver'] = self.model_ver
            
            #### prepare data
            random.seed(0)
            torch.manual_seed(0)
            np.random.seed(0)

            torch.cuda.empty_cache()
            cudnn.deterministic = True
            cudnn.benchmark = False
            
            test_trace = []
            test_rate = []
            test_spike = []
            test_event = []
            train_trace = np.zeros((800*10000,opt.signal_len),dtype='float32')
            train_rate = np.zeros((800*10000,opt.signal_len),dtype='float32')
            train_spike = np.zeros((800*10000,opt.signal_len),dtype='float32')
            train_trace[:] = np.nan
            train_rate[:] = np.nan
            train_spike[:] = np.nan
            count = 0
            cum_time = 0

            if cluster == 'cluster':
                dset_indexes = self.cluster[dsets]
            elif cluster == 'anticluster':
                dset_indexes = self.anticluster[dsets]
            else:
                dset_indexes = np.arange(ds_on, ds_off+1)
            
            for dset_index in dset_indexes:
                if dset_index == dsets:
                    for trial_index in range(len(datasets[dset_index])):
                        test_trace.append(datasets[dset_index][trial_index]['trace_seg'])
                        test_rate.append(datasets[dset_index][trial_index]['rate_seg'])
                        test_spike.append(datasets[dset_index][trial_index]['spike_seg'])
                        test_event.append(datasets[dset_index][trial_index]['events_times'])
                else:
                    if dset_index == 1:
                        continue
                    for trial_index in range(len(datasets[dset_index])):
                        
                        tmp = datasets[dset_index][trial_index]['trace_seg']
                            
                        increment = tmp.shape[0]
                        if not increment:
                            continue

                        train_trace[count:count+increment,:] = tmp
                        train_rate[count:count+increment,:] = datasets[dset_index][trial_index]['rate_seg']
                        train_spike[count:count+increment,:] = datasets[dset_index][trial_index]['spike_seg']
                        count += increment

            train_trace = train_trace[0:count,:]
            train_rate = train_rate[0:count,:]
            train_spike = train_spike[0:count,:]

            if hour != 'all':
                np.random.seed(dsets)
                shrink_idx = (np.random.random_sample(np.ceil(float(hour)*3600/(opt.signal_len/opt.sampling_rate)).astype('int')) * count).astype('int')
                train_trace = train_trace[shrink_idx,:]
                train_rate = train_rate[shrink_idx,:]
                train_spike = train_spike[shrink_idx,:]
            
            if np.sum(train_trace==np.nan) or np.sum(train_rate==np.nan) or np.sum(train_spike==np.nan):
                print('NaN error...')
                break

            Training_dataset = TensorDataset(torch.FloatTensor(train_trace),torch.FloatTensor(train_rate),torch.FloatTensor(train_spike))
            Training_dataloader = DataLoader(Training_dataset, shuffle=True, batch_size=opt.batch_size)

            tqdm.write(f'dataset {dsets}, pair sample {len(Training_dataset)}: start training...')
            
            #### initiate network
            if nets=='UNet':
                if kernel == '3':
                    kernel_size, padding_size = 3, 1
#                     print('Using UNet with 3-size kernels')
                elif kernel == '5':
                    kernel_size, padding_size = 5, 2
                    print('Using UNet with 5-size kernels')
                elif kernel == '7':
                    kernel_size, padding_size = 7, 3
                    print('Using UNet with 7-size kernels')
                else:
                    print('Network error...')
                    break

                if node == '50K':
                    init_features_num = 5
                    print('Using UNet with 50K nodes')
                elif node == '150K':
                    init_features_num = 9
#                     print('Using UNet with 150K nodes')
                elif node == '450K':
                    init_features_num = 16
                    print('Using UNet with 450K nodes')
                else:
                    print('Network error...')
                    break
            
                C = UNet(init_features=init_features_num, kernel_size=kernel_size, padding=padding_size).to(self.DEVICE)
            elif nets=='LeNet':
                C = LeNet().to(self.DEVICE)
            elif nets=='FCNet':
                C = FCNet().to(self.DEVICE)
            else:
                print('Network error...')
                break
            C.apply(weights_init_normal)
            C_optimizer = optim.Adam(C.parameters(), lr=opt.lr)
#             print('Learning rate is '+str(opt.lr))

            mse_loss = torch.nn.MSELoss().to(self.DEVICE)

            early_stopping = EarlyStopping(patience=opt.patience, verbose=True, delta=0.0000)
            is_earlystop = 0
            
            #### start training
            start_time = datetime.datetime.now()
            t = trange(1, opt.epochs+1, leave=True,  ncols=1000)
            for epoch in t:

                # extract training batch
                np.random.seed(epoch)
                rand_idx = (np.random.random_sample(opt.batch_size) * len(Training_dataset)).astype('int')
                [train_trace, train_rate, train_spike] = Training_dataset[rand_idx]
                train_trace = Variable(train_trace.type(torch.FloatTensor)).to(self.DEVICE)
                train_rate = Variable(train_rate.type(torch.FloatTensor)).to(self.DEVICE)
                train_spike = Variable(train_spike.type(torch.FloatTensor)).to(self.DEVICE)
                
                # train model
                C_optimizer.zero_grad()
                C.train()

                if losses=='MSE':
                    loss = mse_loss(C(train_trace), train_rate)
                elif losses=='EucD':
                    loss = eucd_loss(C(train_trace), train_rate, train_spike)
                elif losses=='Corr':
                    loss = -pearson_corr_loss(C(train_trace), train_rate)
                else:
                    print('Loss error...')
                    break

                loss.backward()
                C_optimizer.step()

                early_stopping(loss.item(), C)
                if early_stopping.early_stop:
                    print('Early stopping...Epoch '+str(epoch))
                    is_earlystop = 1
                    
                # gather status
                t.set_description(inputs+' '+nets+' ['+losses+': %0.3f]' % loss.item())

                self.DATA[dsets-1]['loss'].append(loss.item())
                
                # check performance
                if (epoch % (opt.sample_interval) == 0) or is_earlystop:
                    os.makedirs('saved_model/', exist_ok=True)
                    if (epoch == opt.epochs) or is_earlystop:
                        torch.save(C, './saved_model/'+'C_'+str(self.model_ver)+'_Epoch'+str(epoch)+'.pt')

                if is_earlystop:
                    break
            self.MODEL[dsets] = C

            del Training_dataset, Training_dataloader
            del C
            
    
    def predict(self, test_data, state_dict=None):
        test_data = test_data.squeeze()
        Testing_dataset = TensorDataset(torch.FloatTensor(test_data))
        Testing_dataloader = DataLoader(Testing_dataset, shuffle=False, batch_size=8192)
        
        # testing
        if not self.MODEL.get(0, None):
            self.MODEL[0] = UNet() 
        C = self.MODEL[0].to(self.DEVICE)

        if state_dict:
            C.load_state_dict(state_dict)

        C.eval()
        with torch.no_grad():
            calcium = torch.zeros((0,opt.signal_len))
            pd_rate_tmp = torch.zeros((0,opt.signal_len)).type(torch.FloatTensor).to(self.DEVICE)

            for batch, test_data in enumerate(Testing_dataloader):
                test_output = C(Variable(test_data[0].type(torch.FloatTensor).to(self.DEVICE)))

                calcium = torch.cat([calcium, test_data[0]], axis=0)
                pd_rate_tmp = torch.cat([pd_rate_tmp, test_output], axis=0)

        calcium = calcium[:, [opt.signal_len//2]].cpu().numpy()
        pd_rate_tmp = pd_rate_tmp.cpu().numpy()

        pd_rate = np.zeros((1,calcium.shape[0]+opt.signal_len-1))
        for align_idx in range(pd_rate_tmp.shape[0]):
            pd_rate[:,align_idx:align_idx+opt.signal_len] += pd_rate_tmp[align_idx,:]
        pd_rate = pd_rate[:,opt.signal_len//2:opt.signal_len//2+calcium.shape[0]].transpose()/opt.signal_len

        # print('Estimate spike...')
        pd_spike = estimate_spike(pd_rate)
        pd_event = extract_event(pd_spike)
            
        del C
        return calcium, pd_rate, pd_spike, pd_event


def estimate_spike(rate, std=opt.smoothing, debug=False):
    rate = np.float32(np.array(copy.deepcopy(rate))).squeeze()
    # remove bubbles produced by neural network
    if not debug:
        rate[rate<0.02/std] = 0
    else:
        # try to aggresively filter out background firing
        rate[rate<np.sqrt(np.mean(rate[rate>0.02/std]))] = 0
    # initialize
    rate_diff = np.diff(np.int8(rate>0))
    est_spike = np.zeros(rate.shape, dtype='float32')
    est_rate = np.zeros(rate.shape, dtype='float32')
    onset, offset = 0, 0
    for idx in range(len(rate_diff)):
        # locate each piece of slices with spike rate
        if rate_diff[idx] == 1:
            onset = idx+1
        elif rate_diff[idx] == -1:
            if onset > 0:
                offset = idx
                # extract pieces of slices
                slices = rate[onset:offset+1]
                # at least one spike is included when probability is over 0.5
                could_add = True
                cur_spike = np.zeros(slices.shape, dtype='float32')
                if np.sum(slices)>=0.5:
                    cur_spike[np.argmax(slices)] = 1
                cur_rate = scipy.ndimage.filters.gaussian_filter(cur_spike, sigma=std, mode='constant', cval=0.)
                cur_loss = np.sum((slices-cur_rate)**2)
                # iteratively insert spikes that are best-match
                while could_add:
                    candidate_spike = cur_spike + np.eye(len(slices),len(slices),dtype='float32')
                    candidate_rate = scipy.ndimage.filters.gaussian_filter(candidate_spike, sigma=(0,std), mode='constant', cval=0.)
                    candidate_loss = np.sum(np.power(slices-candidate_rate,2),1)
                    new_loss, new_loss_idx = np.amin(candidate_loss), np.argmin(candidate_loss)
                    if new_loss - cur_loss <= -0.00000001:
                        cur_spike = candidate_spike[new_loss_idx,:]
                        cur_rate = candidate_rate[new_loss_idx,:]
                        cur_loss = new_loss
                        could_add = True
                    else:
                        est_spike[onset:offset+1] = cur_spike
                        est_rate[onset:offset+1] = cur_rate
                        could_add = False
        # force estimation with maximum slice length of 500 data points
        elif idx - onset >= 500-1:
            if len(rate_diff) > idx+1:
                rate_diff[idx+1] = -1
            if len(rate_diff) > idx+2:
                rate_diff[idx+2] = 1
    return est_spike


def extract_event(spike):
    spike_input = np.squeeze(copy.deepcopy(spike))
    event_output = []
    while np.sum(spike_input>0):
        event_output += ((np.where(spike_input>0)[0]+1)/opt.sampling_rate).tolist()
        spike_input -= 1
    event_output.sort()
    return event_output


def compile_test_data(src, trial_time, is_norm=False, is_denoise=0):
    trials = src.shape[0]
    frame_rate = src.shape[-1]/trial_time
    sampling_rate = opt.sampling_rate
    
    print("Test data has {} trials.".format(trials))
    print("Recording duration is {}s, equaling {}Hz frame rate.".format(trial_time, frame_rate))
    
    test_data = [[]]*trials
    for trial in range(trials):
        test_data[trial] = {}
        test_data[trial]['neuron'] = trial+1
        
        traces_mean = src[trial,:].squeeze()

        fluo_times = np.arange(1/frame_rate, trial_time+1/frame_rate, 1/frame_rate)
        test_data[trial]['frame_rate'] = np.float32(frame_rate)
        test_data[trial]['fluo_times'] = fluo_times.astype('float32')
        test_data[trial]['raw_dff'] = traces_mean.astype('float32')
        
        # resampling
        num_samples = int(round(traces_mean.shape[0]*sampling_rate/frame_rate))
        (traces_mean,fluo_times_resampled) = scipy.signal.resample(traces_mean,num_samples,np.squeeze(fluo_times),axis=0)
        frame_rate_resampled = 1/np.nanmean(np.diff(fluo_times_resampled))
    
        # normalizing (for testing only, not implemented for ENS2)
        if is_norm:
            traces_mean = (traces_mean-np.quantile(traces_mean,0.005))/(np.quantile(traces_mean,0.995)-np.quantile(traces_mean,0.005))
        
        test_data[trial]['dff_resampled'] = np.float32(traces_mean)
        
        # padding incase recording is too short
        traces_mean = np.concatenate([np.zeros((opt.signal_len//2,)), traces_mean, np.zeros((opt.signal_len//2,))])
    
        # format data into segments
        data_len = len(fluo_times_resampled)
        X = np.zeros((data_len, opt.signal_len), dtype='float32')
        for time_point in range(data_len):
            X[time_point,:] = traces_mean[time_point:time_point+opt.signal_len].astype('float32')

        test_data[trial]['dff_resampled_segment'] = X.astype('float32')
        test_data[trial]['fluo_times_resampled'] = fluo_times_resampled.astype('float32')
        test_data[trial]['frame_rate_resampled'] = np.float32(frame_rate_resampled)
        
    print('Compile data done.')
    return test_data