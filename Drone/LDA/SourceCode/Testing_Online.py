import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
import time
import os, glob
import hdf5storage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
import shutil
from datetime import datetime
import socket
import matplotlib.pyplot as plt

def Re_referencing(eegData, channelNum, sampleNum):
        after_car = np.zeros((channelNum,sampleNum))
        for i in np.arange(channelNum):
            after_car[i,:] = eegData[i,:] - np.mean(eegData,axis=0)
        return after_car

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

def Standardization(Epochs):
    for i in range(Epochs.shape[1]):
        Epochs[:,i,:] = np.subtract(Epochs[:,i,:], np.mean(Epochs[:,i,:]))
        Epochs[:,i,:] = Epochs[:,i,:] / np.std(Epochs[:,i,:])
    
    return Epochs    

def Make_Average_Component(Epochs, AverSize):
    Num = int(np.floor(Epochs.shape[0]/AverSize))
    
    Epochs_Aver = np.zeros((Num, Epochs.shape[1], Epochs.shape[2]))
    for i in range(Num):
        Epochs_Aver[i, :, :] = np.mean(Epochs[AverSize*i:AverSize*i+AverSize, :, :], axis=0)
    
    return Epochs_Aver
    
def Epoching(eegData, stims, code, samplingRate, nChannel, epochSampleNum, epochOffset,baseline):
        Time = stims[np.where(stims[:,1] == code),0][0]
        Time = np.floor(np.multiply(Time,samplingRate)).astype(int)
        Time_after = np.add(Time,epochOffset).astype(int)
        Time_base = np.add(Time,baseline).astype(int)
        Num = Time.shape
        Epochs = np.zeros((Num[0], nChannel, epochSampleNum))
        for j in range(Num[0]):
            Epochs[j, :, :] = eegData[:, Time_after[j]:Time_after[j] + epochSampleNum]
        
        Epochs = Standardization(Epochs)
        Epochs_Aver = np.mean(Epochs, axis=0)

        return Epochs_Aver

def Convert_to_FeatureVector(Epochs, buttonNum, featureNum):
    Features = np.zeros((buttonNum, featureNum))
    for i in range(buttonNum):
        Features[i, :] = np.reshape(Epochs[i, :, :], (1, featureNum))
    return Features
    
def resampling(Epochs, resampleRate, channelNum):
        resampled_epoch = np.zeros((channelNum, resampleRate))
        for j in range(channelNum):
            resampled_epoch[j,:] = signal.resample(Epochs[j,:], resampleRate)
            
        return resampled_epoch
    
def main():
        Classifier_path = '/Users/hyuns/Desktop/LDAModel/'
        classifier_list = sorted(glob.glob(Classifier_path + '*.pickle'), key=os.path.getmtime, reverse=True)
        print("classifer:", classifier_list[0])
        
        lda = joblib.load(classifier_list[0])
        
        root_path = '/Users/hyuns/Desktop/HGU/2020-2/Capstone/Drone Project/EEGData/VR300_Data/0729/hs/'
        resampleRate = 100
        
        mat_path = root_path + 'Online/'
        current_list = sorted(glob.glob(mat_path + '*.mat'), key=os.path.getmtime, reverse=True)
        score = 0
        
        for mat_file in current_list:
            print(mat_file)
            ans = mat_file[-5]
            
            mat = hdf5storage.loadmat(mat_file)
            channelNames = mat['channelNames']
            eegData = mat['eegData']
            samplingFreq = mat['samplingFreq']
            samplingFreq = samplingFreq[0,0]
            stims = mat['stims']
            channelNum = channelNames.shape
            channelNum = channelNum[1]
            eegData = np.transpose(eegData)
            buttonNum = 7
            
            #Bandpass Filter
            eegData = butter_bandpass_filter(eegData, 0.1, 30, samplingFreq, order=4)

            #Epoching
            epochSampleNum = int(np.floor(1.0 * samplingFreq))
            offset = int(np.floor(0.0 * samplingFreq))
            baseline = int(np.floor(1.0 * samplingFreq))
            
            ####### averaging whole epochs
            Epochs_Aver = np.zeros((buttonNum, channelNum, epochSampleNum))
            Epochs_final = np.zeros((buttonNum, channelNum, resampleRate))
            
            featureNum = channelNum*resampleRate
            
            for i in range(buttonNum):
                Epochs_Aver[i] = Epoching(eegData, stims, (i+1), samplingFreq, channelNum, epochSampleNum, offset, baseline)
                Epochs_final[i] = resampling(Epochs_Aver[i], resampleRate, channelNum)
            
            Features = Convert_to_FeatureVector(Epochs_final, buttonNum, featureNum)
        
            Answers = lda.decision_function(Features)
            answer = np.argmax(Answers) + 1
            
            print(Answers)
            if int(ans) == int(answer): 
                score = score + 1
            print('order: ', ans, 'predict: ', answer)
        
        print('score:', score)
if __name__ == "__main__":
    main()
