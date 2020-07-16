import numpy as np
import pickle
import pandas as pd
import statsmodels.api as sm
import hdf5storage
from scipy.signal import butter, lfilter
import os, glob, time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import matlab.engine
from datetime import datetime
from sklearn.externals import joblib

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

def Epoching(eegData, stims, code, samplingRate, nChannel, epochSampleNum, epochOffset,baseline):
        Time = stims[np.where(stims[:,1] == code),0][0]
        Time = np.floor(np.multiply(Time,samplingRate)).astype(int)
        Time_after = np.add(Time,epochOffset).astype(int)
        Time_base = np.add(Time,baseline).astype(int)
        Num = Time.shape
        Epochs = np.zeros((Num[0], nChannel, epochSampleNum))
        for j in range(Num[0]):
            Epochs[j, :, :] = eegData[:, Time_after[j]:Time_after[j] + epochSampleNum]
            for i in range(nChannel):
                Epochs[j, i, :] = np.subtract(Epochs[j, i, :], np.mean(eegData[i,Time_after[j]:Time_base[j]]))
        return [Epochs, Num[0]]

def Convert_to_featureVector(EpochsT, NumT, EpochsN, NumN, featureNum):
        FeaturesT = np.zeros((NumT, featureNum))
        for i in range(NumT):
            FeaturesT[i,:] = np.reshape(EpochsT[i,:,:],(1,featureNum))
        FeaturesN = np.zeros((NumN, featureNum))
        for j in range(NumN):
            FeaturesN[j,:] = np.reshape(EpochsN[j,:,:],(1,featureNum))
        return [FeaturesT,FeaturesN]

def Balancing_DataSet(Epochs, size):
    Epochs_New = np.zeros((size, Epochs.shape[1], Epochs.shape[2]))
    
    index = np.random.choice(Epochs.shape[0], size = size, replace = False)
    
    Epochs_New = Epochs[index, :, :]
    
    return Epochs_New
    
def Normalization(Epochs):
    for i in range(Epochs.shape[1]):
        Epochs[:,i,:] = np.subtract(Epochs[:,i,:], np.mean(Epochs[:,i,:]))
        Epochs[:,i,:] = Epochs[:,i,:] / np.std(Epochs[:,i,:])
        
#     Epochs = np.subtract(Epochs, np.mean(Epochs))
#     Epochs = Epochs / np.std(Epochs)
    
    return Epochs
    
def Make_Average_Component(EpochsT, NumT, EpochsN, NumN, channelNum, epochSampleNum, averTrialNum):
    EpochsT = Normalization(EpochsT)
    EpochsN = Normalization(EpochsN)
    
    NumT_Aver = int(np.floor(NumT/averTrialNum))
    NumN_Aver = int(np.floor(NumN/averTrialNum))
    
    EpochsT_Aver = np.zeros((NumT_Aver, channelNum, epochSampleNum))
    EpochsN_Aver = np.zeros((NumN_Aver, channelNum, epochSampleNum))
    for i in range(NumT_Aver):
        EpochsT_Aver[i, :, :] = np.mean(EpochsT[averTrialNum*i:averTrialNum*i+averTrialNum, :, :], axis=0)
    for j in range(NumN_Aver):
        EpochsN_Aver[j, :, :] = np.mean(EpochsN[averTrialNum*j:averTrialNum*j+averTrialNum, :, :], axis=0)
        
    return [EpochsT_Aver, NumT_Aver, EpochsN_Aver, NumN_Aver]


def main():
        start = time.time()
        
        ##Generate Preprocessing Training data
        ctime = datetime.today().strftime("%m%d_%H%M")
#         Classifier_path = 'C:/Users/sokon/Desktop/Drone/Zero/Model/LDA/' + ctime + 'Classifier.pickle'
        Classifier_path = '/Users/hyuns/Desktop/' + ctime + 'Classifier.pickle'
        channelNum = 7
        
#         ov_Path = "C:/Users/sokon/Desktop/Drone/Zero/Training/Data/"
#         current_list = []
#         current_list = sorted(glob.glob(ov_Path + '*.ov'), key=os.path.getmtime, reverse=True)
#         ovfile_name = current_list[0]
#         matfile_name = current_list[0][:-3] + ".mat"
        matfile_name = '/Users/hyuns/Desktop/HGU/2020-2/Capstone/Drone Project/EEGData/VR300_Data/sion/Training/Training-[2020.02.03-14.59.55].mat'
    
#         print("current ov file path:", current_list[0])
#         eng = matlab.engine.start_matlab()
#         k = eng.convert_ov2mat(ovfile_name, matfile_name)
        mat = hdf5storage.loadmat(matfile_name)
        channelNames = mat['channelNames']
        eegData = mat['eegData']
        samplingFreq = mat['samplingFreq']
        samplingFreq = samplingFreq[0,0]
        stims = mat['stims']
        channelNum = channelNames.shape
        channelNum = channelNum[1]
        eegData = np.transpose(eegData)
        
        ##Preprocessing process
        sampleNum = eegData.shape[1]
        
        #Common Average Reference
#         eegData = Re_referencing(eegData, channelNum, sampleNum)

        #Bandpass Filter
        eegData = butter_bandpass_filter(eegData, 0.2, 30, samplingFreq, 4)
    
        #Epoching
        epochSampleNum = int(np.floor(1.0 * samplingFreq))
        offset = int(np.floor(0.0 * samplingFreq)) 
        baseline = int(np.floor(1.0 * samplingFreq)) 
        [EpochsT, NumT] = Epoching(eegData, stims, 1, samplingFreq, channelNum, epochSampleNum, offset, baseline)
        [EpochsN, NumN] = Epoching(eegData, stims, 0, samplingFreq, channelNum, epochSampleNum, offset, baseline)
        
        EpochsN_New = Balancing_DataSet(EpochsN, NumT)
        
        #Convert to feature vector
        [EpochsT_Aver, NumT_Aver, EpochsN_Aver, NumN_Aver] = Make_Average_Component(EpochsT, NumT, EpochsN_New, NumT, channelNum, epochSampleNum, 20)
        featureNum = channelNum*epochSampleNum
        [FeaturesT, FeaturesN] = Convert_to_featureVector(EpochsT_Aver, NumT_Aver, EpochsN_Aver, NumN_Aver, featureNum)
        TrainData = np.concatenate((FeaturesT, FeaturesN))
        TrainLabel = np.concatenate((np.ones((NumT_Aver,1)).astype(int),np.zeros((NumN_Aver,1)).astype(int))).ravel()
        
        #Feature Selection process
        x = np.arange(featureNum)
        Data = pd.DataFrame(TrainData ,columns=x)
        
        #Saving LDA classifier
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        lda.fit(Data, TrainLabel)
        joblib.dump(lda, Classifier_path, protocol=2)
        #print(SelectedFeatures)
        print("time :", time.time() - start)
        
if __name__ == "__main__":
    main()
