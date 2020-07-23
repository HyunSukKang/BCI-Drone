## DataProcessing and model generation process
import hdf5storage
import numpy as np
from scipy.signal import butter, lfilter
import os
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

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

def Epoching(eegData, stimsT, samplingFreq, channelNum, epochSampleNum, offset, baseline):
        Time_after = np.add(stimsT,offset).astype(int)
        Time_base = np.add(stimsT,baseline).astype(int)
        Num = stimsT.shape[1]
        Epochs = np.zeros((Num, channelNum, epochSampleNum))
        for j in range(Num):
            Epochs[j, :, :] = eegData[:,Time_after[0][j]:Time_after[0][j] + epochSampleNum]
            
        return [Epochs,Num]

def DownsamplingEpoch(EpochsT, EpochsN, downsampleRate):
        num = np.floor(EpochsT.shape[2] / downsampleRate).astype(int)
        DownsampledT = np.zeros((EpochsT.shape[0],EpochsT.shape[1],num))
        DownsampledN = np.zeros((EpochsN.shape[0],EpochsN.shape[1],num))
        for i in range(num):
            for j in range(EpochsT.shape[1]):
                for k in range(EpochsT.shape[0]):
                    DownsampledT[k,j,i] = np.mean(EpochsT[k,j,i*downsampleRate:(i+1)*downsampleRate],dtype=np.float64)
                for l in range(EpochsN.shape[0]):
                    DownsampledN[l,j,i] = np.mean(EpochsN[l,j,i*downsampleRate:(i+1)*downsampleRate],dtype=np.float64)
        return [DownsampledT, DownsampledN, num]

def Make_Average_Component(EpochsT, NumT, EpochsN, NumN, channelNum, epochSampleNum, componentNum):
    EpochsT = Standardization(EpochsT)
    EpochsN = Standardization(EpochsN)
    
    NumT_Aver = NumT-componentNum
    NumN_Aver = NumN-componentNum
    
    EpochsT_Aver = np.zeros((NumT_Aver, channelNum, epochSampleNum))
    EpochsN_Aver = np.zeros((NumN_Aver, channelNum, epochSampleNum))
    for i in range(NumT_Aver):
        EpochsT_Aver[i, :, :] = np.mean(EpochsT[i:i+componentNum, :, :], axis=0)
    for j in range(NumN_Aver):
        EpochsN_Aver[j, :, :] = np.mean(EpochsN[j:j+componentNum, :, :], axis=0)
        
    return [EpochsT_Aver, NumT_Aver, EpochsN_Aver, NumN_Aver]
    
def GenerateP300Data(filename):
        channelNum = 7
        epochSampleNum = 256
        target = np.zeros((280,channelNum,epochSampleNum))
        nontarget = np.zeros((280,channelNum,epochSampleNum))
        for i in np.arange(1,3):
            if (i==2):
                filename = filename + '_2'
            mat = hdf5storage.loadmat(filename)
            eegData = mat['eegData']
            samplingFreq = mat['samplingFreq'][0,0]
            stimsN = mat['stimsN']
            stimsT = mat['stimsT']
            sampleNum = eegData.shape[1]
            channelIndex = [18, 30, 12, 11, 19, 10, 15]
            
            # vr300 7 channel
            # [P4, Fz, Pz, P3, PO8, PO7, Oz]
            # [19, 31, 13, 12, 20, 11, 16]
            
            eegData = eegData[channelIndex]
        
            ## Preprocessing process
        
            #Bandpass Filter
            eegData = butter_bandpass_filter(eegData, 0.23, 30, samplingFreq, 4)
        
            #Epoching
            epochSampleNum = int(np.floor(1.0 * samplingFreq))
            offset = int(np.floor(0.0 * samplingFreq))
            baseline = int(np.floor(1.0 * samplingFreq))
            [EpochsT, NumT] = Epoching(eegData, stimsT, samplingFreq, channelNum, epochSampleNum, offset, baseline)
            [EpochsN, NumN] = Epoching(eegData, stimsN, samplingFreq, channelNum, epochSampleNum, offset, baseline)
            
            NumN = NumT
            EpochsN = Balancing_DataSet(EpochsN, NumN)
            
            #Downsampling
            downsampleRate = 2
            samplingFreq = samplingFreq / 2
            [EpochsT,EpochsN,epochSampleNum] = DownsamplingEpoch(EpochsT, EpochsN, downsampleRate)
            
            [EpochsT, NumT_Aver, EpochsN, NumN_Aver] = Make_Average_Component(EpochsT, NumT, EpochsN, NumN, channelNum, epochSampleNum, 20)
            
            # # plotEEGdata(eegData, channelNum)
            target[140*(i-1):140*i,:,:] = EpochsT
            nontarget[140*(i-1):140*i,:,:] = EpochsN
        
        return [target, nontarget]

def main():
        Datapath = '/Users/hyuns/Desktop/Data'
        createFolder(Datapath)
        createFolder(Datapath + '/PythonData')
        
        root = '/Users/hyuns/Desktop/P300Data/S'
        filename = ''
        channelNum = 7
        epochSampleNum = 256
        epochNum = 280
        alltarget = np.zeros((epochNum*55,channelNum,epochSampleNum))
        allnontarget = np.zeros((epochNum*55,channelNum,epochSampleNum))
        for i in np.arange(1,56):
            if(i<10):
                filename = root + '0' + str(i)
            else:
                filename = root + str(i)
            [alltarget[epochNum*(i-1):epochNum*i,:,:],allnontarget[epochNum*(i-1):epochNum*i,:,:]] = GenerateP300Data(filename)
            print("subject {0} is preprocessed".format(str(i)))
        
        
        trainingData = np.concatenate((alltarget, allnontarget))
        trainingData = np.reshape(trainingData,((epochNum*2)*55,1,channelNum,epochSampleNum));
        label = np.concatenate((np.ones((epochNum*55,1)).astype(int),np.zeros((epochNum*55,1)).astype(int))).ravel()
        label = np_utils.to_categorical(label, 2);
        
        ##Generating CNN model
        model = Sequential();
        #model.add(AveragePooling2D(pool_size=(1, 4), strides=(1,4))) # this was added
        model.add(Conv2D(epochSampleNum, kernel_size=(1, 25),data_format='channels_first',input_shape=(1, channelNum, epochSampleNum)))
        model.add(BatchNormalization())
        model.add(Conv2D(epochSampleNum, (channelNum, 1),data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        #model = multi_gpu_model(model, gpus=2);
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy']);
        
        data = trainingData;
        randIdx = np.random.permutation(55*(epochNum*2));
        trainIdx = randIdx[0:int((epochNum*2)*55*0.95)];
        valIdx = randIdx[int((epochNum*2)*55*0.95):55*(epochNum*2)];
        
        trainData = data[trainIdx,:,:,:];
        trainLabel = label[trainIdx];
        valData = data[valIdx,:,:,:];
        valLabel = label[valIdx];
        
        early_stopping = EarlyStopping(patience = 3);
        
        fittedModel = model.fit(trainData, trainLabel, epochs=10, validation_data=(valData, valLabel), callbacks=[early_stopping]);
        
        model.save('/Users/hyuns/Desktop/Zero/ZeroModel/ZeroCNN_7CH(0.5-10, 0.2-0.6, test).h5')
        model.save_weights('/Users/hyuns/Desktop/Zero/ZeroModel/ZeroCNN_7CH_Weight(0.5-10, 0.2-0.6, test).h5')
            
if __name__ == "__main__":
    main()
