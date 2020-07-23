import numpy as np
from scipy.signal import butter, lfilter
import time
import os, glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
import shutil
from datetime import datetime
import socket

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
    
def main():
#        global file_exist, file1, file2, channelNum
        eegData_txt = 'C:/Users/NTH417/Desktop/Drone/LDA/data/eegData.out'
        stims_txt = 'C:/Users/NTH417/Desktop/Drone/LDA/data/stims.out'
        start_txt = 'C:/Users/NTH417/Desktop/Drone/LDA/data/start.out'
        moveData_eeg = 'C:/Users/NTH417/Desktop/Drone/LDA/Online/Data/txt_files/eegData/'
        moveData_stims = 'C:/Users/NTH417/Desktop/Drone/LDA/Online/Data/txt_files/stims/'
        
        Classifier_path = "C:/Users/NTH417/Desktop/Drone/LDA/Model/"
        
        current_list2 = sorted(glob.glob(Classifier_path + '*.pickle'), key=os.path.getmtime, reverse=True)
        Classifier_real = current_list2[0]
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        lda = joblib.load(Classifier_real)
        
        serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSock.bind(('', 12240))
        serverSock.listen(0)
        connectionSock, addr = serverSock.accept()
        print(str(addr),'에서 접속이 확인되었습니다.')
        
        for i in range(0, 12):
            #load text file
            while True:
                if os.path.isfile(start_txt):
                    break
            start_time = time.time()
            
            while(time.time() - start_time < 25):
                pass
            
            while True:
                if os.path.isfile(eegData_txt) & os.path.isfile(stims_txt):
                    processing_time = time.time()
                    os.remove(start_txt)
                    eegData = np.loadtxt(eegData_txt, delimiter = ",")
                    stims = np.loadtxt(stims_txt, delimiter = ",")
                    ctime = datetime.today().strftime("%m%d_%H%M%S")
                    moveData_e = moveData_eeg + ctime + 'eegData.out'
                    moveData_s = moveData_stims + ctime + 'stims.out'
                    shutil.move(eegData_txt, moveData_e)
                    shutil.move(stims_txt, moveData_s)
                    break
                    
            print("got process")
            channelNum = 7
            samplingFreq = 300
            buttonNum = 7
            
            ### Preprocessing process            

            #Bandpass Filter
            eegData = butter_bandpass_filter(eegData, 0.23, 30, samplingFreq, 4)

            #Epoching
            epochSampleNum = int(np.floor(1.0 * samplingFreq))
            offset = int(np.floor(0.0 * samplingFreq)) 
            baseline = int(np.floor(1.0 * samplingFreq)) 
            
            Epochs_Aver = np.zeros((buttonNum, channelNum, epochSampleNum))
            featureNum = channelNum*epochSampleNum
            
            for i in range(buttonNum):
                Epochs_Aver[i] = Epoching(eegData, stims, (i+1), samplingFreq, channelNum, epochSampleNum, offset, baseline)
                
            Features = Convert_to_FeatureVector(Epochs_Aver, buttonNum, featureNum)
            
            Answers = lda.decision_function(Features)
            answer = np.argmax(Answers) + 1
            
#            np.savetxt(result_txt, answer)
            print("Process time: ", time.time() - processing_time)
            print("Result: ", answer)
            connectionSock.send(str(answer).encode("utf-8"))
            
if __name__ == "__main__":
    main()
