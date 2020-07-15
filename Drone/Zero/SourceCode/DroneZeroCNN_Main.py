import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
import time
import os
from keras.models import load_model
import shutil
from datetime import datetime
import socket

def Re_referencing(eegData, channelNum, sampleNum):
        after_car = np.zeros((channelNum,sampleNum))
        for i in np.arange(channelNum):
            after_car[i,:] = eegData[i,:] - np.mean(eegData,axis=0)
        return after_car

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos
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
#             for i in range(nChannel):
#                 Epochs[j, i, :] = np.subtract(Epochs[j, i, :], np.mean(eegData[i,Time_after[j]:Time_base[j]]))
        return [Epochs,Num[0]]

def resampling(Epochs, EpochNum, resampleRate, channelNum):
            resampled_epoch = np.zeros((EpochNum, channelNum, resampleRate))
            for i in range(EpochNum):
                for j in range(channelNum):
                    resampled_epoch[i,j,:] = signal.resample(Epochs[i,j,:], resampleRate)
            return resampled_epoch

def main():
        #load cnn model and predict result
        model = load_model('C:/Users/wldk5/WorldSystem/Zero/Model/ZeroCNN_7CH(0.5-10, 0.2-0.6).h5')
        model.load_weights('C:/Users/wldk5/WorldSystem/Zero/Model/ZeroCNN_7CH_Weight(0.5-10, 0.2-0.6).h5')
#        global file_exist, file1, file2, channelNum
        eegData_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/eegData.out'
        stims_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/stims.out'
        start_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/start.out'
        moveData_eeg = 'C:/Users/NTH417/Desktop/Drone/Zero/Online/Data/txt_files/eegData/'
        moveData_stims = 'C:/Users/NTH417/Desktop/Drone/Zero/Online/Data/txt_files/stims/'
        
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
            
            ### Preprocessing process            
            sampleNum = eegData.shape[1]
            
            #Common Average Reference
            eegData = Re_referencing(eegData, channelNum, sampleNum)

            #Bandpass Filter
            eegData = butter_bandpass_filter(eegData, 0.5, 10, samplingFreq, 4)
            
            result_tot = np.zeros((1, 6))
            screen_answer = np.zeros((6))
            screen_answer_proba = np.zeros((6))
            
            for i in range(15):
                #Epoching
                epochSampleNum = int(np.floor(0.4 * samplingFreq))
                offset = int(np.floor((0.01*i+0.2) * samplingFreq)) # no delay 기준 0.2 - 0.6
                baseline = int(np.floor((0.01*i+0.6) * samplingFreq)) # delay 있으면 0.3 - 0.7으로 바꿔야함
                [Epochs1, Num1] = Epoching(eegData, stims, 1, samplingFreq, channelNum, epochSampleNum, offset, baseline)
                [Epochs2, Num2] = Epoching(eegData, stims, 2, samplingFreq, channelNum, epochSampleNum, offset, baseline)
                [Epochs3, Num3] = Epoching(eegData, stims, 3, samplingFreq, channelNum, epochSampleNum, offset, baseline)
                [Epochs4, Num4] = Epoching(eegData, stims, 4, samplingFreq, channelNum, epochSampleNum, offset, baseline)
                [Epochs5, Num5] = Epoching(eegData, stims, 5, samplingFreq, channelNum, epochSampleNum, offset, baseline)
                [Epochs6, Num6] = Epoching(eegData, stims, 6, samplingFreq, channelNum, epochSampleNum, offset, baseline)

                epochSampleNum = 51
                #Resampling
                Epochs1 = resampling(Epochs1, Num1, epochSampleNum, channelNum)
                Epochs2 = resampling(Epochs2, Num2, epochSampleNum, channelNum)
                Epochs3 = resampling(Epochs3, Num3, epochSampleNum, channelNum)
                Epochs4 = resampling(Epochs4, Num4, epochSampleNum, channelNum)
                Epochs5 = resampling(Epochs5, Num5, epochSampleNum, channelNum)
                Epochs6 = resampling(Epochs6, Num6, epochSampleNum, channelNum)

                result = np.zeros((1,6))

                Epochs1 = np.reshape(Epochs1, (Num1,1,channelNum,epochSampleNum))
                Epochs2 = np.reshape(Epochs2, (Num2,1,channelNum,epochSampleNum))
                Epochs3 = np.reshape(Epochs3, (Num3,1,channelNum,epochSampleNum))
                Epochs4 = np.reshape(Epochs4, (Num4,1,channelNum,epochSampleNum))
                Epochs5 = np.reshape(Epochs5, (Num5,1,channelNum,epochSampleNum))
                Epochs6 = np.reshape(Epochs6, (Num6,1,channelNum,epochSampleNum))

                a1 = model.predict(Epochs1)
                a2 = model.predict(Epochs2)
                a3 = model.predict(Epochs3)
                a4 = model.predict(Epochs4)
                a5 = model.predict(Epochs5)
                a6 = model.predict(Epochs6)

                result[0,0] = np.sum(a1[:,1])
                result[0,1] = np.sum(a2[:,1])
                result[0,2] = np.sum(a3[:,1])
                result[0,3] = np.sum(a4[:,1])
                result[0,4] = np.sum(a5[:,1])
                result[0,5] = np.sum(a6[:,1])
                
                answer = np.argmax(result)
                screen_answer[answer] += 1
                screen_answer_proba[answer] += result[0, answer]
                

            answer = np.argmax(screen_answer) + 1
#             answer = np.argmax(screen_answer_proba) + 1
            
#            np.savetxt(result_txt, answer)
            print("Process time: ", time.time() - processing_time)
            print("Result: ", answer)
            connectionSock.send(str(answer).encode("utf-8"))
            
if __name__ == "__main__":
    main()