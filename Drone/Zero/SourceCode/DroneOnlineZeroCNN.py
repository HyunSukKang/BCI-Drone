import numpy as np
import sys

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None

	def initialize(self):
            print('Python initialize function started')
            global eegData, stims, trigger, eegData_txt, stims_txt, start_txt, result_txt
            eegData = np.zeros((32,1))
            stims = np.zeros((1,3))
            trigger = 0.
            #filename = 'C:/Users/Ahn-Lab/Documents/SelectedFeatures.pickle' # Training data and selectedFeature index information are stored in this file.
            eegData_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/eegData.out'
            stims_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/stims.out'
            start_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/start.out'
            result_txt = 'C:/Users/NTH417/Desktop/Drone/Zero/CNNtemp/result.out'
            start_txt_trigger(start_txt)
            
	def process(self):
            global eegData, stims, trigger, eegData_txt, stims_txt, result_txt
            #Signal acquisition
            for chunkIndex in range( len(self.input[0]) ):
                if(type(self.input[0][chunkIndex]) == OVSignalHeader):
    				self.signalHeader = self.input[0].pop()                
                elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
                    chunk = self.input[0].pop()
                    signalRaw = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                    eegData = np.append(eegData,signalRaw,axis=1)
            #Stimulation acquisition
            for chunkIndex in range( len(self.input[1]) ):
                chunk = self.input[1].pop()
                if(type(chunk) == OVStimulationSet):
                    for stimIdx in range(len(chunk)):
                        stim=chunk.pop();
                        x = np.array([[stim.date,stim.identifier,0.0]])
                        stims = np.append(stims, x, axis=0)
                        trigger = stim.identifier
                        if(trigger == 7.):
                            print('got here')
                            trigger = 0.
                            stims = np.delete(stims,0,0)
                            save_data(eegData, stims, eegData_txt, stims_txt)
                            
    def start_txt_trigger(start_txt):
        np.savetxt(start_txt, [1])
    
    def save_data(eegData, stims, eegData_txt, stims_txt):
        np.savetxt(eegData_txt, eegData, delimiter = ",")
        np.savetxt(stims_txt, stims, delimiter = ",")
        
	def uninitialize(self):
            print('Python uninitialize function started')
            return

box = MyOVBox()

	
