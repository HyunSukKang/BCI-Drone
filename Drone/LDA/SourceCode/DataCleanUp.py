import matlab.engine
import os, glob
import shutil

eng = matlab.engine.start_matlab()

def ov2mat(file_list, move_path):
    for ovfile in file_list:
        print(ovfile)
        ovhead, ovtail = os.path.split(ovfile)
        
        matfile_name = ovfile[:-3] + ".mat"
        mathead, mattail = os.path.split(matfile_name)
        
        k = eng.convert_ov2mat(ovfile, matfile_name)
        
        shutil.move(ovfile, move_path + 'ov\\' + ovtail)
        shutil.move(matfile_name, move_path + 'mat\\' + mattail)

def clean():
    desktop = 'C:\\Users\\user\\Desktop\\'
    train_path = 'C:\\Users\\user\\Desktop\\Drone\\LDA\\Training\\'
    online_path = 'C:\\Users\\user\\Desktop\\Drone\\LDA\\Online\\ov\\'
    txt_path = 'C:\\Users\\user\\Desktop\\DSI_Drone_AR_VR\\DSI_Drone\\Assets\\StreamingAssets\\'
    txt_list = sorted(glob.glob(txt_path + '*.txt'), key=os.path.getmtime, reverse=True)
    train_list = sorted(glob.glob(train_path + '*.ov'), key=os.path.getmtime)
    online_list = sorted(glob.glob(online_path + '*.ov'), key=os.path.getmtime)
    
    # Create Folder
    txt_head, txt_tail = os.path.split(txt_list[0])
    folder_name = txt_tail[:-4]
    
    final_path = desktop + folder_name
    os.mkdir(final_path)
    os.mkdir(final_path + '\\Training')
    os.mkdir(final_path + '\\Training\\ov')
    os.mkdir(final_path + '\\Training\\mat')
    
    os.mkdir(final_path + '\\Online')
    os.mkdir(final_path + '\\Online\\ov')
    os.mkdir(final_path + '\\Online\\mat')
    
    # Remove last ov file
    os.remove(online_list[-1])
    online_list = online_list[:-1]
    
    # Convert Ov files to Mat files
    ov2mat(train_list, final_path + '\\Traning\\')
    ov2mat(online_list, final_path + '\\Online\\')
    
    
    
def main():
    clean()
    
    
if __name__ == "__main__":
    main()
