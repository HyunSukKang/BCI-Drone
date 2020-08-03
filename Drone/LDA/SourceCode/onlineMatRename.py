import os, glob

result_file = open('/Users/hyuns/Desktop/HGU/2020-2/Capstone/Drone Project/EEGData/VR300_Data/0729/hs/hs.txt', 'r', encoding = 'utf-16')
online_path = '/Users/hyuns/Desktop/HGU/2020-2/Capstone/Drone Project/EEGData/VR300_Data/0729/hs/Online/'

onlineMat_list = sorted(glob.glob(online_path + '*.mat'), key=os.path.getmtime, reverse=True)

lines = result_file.readlines()

# print(onlineMat_list[0])
# print(onlineMat_list[1])

for i in range(len(lines)-1):
    line = lines[i]
    mat_file = onlineMat_list[i]
    
    index = [int(s) for s in line.split() if s.isdigit()]
    
    path, filename = os.path.split(mat_file)
    
    
    newname = path + '/'+ str(i+1) + '_' + str(index[0]) + '.mat'
    os.rename(mat_file, newname)
    
    print('before:', mat_file)
    print('after:', newname)
