import os
from pdb import set_trace
import joblib
import tqdm
import numpy as np

path = '/home/dobbang/jeon/HACTrack/posetrack/eval/posetrack21/posetrack21/data/gt/PoseTrackReID/posetrack_data/mot/val'
videos = np.sort(np.array(os.listdir(path)))
data_gt = joblib.load('/home/dobbang/jeon/HACTrack/phalp_prime/_DATA/posetrack/posetrack18_gt_data.pickle') 

for v in range(len(videos)):
    video = videos[v]
    data=data_gt[video]
    
    gt_txt = os.path.join(path,video,'gt/gt.txt')
    gt_sampling = open(path+'/'+video+'/gt/gt_sampling.txt','w')
    
    for t,frames in enumerate(data.keys()):
        idx = t+1
        print(t,frames)
        f = open(gt_txt)
        lines = f.readlines()
        for line in lines:
            if int(str(line).split(',')[0]) == int(str(frames).split('.')[0])+1:
                tmp_line = str(line).split(',')[1:]
                tmp_line.insert(0,str(idx))
                new_line = ','.join(tmp_line)
                gt_sampling.write(new_line)
    gt_sampling.close()
                
