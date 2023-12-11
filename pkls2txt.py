from tqdm import tqdm
import trackeval as trackeval
import sys
from tqdm import tqdm
import joblib

if __name__ == '__main__':

    data_gt = joblib.load('_DATA/posetrack/posetrack18_gt_data.pickle') 

    results_dir = str(sys.argv[1])
    method      = str(sys.argv[2])
    dataset     = str(sys.argv[3])

    data_all = joblib.load(results_dir + '/'+str(dataset)+'_'+str(method)+'.pkl')        
    txt_path = results_dir+'/txt/'

    for video in tqdm(list(data_all.keys())):
        file = open(txt_path+video+'.txt','w')
        print("processing: "+video )    
        for t, frame in enumerate(data_all[video].keys()):
            frame_index = t+1
            
            if video == '001022_mpii_test' and frame == '000105.jpg' :
                continue
            if video == '023963_mpii_test' and frame == '000146.jpg' :
                continue
            
            data = data_all[video][frame]
            pt_ids_      = data[5]
            pt_bbox_     = data[6]
            
            for p_, b_ in zip(pt_ids_, pt_bbox_):
                id = p_ # id
                x1 = int(b_[0]) #x
                y1 = int(b_[1]) #y
                w = int(b_[2]) #width
                h = int(b_[3]) #height

                file.write(str(frame_index)+','+str(id)+','+str(x1)+','+str(y1)+','+str(w)+','+str(h)+',1,-1,-1,-1')
                file.write('\n')
        
        file.close()