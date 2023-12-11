import numpy as np
import trackeval as trackeval
import sys
import joblib
import tqdm

def evaluate_trackers(results_dir, method="phalp", dataset="posetrack", make_video=0):   
    
    if(dataset=="posetrack"): data_gt = joblib.load('_DATA/posetrack/posetrack18_gt_data.pickle')     ; base_dir = "_DATA/posetrack/posetrack18_gt_data/"
    if(dataset=="mupots"):    data_gt = joblib.load('_DATA/mupots/gt_data.pickle')        ; base_dir = "_DATA/mupots/mupots_data/"
    if(dataset=="ava"):       data_gt = joblib.load('_DATA/ava/gt_data.pickle')           ; base_dir = "_DATA/ava/ava_data/"
        
    data_all              = {}
    total_annotated_frames = 0
    total_detected_frames = 0
    if(method=='phalp'):
        for video_ in data_gt.keys():
            print(video_)
            try:
                PHALP_predictions = joblib.load(results_dir + "demo"+ "_" + video_ + ".pkl")

            except: 
                continue
            list_of_gt_frames = np.sort(list(data_gt[video_].keys()))
            tracked_frames    = list(PHALP_predictions.keys())
            data_all[video_]  = {}
            
            for i in range(len(list_of_gt_frames)):
                frame_gt        = list_of_gt_frames[i]
                total_annotated_frames += 1
                
                frame_ = '/home/dobbang/jeon/PoseTrack/val/'+str(video_)+'/'+frame_gt
                if(frame_ in tracked_frames):
                    tracked_data = PHALP_predictions[frame_]
                    
                    if(len(data_gt[video_][frame_gt][0])>0):
                        if(dataset=="ava"):  assert data_gt[video_][frame_gt][0][0][0].split("/")[-1] == frame_gt
                        else:                assert data_gt[video_][frame_gt][0][0].split("/")[-1]    == frame_gt
                        
                        if tracked_data['tracked_bbox'] == []:
                            data_all[video_][frame_] = [data_gt[video_][frame_gt][0], data_gt[video_][frame_gt][1], data_gt[video_][frame_gt][2], data_gt[video_][frame_gt][3], [], [], []]
                        else:
                            data_all[video_][frame_] = [data_gt[video_][frame_gt][0], data_gt[video_][frame_gt][1], data_gt[video_][frame_gt][2], data_gt[video_][frame_gt][3], frame_, tracked_data['tracked_ids'], tracked_data['tracked_bbox']] 
                            total_detected_frames   += 1
                else:
                    data_all[video_][frame_] = [data_gt[video_][frame_gt][0], data_gt[video_][frame_gt][1], data_gt[video_][frame_gt][2], data_gt[video_][frame_gt][3], [], [], []]; print("Error!")
                    data_all[video_][frame_] = [data_gt[video_][frame_gt][0], data_gt[video_][frame_gt][1], data_gt[video_][frame_gt][2], data_gt[video_][frame_gt][3], [], [], []]; print("Error!")
                   
    print("Total annotated frames ", total_annotated_frames)
    print("Total detected frames ", total_detected_frames)
    joblib.dump(data_all, results_dir + '/'+str(dataset)+'_'+str(method)+'.pkl')        

def pkl2txt(results_dir,method,dataset):

    data_all = joblib.load(results_dir + '/'+str(dataset)+'_'+str(method)+'.pkl')        
    txt_path = results_dir+'/txt/'

    for video in tqdm(list(data_all.keys())):
        file = open(txt_path+video+'.txt','w')
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

if __name__ == '__main__':
    
    results_dir = str(sys.argv[1])
    method      = str(sys.argv[2])
    dataset     = str(sys.argv[3])

    evaluate_trackers(results_dir, method=method, dataset=dataset, make_video=0)
    pkl2txt(results_dir,method,dataset)