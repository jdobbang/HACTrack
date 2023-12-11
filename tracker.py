"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import

import numpy as np
import torch

from . import linear_assignment
from .track import Track
from filterpy.kalman import KalmanFilter
import cv2
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
from pdb import set_trace

def linear_assignment2(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
    
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    detections_bbox = []
    trackers_bbox = []

    for det in range(len(detections)):
        x1 = detections[det].__dict__['tlwh'][0]
        y1 = detections[det].__dict__['tlwh'][1]
        x2 = detections[det].__dict__['tlwh'][0]+detections[det].__dict__['tlwh'][2]
        y2 = detections[det].__dict__['tlwh'][1]+detections[det].__dict__['tlwh'][3]
        detections_bbox.append([x1,y1,x2,y2])
    detections = np.array(detections_bbox)

    for trk in range(len(trackers)):
       x1 = trackers[trk].__dict__['track_data']['history'][-1]['kalman_bbox'][0][0]
       y1 = trackers[trk].__dict__['track_data']['history'][-1]['kalman_bbox'][0][1]
       x2 = trackers[trk].__dict__['track_data']['history'][-1]['kalman_bbox'][0][2]
       y2 = trackers[trk].__dict__['track_data']['history'][-1]['kalman_bbox'][0][3]
       
       trackers_bbox.append([x1,y1,x2,y2])
    trackers = np.array(trackers_bbox)
    
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment2(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)



class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg              = cfg
        self.metric           = metric
        self.max_age          = max_age
        self.n_init           = n_init
        self.tracks           = []
        self._next_id         = 1
        self.tracked_cost     = {}
        self.phalp_tracker    = phalp_tracker
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)

    def update(self, detections, frame_t, image_name, shot):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)
        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1], statistics[2], statistics[3], statistics[4]] 
        if(self.cfg.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx, shot)
        self.accumulate_vectors([i[0] for i in matches], features=self.cfg.phalp.predict)
 
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(unmatched_tracks, features=self.cfg.phalp.predict)
    
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
                         
            appe_features += [track.track_data['prediction']['appe'][-1]]
            loca_features += [track.track_data['prediction']['loca'][-1]]
            pose_features += [track.track_data['prediction']['pose'][-1]]
            uv_maps       += [track.track_data['prediction']['uv'][-1]]
            targets       += [track.track_id]
            
            
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        
        return matches
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb          = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb          = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i].detection_data['pose'] for i in detection_indices])
            uv_maps           = np.array([dets[i].detection_data['uv'] for i in detection_indices])
            targets           = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix       = self.metric.distance([appe_emb, loca_emb, pose_emb, uv_maps], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)


        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d.detection_data['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        if(self.cfg.use_gt): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection.detection_data, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1

    def accumulate_vectors(self, track_ids, features="APL"):
        
        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.cfg.phalp.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.cfg.phalp.track_history)]))
            is_tracks = 1

        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        if(is_tracks):
            with torch.no_grad():
                if("P" in features): p_pred = self.phalp_tracker.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.phalp_tracker.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None, 
                                                     loca=l_pred[p_id] if("L" in features) else None)
                

class ByteTracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg              = cfg
        self.metric           = metric
        self.max_age          = max_age
        self.n_init           = n_init
        self.tracks           = []
        self._next_id         = 1
        self.tracked_cost     = {}
        self.phalp_tracker    = phalp_tracker
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)
            track.__dict__['track_data']['history'][-1]['kalman_bbox'] = track.__dict__['track_data']['history'][-1]['kalman_tracker'].predict()

    def update(self, detections, frame_t, image_name, shot):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)

        #update matched tracker with assgined detections
        for i in range(len(matches)):
            trk_idx, det_idx = matches[i]
            track_ =  self.tracks[trk_idx]
            detection_ =  detections[det_idx]
            
            x1 = detection_.__dict__['tlwh'][0]
            y1 = detection_.__dict__['tlwh'][1]
            x2 = detection_.__dict__['tlwh'][2]+detection_.__dict__['tlwh'][0]
            y2 = detection_.__dict__['tlwh'][3]+detection_.__dict__['tlwh'][1]
            track_.__dict__['track_data']['history'][-1]['kalman_tracker'].update([x1,y1,x2,y2])
            detection_.__dict__['detection_data']['kalman_tracker'] = track_.__dict__['track_data']['history'][-1]['kalman_tracker']

        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1], statistics[2], statistics[3], statistics[4]] 
        if(self.cfg.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx, shot)
        self.accumulate_vectors([i[0] for i in matches], features=self.cfg.phalp.predict)
 
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(unmatched_tracks, features=self.cfg.phalp.predict)
    
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
            appe_features += [track.track_data['prediction']['appe'][-1]]
            loca_features += [track.track_data['prediction']['loca'][-1]]
            pose_features += [track.track_data['prediction']['pose'][-1]]
            uv_maps       += [track.track_data['prediction']['uv'][-1]]
            """
            import cv2
            
            for i in range(len(uv_maps)):
                mean = np.array([123.675, 116.280, 103.530])
                std = np.array([58.395, 57.120, 57.375])
                uv_image = uv_maps[i]
                uv_image        = uv_image[:3, :, :]
                
                uv_0 = (uv_image[0]*5*std[0]+mean[0]).astype(np.uint8)
                uv_1 = (uv_image[1]*5*std[1]+mean[1]).astype(np.uint8)
                uv_2 = (uv_image[2]*5*std[2]+mean[2]).astype(np.uint8)
                uv_vis = cv2.merge((uv_0,uv_1,uv_2))

                storage_name = '/home/dobbang/jeon/4dhumans/nerf/pre/'+str(i)+'_'+str(track)+'_'+str(image_name).split('/')[-1]
                
                cv2.imwrite(storage_name,uv_vis)
            """
            targets       += [track.track_id]
            
            
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        
        return matches
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb          = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb          = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i].detection_data['pose'] for i in detection_indices])
            uv_maps           = np.array([dets[i].detection_data['uv'] for i in detection_indices])
            targets           = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix       = self.metric.distance([appe_emb, loca_emb, pose_emb, uv_maps], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        ############################split detection by its confidence score and save its location######################################
        matches = []
        unmatched_tracks = []
        unmatched_detections= []
        
        DH = []
        DL = []
        DH_loc = []
        DL_loc = []

        #detection의 bbox는 tlwh 형태
        for d in range(len(detections)):
            c_h = self.cfg.phalp.low_th_c
            if detections[d].__dict__['detection_data']['conf'] >= c_h:
                DH.append(detections[d])
                DH_loc.append((d,len(DH)-1)) # (detection 전체에서의 index, DH에서의 index)
            if detections[d].__dict__['detection_data']['conf']  < c_h :
                DL.append(detections[d])
                DL_loc.append((d,len(DL)-1)) # (detection 전체에서의 index, DL에서의 index)
        DH = np.array(DH)
        DL = np.array(DL)

        DH_loc = np.array(DH_loc)
        DL_loc = np.array(DL_loc)
        ##########################################################################################################

        #####################################first association####################################################
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        # matches는 (track idx, det idx) 순서임에 주의하기/ match 결과에 high 붙이는게 변수 관리에 편함/cost_matrix는 신경 안써도 됨
        matches_high, unmatched_tracks_high, unmatched_detections_high, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, DH, confirmed_tracks)

        #second association에서 쓸 self.track        
        c_tracks = []
        for i in range(len(confirmed_tracks)):
            c_tracks.append(self.tracks[confirmed_tracks[i]])

        matches = matches_high
        unmatched_tracks = unmatched_tracks_high
        unmatched_detections = unmatched_detections_high

        #####################################second association####################################################
        if len(unmatched_tracks_high)>0 and len(DL) >0:
            ####################2nd tracklet###################### 
            tracks_left = []
            tracks_left_loc = []
            for u in range(len(unmatched_tracks_high)):
                tracks_left.append(c_tracks[int(unmatched_tracks_high[u])])
                tracks_left_loc.append(((int(unmatched_tracks_high[u])),len(tracks_left)-1))#trks_left가 trks에서 어떤 index, ttrack_left에서의 idx
            tracks_left=np.array(tracks_left)
            tracks_left_loc=np.array(tracks_left_loc)

            # matches >> (detection idx, tracklets idx) 순서임
            # bbox 입력 형태 (x1,y1,x2,y2)
            matches_low, unmatched_detections_low, unmatched_tracks_low = associate_detections_to_trackers(DL,tracks_left, 0.3)

            for m_l in range(len(matches_low)): # 2nd association에서의 matches를 추가
                track_loc = tracks_left_loc[int(np.where(tracks_left_loc[:,1]==matches_low[m_l][1])[0][0])][0]
                detection_loc = DL_loc[int(np.where(DL_loc[:,1]==matches_low[m_l][0])[0][0])][0]
                matches.append((track_loc,detection_loc))

            for u_t_l in range(len(unmatched_tracks_low)):
                u_track_loc = tracks_left_loc[int(np.where(tracks_left_loc[:,1]==unmatched_tracks_low[u_t_l])[0][0])][0]
                unmatched_tracks.append(u_track_loc)
    
    ############################################################################################################

        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d.detection_data['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        if(self.cfg.use_gt): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection.detection_data, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        ###############################kalman tracker initiate###############################################
        x1 = detection.__dict__['tlwh'][0]
        y1 = detection.__dict__['tlwh'][1]
        x2 = detection.__dict__['tlwh'][2]+detection.__dict__['tlwh'][0]
        y2 = detection.__dict__['tlwh'][3]+detection.__dict__['tlwh'][1]
        new_track.__dict__['track_data']['history'][-1]['kalman_tracker']=KalmanBoxTracker([x1,y1,x2,y2])
        self.tracks.append(new_track)
        self._next_id += 1

    def accumulate_vectors(self, track_ids, features="APL"):
        
        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.cfg.phalp.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.cfg.phalp.track_history)]))
            is_tracks = 1

        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        if(is_tracks):
            with torch.no_grad():
                if("P" in features): p_pred = self.phalp_tracker.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.phalp_tracker.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None, 
                                                     loca=l_pred[p_id] if("L" in features) else None)
                


class HACTracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg              = cfg
        self.metric           = metric
        self.max_age          = max_age
        self.n_init           = n_init
        self.tracks           = []
        self._next_id         = 1
        self.tracked_cost     = {}
        self.phalp_tracker    = phalp_tracker
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)
            track.__dict__['track_data']['history'][-1]['kalman_bbox'] = track.__dict__['track_data']['history'][-1]['kalman_tracker'].predict()

    def update(self, detections, frame_t, image_name, shot):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)

        #update matched tracker with assgined detections
        for i in range(len(matches)):
            trk_idx, det_idx = matches[i]
            track_ =  self.tracks[trk_idx]
            detection_ =  detections[det_idx]
            
            x1 = detection_.__dict__['tlwh'][0]
            y1 = detection_.__dict__['tlwh'][1]
            x2 = detection_.__dict__['tlwh'][2]+detection_.__dict__['tlwh'][0]
            y2 = detection_.__dict__['tlwh'][3]+detection_.__dict__['tlwh'][1]
            track_.__dict__['track_data']['history'][-1]['kalman_tracker'].update([x1,y1,x2,y2])
            detection_.__dict__['detection_data']['kalman_tracker'] = track_.__dict__['track_data']['history'][-1]['kalman_tracker']

        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1], statistics[2], statistics[3], statistics[4]] 
        if(self.cfg.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx, shot)
        self.accumulate_vectors([i[0] for i in matches], features=self.cfg.phalp.predict)
 
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(unmatched_tracks, features=self.cfg.phalp.predict)
    
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
            appe_features += [track.track_data['prediction']['appe'][-1]]
            loca_features += [track.track_data['prediction']['loca'][-1]]
            pose_features += [track.track_data['prediction']['pose'][-1]]
            uv_maps       += [track.track_data['prediction']['uv'][-1]]

            targets       += [track.track_id]
            
            
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        
        return matches
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb          = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb          = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i].detection_data['pose'] for i in detection_indices])
            uv_maps           = np.array([dets[i].detection_data['uv'] for i in detection_indices])
            targets           = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix       = self.metric.distance([appe_emb, loca_emb, pose_emb, uv_maps], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        ############################split detection by its confidence score and save its location######################################
        matches = []
        unmatched_tracks = []
        unmatched_detections= []
        
        DH = []
        DL = []
        DH_loc = []
        DL_loc = []

        ####################################################################################################################################################################################
        #json_path = '/home/dobbang/jeon/4dhumans/detections/'+
        #detection의 bbox는 tlwh 형태
        for d in range(len(detections)):
            c_h = self.cfg.phalp.low_th_c
            if detections[d].__dict__['detection_data']['conf'] >= c_h:
                DH.append(detections[d])
                DH_loc.append((d,len(DH)-1)) # (detection 전체에서의 index, DH에서의 index)
            if detections[d].__dict__['detection_data']['conf']  < c_h :
                DL.append(detections[d])
                DL_loc.append((d,len(DL)-1)) # (detection 전체에서의 index, DL에서의 index)
        DH = np.array(DH)
        DL = np.array(DL)
        ####################################################################################################################################################################################

        DH_loc = np.array(DH_loc)
        DL_loc = np.array(DL_loc)
        ##########################################################################################################

        #####################################first association####################################################
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        # matches는 (track idx, det idx) 순서임에 주의하기/ match 결과에 high 붙이는게 변수 관리에 편함/cost_matrix는 신경 안써도 됨
        matches_high, unmatched_tracks_high, unmatched_detections_high, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, DH, confirmed_tracks)

        #second association에서 쓸 self.track        
        c_tracks = []
        for i in range(len(confirmed_tracks)):
            c_tracks.append(self.tracks[confirmed_tracks[i]])

        matches = matches_high
        unmatched_tracks = unmatched_tracks_high
        unmatched_detections = unmatched_detections_high

        #####################################second association####################################################
        if len(unmatched_tracks_high)>0 and len(DL) >0:
            ####################2nd tracklet###################### 
            tracks_left = []
            tracks_left_loc = []
            for u in range(len(unmatched_tracks_high)):
                tracks_left.append(c_tracks[int(unmatched_tracks_high[u])])
                tracks_left_loc.append(((int(unmatched_tracks_high[u])),len(tracks_left)-1))#trks_left가 trks에서 어떤 index, ttrack_left에서의 idx
            tracks_left=np.array(tracks_left)
            tracks_left_loc=np.array(tracks_left_loc)

            # matches >> (detection idx, tracklets idx) 순서임
            # bbox 입력 형태 (x1,y1,x2,y2)
            matches_low, unmatched_detections_low, unmatched_tracks_low = associate_detections_to_trackers(DL,tracks_left, 0.3)

            for m_l in range(len(matches_low)): # 2nd association에서의 matches를 추가
                track_loc = tracks_left_loc[int(np.where(tracks_left_loc[:,1]==matches_low[m_l][1])[0][0])][0]
                detection_loc = DL_loc[int(np.where(DL_loc[:,1]==matches_low[m_l][0])[0][0])][0]
                matches.append((track_loc,detection_loc))

            for u_t_l in range(len(unmatched_tracks_low)):
                u_track_loc = tracks_left_loc[int(np.where(tracks_left_loc[:,1]==unmatched_tracks_low[u_t_l])[0][0])][0]
                unmatched_tracks.append(u_track_loc)

            #for u_d_l in range(len(unmatched_detections_low)):
            #    u_detection_loc = DL_loc[int(np.where(DL_loc[:,1]==unmatched_detections_low[u_d_l])[0][0])][0]
            #    unmatched_detections.append(u_detection_loc)
    
    ############################################################################################################

        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d.detection_data['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        if(self.cfg.use_gt): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection.detection_data, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        
        ###############################kalman tracker initiate###############################################
        x1 = detection.__dict__['tlwh'][0]
        y1 = detection.__dict__['tlwh'][1]
        x2 = detection.__dict__['tlwh'][2]+detection.__dict__['tlwh'][0]
        y2 = detection.__dict__['tlwh'][3]+detection.__dict__['tlwh'][1]
        new_track.__dict__['track_data']['history'][-1]['kalman_tracker']=KalmanBoxTracker([x1,y1,x2,y2])
        self.tracks.append(new_track)
        self._next_id += 1

    def accumulate_vectors(self, track_ids, features="APL"):
        
        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.cfg.phalp.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.cfg.phalp.track_history)]))
            is_tracks = 1

        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        if(is_tracks):
            with torch.no_grad():
                if("P" in features): p_pred = self.phalp_tracker.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.phalp_tracker.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None, 
                                                     loca=l_pred[p_id] if("L" in features) else None)