import os


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    # model_path = "leaderboard/team_code/interfuser.pth.tar"
    model_path = "interfuser/output/baseline_80w/checkpoint-6.pth.tar"
    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04
    
    seq_len = 1

    model = "interfuser_baseline"

    def __init__(self, my_param = None, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.model = my_param['ckpt_name']
        self.model_path = my_param['ckpt_path']
        
        if self.model[:3]=='tcp':

            self.turn_KP = 0.75
            self.turn_KI = 0.75
            self.turn_KD = 0.3
            self.turn_n = 40 # buffer size

            self.speed_KP = 5.0
            self.speed_KI = 0.5
            self.speed_KD = 1.0
            self.speed_n = 40 # buffer size

            self.max_throttle = 0.75 # upper limit on throttle signal value in dataset
            self.brake_speed = 0.4 # desired speed below which brake is triggered
            self.brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
            self.clip_delta = 0.25 # maximum change in speed input to logitudinal controller
            
            
            self.aim_dist = 4.0 # distance to search around for aim point
            self.angle_thresh = 0.3 # outlier control detection angle
            self.dist_thresh = 10 # target point y-distance for outlier filtering


            self.speed_weight = 0.05
            self.value_weight = 0.001
            self.features_weight = 0.05
