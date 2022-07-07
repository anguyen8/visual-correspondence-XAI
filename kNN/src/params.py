class RunningParams(object):
    def __init__(self):
        self.VISUALIZATION = False
        self.UNIFORM = False
        self.MIX_DISTANCE = False
        # Calculate distance of two images based on 5 patches only (not entire image)
        self.CLOSEST_PATCH_COMPARISON = True
        self.IMAGENET_REAL = False
        self.INAT = False
        self.Deformable_ProtoPNet = False

        self.DEEP_NN_TEST = True
        self.KNN_RESULT_SAVE = True

        self.UNEQUAL_W = False

        self.layer4_fm_size = 7

        # Feature space to be used
        self.DIML_FEAT = True
        self.RANDOM_SHUFFLE = True
        self.AP_FEATURE = True
        self.DUPLICATE_THRESHOLD = 0.9

        self.N_test = 50000
        self.K_value = 50
        self.MajorityVotes_K = 20

        if self.VISUALIZATION is True:
            self.AP_FEATURE = True
            self.IMAGENET_REAL = True
            self.CLOSEST_PATCH_COMPARISON = True
            self.K_value = 50
            self.MajorityVotes_K = 20
