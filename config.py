class Model2Config:
    def __init__(
        self,
        d_model = 512,
        layers = 12,
        num_channels = 4,
        moving_avg = 3,
        e_layers = 2,
        d_layers = 2,
        enc_in = 4,
        seq_len = 12,
        pred_len = 3,
        ratio = 0.7, 
        decomp_method = "moving_avg",
        use_norm = 0, 
        down_sampling_window = 2,
        down_sampling_layers = 2,
        down_sampling_method = "conv",
        channel_independence = 0,
        factor = 1,
        dropout = 0.1,
        c_out = 4,
        training = True,
        d_ff = 1024,
        training_batchsize = 2,
        validate_batchsize = 2,
        train_epoch = 20,
        early_stopping = 3,
        lr = 1e-5,
        max_lr = 1e-4,
        step = 10,
        weight_decay = 1e-7,
        **kwargs

    ):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.layers = layers
        self.moving_avg = moving_avg
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ratio = ratio
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.use_norm = use_norm
        self.channel_independence = channel_independence
        self.factor = factor
        self.dropout = dropout
        self.training = training
        self.c_out = c_out
        self.d_ff = d_ff
        self.training_batchsize = training_batchsize
        self.validate_batchsize = validate_batchsize
        self.train_epoch = train_epoch
        self.early_stopping = early_stopping
        self.lr = lr
        self.step = step
        self.weight_decay = weight_decay
        self.max_lr = max_lr


# TimeMixer Kan
class Model3Config: 
    def __init__(
        self,
        d_model = 4,
        num_channels = 1,
        moving_avg = 3,
        layers = 2,
        e_layers = 2,
        enc_in = 4,
        c_out = 4,
        seq_len = 12,
        pred_len = 3,
        ratio = 0.7, 
        decomp_method = "moving_avg",
        use_norm = 1, 
        down_sampling_window = 2,
        down_sampling_layers = 2,
        down_sampling_method = "conv", #
        channel_independence = False,
        factor = 1, #
        dropout = 0.1, #
        training = True, #
        d_ff = 10,
        num_experts = 1,
        training_batchsize = 2,
        validate_batchsize = 2,
        expert_type = 'mlp', # mlp or kan
        control_points = 10, 
        train_epoch = 20,
        early_stopping = 3,
        lr = 1e-5,
        max_lr = 1e-4,
        step = 10,
        weight_decay = 1e-7,
        **kwargs

    ):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.layers = layers
        self.moving_avg = moving_avg
        self.e_layers = e_layers
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ratio = ratio
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.use_norm = use_norm
        self.channel_independence = channel_independence
        self.factor = factor
        self.dropout = dropout
        self.training = training
        self.c_out = c_out
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.training_batchsize = training_batchsize
        self.validate_batchsize = validate_batchsize
        self.expert_type = expert_type
        self.control_points = control_points
        self.train_epoch = train_epoch
        self.early_stopping = early_stopping
        self.lr = lr
        self.step = step
        self.weight_decay = weight_decay
        self.max_lr = max_lr


        