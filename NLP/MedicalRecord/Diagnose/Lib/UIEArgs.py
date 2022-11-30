class uie_args:
    def __init__(self, model_path_prefix, schema, max_seq_len=512, position_prob=0.5, batch_size=4, device='cpu', device_id=0):
        self.model_path_prefix = model_path_prefix
        self.position_prob = position_prob
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.device_id = device_id
        self.schema = schema
