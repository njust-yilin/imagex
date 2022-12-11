class UIQueueElement(object):
    def __init__(self,
        image_index:int,
        mask_index:int,
        image_hwc:tuple,
        timestamp:int,
        camera_id:str,
        part_id:str,
        part_timestamp:int,
        network:str,
        capture_config:str,
        capture_index:int,
        ok:bool
    ):
        self.image_index = image_index
        self.mask_index = mask_index
        self.image_hwc = image_hwc
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.image_index = image_index
        self.mask_index = mask_index
        self.image_hwc = image_hwc
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.part_id = part_id
        self.part_timestamp = part_timestamp
        self.network = network
        self.capture_config = capture_config
        self.capture_index = capture_index
        self.ok = ok
    
    def __str__(self) -> str:
        return str(self.__dict__)