class ImageXQueueElement(object):
    def __init__(self,
        image_index:int,
        mask_index:int,
        timestamp:int,
        camera_id:str,
    ):
        self.image_index = image_index
        self.mask_index = mask_index
        self.timestamp = timestamp
        self.camera_id = camera_id

    def __str__(self) -> str:
        return str(self.__dict__)