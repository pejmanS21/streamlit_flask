"""Basic Classes to be overridden within the framework"""


class BaseModel(object):
    """Base models that only implements a loading func, (e.g. classification regression whatever)"""
    pass

    def __load_model(self, *args, **kwargs):
        """ model loader """
        pass


# e.g. this could be SegmentationModel where the predict method returns an image.
class SegmentationModel(BaseModel):
    """Classification Base class"""

    def __load_model(self, *args, **kwargs):
        """
        Base model setter.
        """
        raise NotImplementedError

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photos to make the prediction on.
        :return: predicted class.
        """
        raise NotImplementedError

