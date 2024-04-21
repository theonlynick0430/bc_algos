import bc_algos.envs.env_base as EB
import numpy as np


class IsaacGymEnv(EB.BaseEnv):

    @classmethod
    def preprocess_img(cls, img):
        """
        Helper function to preprocess images from Isaac Gym environment. 
        Specifically does the following:
        1) Removes alpha (last) channel from @img.
        2) Changes shape of @img from [H, W, 3] to [3, H, W]
        3) Changes scale of @img from [0, 255] to [0, 1]

        Args: 
            img (np.array): image data of shape [..., H, W, 4]

        Returns: preprocessed @img of shape [..., 3, H, W].
        """
        img = img[:, :, :-1]
        img = np.moveaxis(img.astype(float), -1, -3)
        img /= 255.
        return img.clip(0., 1.)
