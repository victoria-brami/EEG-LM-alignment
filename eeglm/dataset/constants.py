import numpy as np
import omegaconf


class BrainMEGElectrodes:

    def __init__(self, use_config="all"):
        # Those correspond to the basic indices: an electrode ID has three sensors placed at 3*ID|3*ID+1|3*ID+2
        self._FRONTAL_ELECTRODE_ID = [1, 8, 16, 17, 28, 30, 31, 42, 50, 11, 18, 19,
                                      29, 33, 32, 43, 9, 20, 34, 35, 44, 10, 23, 21, 36, 45]
        self._LEFT_TEMPORAL_ELECTRODE_ID = [0, 3, 2, 4, 5, 54, 7, 6, 57, 55, 58, 59, 56]
        self._RIGHT_TEMPORAL_ELECTRODE_ID = [51, 46, 47, 53, 52, 49, 48, 98, 90, 91, 101, 99, 100]
        self._PARIETAL_ELECTRODE_ID = [12, 13, 22, 37, 38, 39,
                                       15, 14, 24, 25, 41, 40,
                                       66, 67, 27, 26, 82, 83,
                                       60, 69, 68, 85, 84, 93, 74, 75]
        self._OCCIPITAL_ELECTRODE_ID = [63, 61, 70, 77, 76, 86, 92, 95, 73, 71, 78, 89,
                                        87, 62, 64, 72, 79, 88, 94, 96, 65, 81, 80, 97]

        self._LANGUAGE_ELECTRODE_ID = [0, 1, 2, 3, 4, 7, 11, 54, 55, 56, 57, 58]
        self._LANGUAGE_ELECTRODE_ID = [1, 2, 4, 5, 6, 7, 9, 11, 54, 55, 58, 59]
        self._5_CHANNELS_LANGUAGE_ELECTRODE_ID = [2, 4, 7, 9, 54, 58, 59]
        self._LEFT_HEMISPHERE_ELECTRODE_ID = [8, 16, 18, 1, 11, 9, 10, 0, 3, 2, 4, 5, 12, 13, 54, 7, 6, 15, 14, 57, 55,
                                              58, 59, 66, 67, 56, 63, 61, 60, 69, 70, 73, 64, 62, 65]
        self._RIGHT_HEMISPHERE_ELECTRODE_ID = []

        self.all_areas = ["parietal", "occipital", "frontal", "left_temporal", "right_temporal", "language",
                          "five_channels_language", "left_hemisphere", "right_hemisphere"]
        self._use_config = use_config
        self._use_electrodes = {"x_only": [True, False, False],
                                "y_only": [False, True, False],
                                "mag_only": [False, False, True],
                                "x_and_y": [True, True, False],
                                "all": [True, True, True]}

    @property
    def use_config(self):
        return self._use_config

    @use_config.setter
    def use_config(self, cfg: str):
        self._use_config = cfg

    @property
    def occipital(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._OCCIPITAL_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._OCCIPITAL_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._OCCIPITAL_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._OCCIPITAL_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in self._OCCIPITAL_ELECTRODE_ID]).flatten().tolist()

    @property
    def left_hemisphere(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._LEFT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._LEFT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._LEFT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._LEFT_HEMISPHERE_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in self._LEFT_HEMISPHERE_ELECTRODE_ID]).flatten().tolist()

    @property
    def right_hemisphere(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._RIGHT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._RIGHT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._RIGHT_HEMISPHERE_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._RIGHT_HEMISPHERE_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in self._RIGHT_HEMISPHERE_ELECTRODE_ID]).flatten().tolist()

    @property
    def language(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._LANGUAGE_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in self._LANGUAGE_ELECTRODE_ID]).flatten().tolist()

    @property
    def five_channels_language(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._5_CHANNELS_LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._5_CHANNELS_LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._5_CHANNELS_LANGUAGE_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._5_CHANNELS_LANGUAGE_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in self._5_CHANNELS_LANGUAGE_ELECTRODE_ID]).flatten().tolist()

    @property
    def parietal(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._PARIETAL_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._PARIETAL_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._PARIETAL_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._PARIETAL_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in self._PARIETAL_ELECTRODE_ID]).flatten().tolist()

    @property
    def left_temporal(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._LEFT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._LEFT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._LEFT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._LEFT_TEMPORAL_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in self._LEFT_TEMPORAL_ELECTRODE_ID]).flatten().tolist()

    @property
    def right_temporal(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._RIGHT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._RIGHT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._RIGHT_TEMPORAL_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._RIGHT_TEMPORAL_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array(
                [[3 * i, 3 * i + 1, 3 * i + 2] for i in self._RIGHT_TEMPORAL_ELECTRODE_ID]).flatten().tolist()

    @property
    def frontal(self):
        if self.use_config == "x_only":
            return [3 * i for i in self._FRONTAL_ELECTRODE_ID]
        elif self.use_config == "y_only":
            return [3 * i + 1 for i in self._FRONTAL_ELECTRODE_ID]
        elif self.use_config == "mag_only":
            return [3 * i + 2 for i in self._FRONTAL_ELECTRODE_ID]
        elif self.use_config == "x_and_y":
            return np.array([[3 * i, 3 * i + 1] for i in self._FRONTAL_ELECTRODE_ID]).flatten().tolist()
        else:
            return np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in self._FRONTAL_ELECTRODE_ID]).flatten().tolist()

    def get_electrodes(self, areas):
        if not isinstance(areas, list) and not isinstance(areas, omegaconf.listconfig.ListConfig):
            areas = [areas]
        # print(set(areas).intersection(set(self.all_areas)),set(self.all_areas), set(areas))
        assert set(areas).intersection(set(self.all_areas)) == set(
            areas), f"{len(areas) - len(set(areas).intersection(self.all_areas))} brain regions not found"
        return np.concatenate([getattr(self, area) for area in areas])


