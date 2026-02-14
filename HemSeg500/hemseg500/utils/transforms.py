from monai.transforms import Transform,MapTransform
import numpy as np

class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        """
        初始化CTNormalization转换。
        :param keys: 字典中要转换的键列表
        :param intensity_properties: 包含强度相关属性的字典（均值、标准差、百分位数边界等）
        :param target_dtype: 转换目标的数据类型
        """
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
        """
        在图像上应用CT标准化。
        :param data: 包含图像数据的字典
        :return: 包含标准化图像数据的字典
        """
        d = dict(data)
        for key in self.keys:
            assert self.intensity_properties is not None, "CTNormalizationd requires intensity properties"
            d[key] = d[key].astype(self.target_dtype)
            mean_intensity = self.intensity_properties['mean']
            std_intensity = self.intensity_properties['std']
            lower_bound = self.intensity_properties['percentile_00_5']
            upper_bound = self.intensity_properties['percentile_99_5']
            d[key] = np.clip(d[key], lower_bound, upper_bound)
            d[key] = (d[key] - mean_intensity) / max(std_intensity, 1e-8)
        return d

