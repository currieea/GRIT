from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

from .wilds_cf_dataset import WILDSCFDataset  # Ensure this is imported first for the base class
from .celeba import CounterfactualCelebADataset, CounterfactualCelebAClipDataset
from .colored_mnist import ColoredMNISTDataset, ColoredMNISTClipDataset
from .rotated_mnist import RotatedMNISTDataset, RotatedMNISTClipDataset
from .pacs import PACSDataset, PACSClipDataset
from .waterbirds import CounterfactualWaterbirdsDataset, CounterfactualWaterbirdsClipDataset
from .camelyon import Camelyon17Dataset, Camelyon17ClipDataset
from .lisa_cmnist import LISAColoredMNISTDataset, LISAColoredMNISTClipDataset

__all__ = [
    'WaterbirdsDataset',
    'CounterfactualCelebADataset',
    'CounterfactualCelebAClipDataset',
    'ColoredMNISTDataset',
    'ColoredMNISTClipDataset',
    'RotatedMNISTDataset',
    'RotatedMNISTClipDataset',
    'PACSDataset',
    'PACSClipDataset',
    'CounterfactualWaterbirdsDataset',
    'CounterfactualWaterbirdsClipDataset',
    'Camelyon17Dataset',
    'Camelyon17ClipDataset',
    'WILDSCFDataset',
    'LISAColoredMNISTDataset',
    'LISAColoredMNISTClipDataset'
]