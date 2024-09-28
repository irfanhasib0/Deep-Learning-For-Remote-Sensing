# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BioMassters Dataset."""

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, percentile_normalization


class BioMassters(NonGeoDataset):
    """BioMassters Dataset for Aboveground Biomass prediction.

    Dataset intended for Aboveground Biomass (AGB) prediction
    over Finnish forests based on Sentinel 1 and 2 data with
    corresponding target AGB mask values generated by Light Detection
    and Ranging (LiDAR).

    Dataset Format:

    * .tif files for Sentinel 1 and 2 data
    * .tif file for pixel wise AGB target mask
    * .csv files for metadata regarding features and targets

    Dataset Features:

    * 13,000 target AGB masks of size (256x256px)
    * 12 months of data per target mask
    * Sentinel 1 and Sentinel 2 data for each location
    * Sentinel 1 available for every month
    * Sentinel 2 available for almost every month
      (not available for every month due to ESA aquisition halt over the region
      during particular periods)

    If you use this dataset in your research, please cite the following paper:

    * https://nascetti-a.github.io/BioMasster/

    .. versionadded:: 0.5
    """

    valid_splits = ('train', 'test')
    valid_sensors = ('S1', 'S2')

    metadata_filename = 'The_BioMassters_-_features_metadata.csv.csv'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        sensors: Sequence[str] = ['S1', 'S2'],
        as_time_series: bool = False,
    ) -> None:
        """Initialize a new instance of BioMassters dataset.

        If ``as_time_series=False`` (the default), each time step becomes its own
        sample with the target being shared across multiple samples.

        Args:
            root: root directory where dataset can be found
            split: train or test split
            sensors: which sensors to consider for the sample, Sentinel 1 and/or
                Sentinel 2 ('S1', 'S2')
            as_time_series: whether or not to return all available
                time-steps or just a single one for a given target location

        Raises:
            AssertionError: if ``split`` or ``sensors`` is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root

        assert (
            split in self.valid_splits
        ), f'Please choose one of the valid splits: {self.valid_splits}.'
        self.split = split

        assert set(sensors).issubset(
            set(self.valid_sensors)
        ), f'Please choose a subset of valid sensors: {self.valid_sensors}.'
        self.sensors = sensors
        self.as_time_series = as_time_series

        self._verify()

        # open metadata csv files
        self.df = pd.read_csv(os.path.join(self.root, self.metadata_filename))

        # filter sensors
        self.df = self.df[self.df['satellite'].isin(self.sensors)]

        # filter split
        self.df = self.df[self.df['split'] == self.split]

        # generate numerical month from filename since first month is September
        # and has numerical index of 0
        self.df['num_month'] = (
            self.df['filename']
            .str.split('_', expand=True)[2]
            .str.split('.', expand=True)[0]
            .astype(int)
        )

        # set dataframe index depending on the task for easier indexing
        if self.as_time_series:
            self.df['num_index'] = self.df.groupby(['chip_id']).ngroup()
        else:
            filter_df = (
                self.df.groupby(['chip_id', 'month'])['satellite'].count().reset_index()
            )
            filter_df = filter_df[filter_df['satellite'] == len(self.sensors)].drop(
                'satellite', axis=1
            )
            # guarantee that each sample has corresponding number of images available
            self.df = self.df.merge(filter_df, on=['chip_id', 'month'], how='inner')

            self.df['num_index'] = self.df.groupby(['chip_id', 'month']).ngroup()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """
        sample_df = self.df[self.df['num_index'] == index].copy()

        # sort by satellite and month to return correct order
        sample_df.sort_values(
            by=['satellite', 'num_month'], inplace=True, ascending=True
        )

        filepaths = sample_df['filename'].tolist()
        sample: dict[str, Tensor] = {}
        for sens in self.sensors:
            sens_filepaths = [fp for fp in filepaths if sens in fp]
            sample[f'image_{sens}'] = self._load_input(sens_filepaths)

        if self.split == 'train':
            sample['label'] = self._load_target(
                sample_df['corresponding_agbm'].unique()[0]
            )

        return sample

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df['num_index'].unique())

    def _load_input(self, filenames: list[Path]) -> Tensor:
        """Load the input imagery at the index.

        Args:
            filenames: list of filenames corresponding to input

        Returns:
            input image
        """
        filepaths = [
            os.path.join(self.root, f'{self.split}_features', f) for f in filenames
        ]
        arr_list = [rasterio.open(fp).read() for fp in filepaths]
        if self.as_time_series:
            arr = np.stack(arr_list, axis=0)
        else:
            arr = np.concatenate(arr_list, axis=0)
        return torch.tensor(arr.astype(np.int32))

    def _load_target(self, filename: Path) -> Tensor:
        """Load the target mask at the index.

        Args:
            filename: filename of target to index

        Returns:
            target mask
        """
        with rasterio.open(os.path.join(self.root, 'train_agbm', filename), 'r') as src:
            arr: np.typing.NDArray[np.float64] = src.read()

        target = torch.from_numpy(arr).float()
        return target

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        exists = []

        filenames = [f'{self.split}_features', self.metadata_filename]
        for filename in filenames:
            pathname = os.path.join(self.root, filename)
            exists.append(os.path.exists(pathname))
        if all(exists):
            return

        raise DatasetNotFoundError(self)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = len(self.sensors) + 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(1, ncols=ncols, figsize=(5 * ncols, 10))
        for idx, sens in enumerate(self.sensors):
            img = sample[f'image_{sens}'].numpy()
            if self.as_time_series:
                # plot last time step
                img = img[-1, ...]
            if sens == 'S2':
                img = img[[2, 1, 0], ...]
                img = percentile_normalization(img.transpose(1, 2, 0))
            else:
                co_polarization = img[0]  # transmit == receive
                cross_polarization = img[1]  # transmit != receive
                ratio = co_polarization / cross_polarization

                # https://gis.stackexchange.com/a/400780/123758
                co_polarization = np.clip(co_polarization / 0.3, a_min=0, a_max=1)
                cross_polarization = np.clip(
                    cross_polarization / 0.05, a_min=0, a_max=1
                )
                ratio = np.clip(ratio / 25, a_min=0, a_max=1)

                img = np.stack((co_polarization, cross_polarization, ratio), axis=-1)

            axs[idx].imshow(img)
            axs[idx].axis('off')
            if show_titles:
                axs[idx].set_title(sens)

        if showing_predictions:
            pred = axs[ncols - 2].imshow(
                sample['prediction'].permute(1, 2, 0), cmap='YlGn'
            )
            plt.colorbar(pred, ax=axs[ncols - 2], fraction=0.046, pad=0.04)
            axs[ncols - 2].axis('off')
            if show_titles:
                axs[ncols - 2].set_title('Prediction')

        # plot target / only available in train set
        if 'label' in sample:
            target = axs[-1].imshow(sample['label'].permute(1, 2, 0), cmap='YlGn')
            plt.colorbar(target, ax=axs[-1], fraction=0.046, pad=0.04)
            axs[-1].axis('Off')
            if show_titles:
                axs[-1].set_title('Target')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
