"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        aug_backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.aug_backbone = aug_backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        aug_feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.aug_backbone, self.layers_to_extract_from, self.device
        )

        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # feature_dimensions = [512, 1024, 512]
        aug_feature_dimensions = aug_feature_aggregator.feature_dimensions(input_shape)
        # aug_feature_dimensions = [512, 1024, 512]
        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.forward_modules["aug_feature_aggregator"] = aug_feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def merege(self, features, aug_features, method):
        if method == "add":

            features['layer2'] = (features['layer2'] + aug_features['layer2'])/2
            features['layer3'] = (features['layer3'] + aug_features['layer3'])/2
        if method == "pca":
            from sklearn import decomposition

            features['layer2'] = features['layer2'].reshape(-1, 512)
            aug_features['layer2'] = aug_features['layer2'].reshape(-1, 512)

            features['layer2'] = torch.cat((features['layer2'], aug_features['layer2']), dim=1)
            pca1 = decomposition.PCA(n_components=512)
            pca1.fit(features['layer2'].data.cpu())
            features['layer2'] = pca1.transform(features['layer2'].data.cpu())
            features['layer2'] = torch.tensor(features['layer2'])
            features['layer2'] = features['layer2'].reshape(-1, 512, 28, 28)
            features['layer2'] = features['layer2'].to(torch.float).to(self.device)

            features['layer3'] = features['layer3'].reshape(-1, 1024)
            aug_features['layer3'] = aug_features['layer3'].reshape(-1, 1024)
            features['layer3'] = torch.cat((features['layer3'], aug_features['layer3']), dim=1)
            pca2 = decomposition.PCA(n_components=1024)
            pca2.fit(features['layer3'].data.cpu())
            features['layer3'] = pca2.transform(features['layer3'].data.cpu())
            features['layer3'] = torch.tensor(features['layer3'])
            features['layer3'] = features['layer3'].reshape(-1, 1024, 14, 14)
            features['layer3'] = features['layer3'].to(torch.float).to(self.device)

        if method =="concat":
            features['layer2'] = torch.cat((features['layer2'], aug_features['layer2']), dim=1)
            features['layer3'] = torch.cat((features['layer3'], aug_features['layer3']), dim=1)

        if method == "pooling":
            batsize = features['layer2'].shape[0]
            h = features['layer2'].shape[2]
            w = features['layer2'].shape[3]
            features['layer2'] = torch.cat((features['layer2'], aug_features['layer2']), dim=1)
            a = features['layer2'].reshape(-1, 1024)
            features['layer2'] = F.adaptive_avg_pool1d(a, 512)
            features['layer2'] = features['layer2'].reshape(batsize, -1, h, w)

            features['layer3'] = torch.cat((features['layer3'], aug_features['layer3']), dim=1)
            batsize1 = features['layer3'].shape[0]
            h1 = features['layer3'].shape[2]
            w1 = features['layer3'].shape[3]
            b = features['layer3'].reshape(-1, 2048)
            features['layer3'] = F.adaptive_avg_pool1d(b, 1024)
            features['layer3'] = features['layer3'].reshape(batsize1, -1, h1, w1)

        return features
    def _embed(self, images, aug_images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
            aug_features = self.forward_modules["aug_feature_aggregator"](aug_images)
            method = "add"

            # features = self.merege(features, aug_features, method)
        # features['layer3'] = (features['layer3'] + features['histogram_layer']) / 2
        # features = [features[layer] for layer in self.layers_to_extract_from[:2]]

        def com(features):
            features = [features[layer] for layer in self.layers_to_extract_from]

            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]

            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]

                # TODO(pgehler): Add comments
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]

            # As different feature backbones & patching provide differently
            # sized features, these are brought into the correct form here.

            features = self.forward_modules["preprocessing"](features)
            features = self.forward_modules["preadapt_aggregator"](features)
            return features, patch_shapes

        features, patch_shapes = com(features)
        # aug_features, _ = com(aug_features)

        # features = torch.cat((features, aug_features), dim=0)
        # features = (features + aug_features)/2
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data, aug_training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, aug_training_data)

    def _fill_memory_bank(self, input_data, aug_input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image, aug_input_data):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                aug_input_data = aug_input_data.to(torch.float).to(self.device)
                return self._embed(input_image, aug_input_data)

        features = []
        # features_aug = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image, aug_image in zip(data_iterator, aug_input_data):
                if isinstance(image, dict):
                    image = image["image"]
                if isinstance(aug_image, dict):
                    aug_image = aug_image["image"]
                features.append(_image_to_features(image, aug_image))

        # with tqdm.tqdm(
        #         input_data, desc="Computing support features...", position=1, leave=False
        # ) as data_iterator:
        #     for image, aug_image in zip(data_iterator, aug_input_data):
        #         if isinstance(image, dict):
        #             image = image["image"]
        #         if isinstance(aug_image, dict):
        #             aug_image = aug_image["image"]
        #         features_aug.append(_image_to_features(aug_image, image))


        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        # features_aug = np.concatenate(features_aug, axis=0)
        # features_aug = self.featuresampler.run(features_aug)

        # features = torch.cat((torch.tensor(features), torch.tensor(features_aug)), dim=0)
        # features = np.array(features)

        # features = np.concatenate(features, axis=0)
        # features_aug = np.concatenate(features_aug, axis=0)
        # features = torch.cat((torch.tensor(features), torch.tensor(features_aug)), dim=0)
        # features = np.array(features)
        #
        # features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data, aug_data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, aug_data)
        return self._predict(data, aug_data)

    def _predict_dataloader(self, dataloader, aug_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image, aug_image in zip(data_iterator, aug_dataloader):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                if isinstance(aug_image, dict):
                    aug_image = aug_image["image"]

                _scores, _masks = self._predict(image, aug_image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images, aug_images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        aug_images = aug_images.to(torch.float).to(self.device)

        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, aug_images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)

        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
