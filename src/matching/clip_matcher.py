from pathlib import Path
import numpy as np
import requests
import logging
import torch
import time
import os
from video.embedder import CLIPVideoEmbedder
from matching.algorithms import MeanMatcher

logger = logging.getLogger(__name__)


class CLIPMatcher:
    """CLIP-based matcher for images and videos in a specified folder."""

    def __init__(
        self,
        image_video_folder: str,
        top_k: int = 10,
        port: int = 5000,
        frames_per_video_clip_max: int = None,
    ) -> None:
        self.image_video_folder = image_video_folder
        self.embedding_folder = Path("data/.embeddings")
        self.top_k = top_k
        self.port = port
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = f"http://localhost:{self.port}"
        self.all_embeddings = None
        self.all_filenames_time_stamps = None

        model_name = requests.get(
            f"http://localhost:{self.port}/clip/model_name", timeout=10).json()['clip_model_name']
        model_name_safe = model_name.replace("/", "_")
        self.clip_video_embedder = CLIPVideoEmbedder(
            frames_per_video_clip_max, self.server_url)

        self.embedding_file = self.embedding_folder / \
            f"{model_name_safe}_embeddings.npy"
        self.video_embedding_file = self.embedding_folder / \
            f"{model_name_safe}_embeddings.npy"
        self.filename_file = self.embedding_folder / \
            f"{model_name_safe}_filenames.npy"
        self.video_timestamp_file = self.embedding_folder / \
            f"{model_name_safe}_timestamps.npy"

        # Compute embeddings of existing img/videos in gallery or load embeddings from existing file
        if self.embedding_file.exists() and self.filename_file.exists():
            self.image_embeddings, self.image_filenames = self.load_embeddings()
        else:
            self.image_embeddings, self.image_filenames = self.compute_embeddings(
                self.get_all_filepaths())

        if self.video_embedding_file.exists() and self.video_timestamp_file.exists():
            self.video_embeddings, self.video_timestamps = self.load_video_embeddings()
        else:
            self.video_embeddings, self.video_timestamps = self.compute_video_embeddings(
                self.get_all_filepaths())

        self.save()  # save embeddings to file

        # get paths of all files that are embedded
        self.embedded_filepaths = self.get_embedded_filepaths()

    def get_all_filepaths(self) -> list:
        """Get paths to all files in image_video folder"""
        return [
            str(f)
            for f in Path(self.image_video_folder).iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpeg", ".jpg", ".mp4"}
        ]

    def get_embedded_filepaths(self) -> set:
        """Get set of paths to all files that are currently embedded."""
        emb_img_filepaths = set(
            self.image_filenames) if self.image_filenames is not None else set()
        emb_vid_filepaths = set([f"{video.split('.mp4_')[0]}.mp4" for video in self.video_timestamps]
                                ) if self.video_timestamps is not None else set()
        return emb_img_filepaths | emb_vid_filepaths

    def concat_image_video_emb(
        self,
        retr_imgs: bool = True,
        retr_vids: bool = True
    ) -> None:
        """Concatenates image and video embeddings into one all embeddings file"""
        # Check what type of embeddings are avilable: img + video OR only image/video OR none
        if self.image_embeddings is not None and self.video_embeddings is not None and retr_imgs and retr_vids:
            # video embeddings AND image embeddings exist and should be retrieved
            self.all_embeddings = np.concatenate(
                [self.image_embeddings, self.video_embeddings], axis=0)
            self.all_filenames_time_stamps = list(
                self.image_filenames) + list(self.video_timestamps)
            logger.debug("Loaded video AND image embeddings")
        elif self.image_embeddings is not None and retr_imgs:
            # image embeddings exist and only images should be retrieved
            self.all_embeddings = self.image_embeddings
            self.all_filenames_time_stamps = self.image_filenames
            logger.debug("Loaded ONLY image embeddings")
        elif self.video_embeddings is not None and retr_vids:
            # video embeddings exist and only videos should be retrieved
            self.all_embeddings = self.video_embeddings
            self.all_filenames_time_stamps = self.video_timestamps
            logger.debug("Loaded ONLY video embeddings")
        else:
            # NO video embeddings AND image embeddings
            self.all_embeddings = None
            self.all_filenames_time_stamps = None

    def save(self) -> None:
        """Save image and video embeddings to file"""
        if self.image_embeddings is not None:
            np.save(self.embedding_file, self.image_embeddings)
            np.save(self.filename_file, self.image_filenames)
            logger.debug("Saved image embeddings")

        if self.video_embeddings is not None:
            np.save(self.video_embedding_file, self.video_embeddings)
            np.save(self.video_timestamp_file, self.video_timestamps)
            logger.debug("Saved video embeddings")

    def get_image_embedding(
        self,
        image_path: str
    ) -> np.ndarray:
        """Get image embedding from CLIP server."""
        response = requests.post(
            f"{self.server_url}/clip/embed_image",
            json={"image_path": image_path},
            timeout=30  # 30 seconds for image processing
        )
        return np.array(response.json()['result'])

    def get_text_features(
        self,
        prompt: str,
    ) -> np.ndarray:
        """Get text embedding from CLIP server."""
        response = requests.post(
            f"{self.server_url}/clip/embed_text",
            json={"text": prompt},
            timeout=30  # 30 seconds for text processing
        )
        return np.array(response.json()['result'])

    def compute_embeddings(
        self,
        filepaths: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes image embeddings for images provided in filepaths"""
        logger.debug("Computing image embeddings...")
        image_embeddings = []
        image_filenames = []
        i = 0

        for filename in filepaths:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = str(Path(self.image_video_folder) / filename)
                try:
                    image_feature = self.get_image_embedding(image_path)
                    image_embeddings.append(image_feature)
                    image_filenames.append(filename)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to process {filename}: {e}")
                i += 1
                if i % 1000 == 0:
                    logger.debug(f"Processed {i} images...")

        # Check if images were found and corresponding embeddings were computed:
        if image_filenames:
            image_embeddings = np.vstack(image_embeddings)
            image_filenames = np.array(image_filenames)
            logger.debug(f"Computed {i} image embeddings")
        else:
            image_embeddings, image_filenames = None, None
            logger.debug(f"No images found")

        return image_embeddings, image_filenames

    def compute_video_embeddings(
        self,
        filepaths: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute embeddings of keyframes from videos provided in filepaths"""
        logger.debug("Computing video embeddings...")
        video_embeddings = []
        video_codes = []

        for filename in filepaths:
            if filename.lower().endswith((".mp4")):
                video_path = str(Path(self.image_video_folder) / filename)
                video_features, video_timestamps = self.clip_video_embedder.get_video_embedding_and_timestamps(
                    video_path)
                for i, feature in enumerate(video_features):
                    video_embeddings.append(feature)
                    video_codes.append(video_timestamps[i])

        # Check if videos were found and corresponding embeddings were computed:
        if video_codes:
            # video embeddings were computed
            video_embeddings = np.vstack(video_embeddings)
            video_codes = np.array(video_codes)
            logger.debug(f"Saved video embeddings")

        else:
            # if no video embeddings were computed
            video_embeddings, video_codes = None, None
            logger.debug(f"No videos found")
        return video_embeddings, video_codes

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Loads saved image embeddings from file"""
        logger.debug("Loading saved embeddings from file...")
        return np.load(self.embedding_file), np.load(self.filename_file)

    def load_video_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Loads saved video embeddings from file"""
        logger.debug("Loading saved video embeddings from file...")
        return np.load(self.video_embedding_file), np.load(self.video_timestamp_file)

    def add_embeddings(
        self,
        new_img_emb: np.ndarray,
        new_img_filenames: np.ndarray,
        new_video_emb: np.ndarray,
        new_video_timestamps: np.ndarray
    ) -> None:
        """Adds new embeddings and saves all embeddings to file"""
        if new_img_emb is not None:  # added files contain at least one image
            if self.image_embeddings is not None:  # image embeddings already exist --> append new embeddings
                self.image_embeddings = np.concatenate(
                    [self.image_embeddings, new_img_emb], axis=0)
                self.image_filenames = np.array(
                    list(self.image_filenames) + list(new_img_filenames))
            else:  # no image embeddings exist yet
                self.image_embeddings = new_img_emb
                self.image_filenames = new_img_filenames

        if new_video_emb is not None:  # added files contain at least one video
            if self.video_embeddings is not None:  # video embeddings alread exist
                self.video_embeddings = np.concatenate(
                    [self.video_embeddings, new_video_emb], axis=0)
                self.video_timestamps = np.array(
                    list(self.video_timestamps) + list(new_video_timestamps))
            else:  # no video embeddings exist yet
                self.video_embeddings = new_video_emb
                self.video_timestamps = new_video_timestamps

        self.save()  # save embeddings to file

    def rm_embeddings(
        self,
        deleted_filepaths: str
    ) -> None:
        """Removes embeddings of deleted files"""
        deleted_img_filepaths = [
            d for d in deleted_filepaths if d.lower().endswith((".jpg", ".jpeg", ".png"))]
        deleted_video_filepaths = [
            d for d in deleted_filepaths if d.lower().endswith((".mp4"))]

        if deleted_filepaths:  # only delete if there are images to delete
            # image indices to be removed
            img_indices = [
                list(self.image_filenames).index(deleted_img)
                for deleted_img in deleted_img_filepaths
            ]
            self.image_embeddings = np.delete(
                self.image_embeddings, img_indices, axis=0)  # delete emb at index
            self.image_filenames = np.delete(
                self.image_filenames, img_indices)  # delete filenames at index

        if deleted_video_filepaths:
            vid_indices = [
                i for i, timestamp in enumerate(list(self.video_timestamps))
                if any(timestamp.startswith(videopath + "_") for videopath in deleted_video_filepaths)
            ]
            # delete all keyframe emb for videos
            self.video_embeddings = np.delete(
                self.video_embeddings, vid_indices, axis=0)
            # delete all timestamps for videos
            self.video_timestamps = np.delete(
                self.video_timestamps, vid_indices)

        self.save()  # save embeddings to file

    def find_top_matches(
        self,
        prompt: str,
        top_k: int,
        retr_imgs: bool,
        retr_vids: bool
    ) -> tuple[list[str], list[float]]:
        """Find top_k matches for a given prompt in the image_video folder."""
        self.top_k = top_k  # set new top_k setting

        start_time = time.time()
        # check if items were added/removed to/from the gallery
        added = set(self.get_all_filepaths()) - self.embedded_filepaths
        removed = self.embedded_filepaths - set(self.get_all_filepaths())
        update_check = time.time() - start_time

        if len(added):  # Files were added --> compute new embeddings and add
            logger.debug("Computing embeddings of newly added files")
            added_image_emb, added_img_filenames = self.compute_embeddings(
                list(added))
            added_video_emb, added_video_timestamps = self.compute_video_embeddings(
                list(added))
            self.add_embeddings(
                added_image_emb, added_img_filenames, added_video_emb, added_video_timestamps)
            self.embedded_filepaths = self.get_embedded_filepaths()

        if len(removed):  # Files were deleted --> rm embeddings
            self.rm_embeddings(list(removed))
            self.embedded_filepaths = self.get_embedded_filepaths()

        # concat embeddings to allow search for images and videos
        self.concat_image_video_emb(retr_imgs, retr_vids)
        concat_time = time.time() - start_time

        # If there are no embeddings and return None to avoid crash
        if self.all_embeddings is None:
            logger.debug("No images to search for")
            return None, None

        prompt_emb = self.get_text_features(prompt)
        prompt_time = time.time() - start_time
        matcher = MeanMatcher(self.all_embeddings, prompt_emb)
        selected_imgs_vid, similarities = matcher.match(
            self.all_filenames_time_stamps, self.top_k)
        logger.debug(
            f"Timing - Update: {update_check:.2f}s, Concat: {concat_time:.2f}s, Prompt: {prompt_time:.2f}s, Total: {time.time()-start_time:.2f}s")

        return selected_imgs_vid, similarities
