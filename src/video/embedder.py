from pathlib import Path
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np
import subprocess
import requests
import logging
import base64
import decord
import ffmpeg
import torch
import io
import re

logger = logging.getLogger(__name__)


def get_num_frames(
    video_path: str,
    max_frames: int,
) -> int:
    """Determine number of frames to extract based on video duration, capped at max_frames"""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps
    num_frames = max(1, min(max_frames, int(duration / 5))
                     )  # 1 frame every ~5s
    return num_frames


def extract_at_timestamps(
    video_path: str,
    timestamps: List[float]
) -> List[Image.Image]:
    """Helper to extract frames at specific timestamps"""
    frames = []
    for ts in timestamps:
        out, _ = (
            ffmpeg.input(video_path, ss=ts)
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        frames.append(Image.open(io.BytesIO(out)))
    return frames


def embed_frames(
    server_url: str,
    frames: List[Image.Image]
) -> torch.Tensor:
    """Generate CLIP embeddings for a list of frames"""
    with torch.no_grad():
        images = [pil_to_base64(frame) for frame in frames]
        # Send to server
        response = requests.post(
            f'{server_url}/clip/embed_images',
            json={"images": images},
            timeout=60  # 1 minute for batch image embedding
        )
        frame_embeddings = np.array(response.json()['result'])
    return frame_embeddings


def pil_to_base64(
    img,
    format='JPEG'
) -> str:
    """ Convert a PIL Image to a base64-encoded string """
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def detect_scene_changes_direct(
    video_path: str,
    threshold: float = 0.05
) -> List[float]:
    """Detect scene changes with their scores using ffmpeg metadata print filter."""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:v', 'select=gt(scene\\,0),metadata=print',
            '-an', '-f', 'null', '-'
        ]
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT).decode()

        scene_changes = []
        scene_thresholds = []
        lines = output.splitlines()
        for i in range(len(lines) - 1):
            if "pts_time" in lines[i] and "lavfi.scene_score" in lines[i + 1]:
                time_match = re.search(r'pts_time:(\d+(\.\d+)?)', lines[i])
                score_match = re.search(
                    r'scene_score=(\d+(\.\d+)?)', lines[i + 1])
                if time_match and score_match:
                    time = float(time_match.group(1))
                    score = float(score_match.group(1))
                    if score > threshold:
                        scene_changes.append(time)
                        scene_thresholds.append(score)

        return scene_changes, scene_thresholds

    except subprocess.CalledProcessError as e:
        logger.error(f"Scene detection failed: {e.output.decode()}")
        return []


def select_keyframes_hybrid(
    timestamps: List[float],
    scores: List[float],
    duration: float,
    max_k: int = None,
    score_thresh: float = 0.2,
    std_thresh: float = 0.03,
    min_k: int = 1
) -> List[float]:
    """ Select keyframes based on scene change scores and diversity."""
    # Fallback to uniform sampling if no keyframes at all
    if not timestamps or not scores or len(timestamps) == 0 or len(scores) == 0:
        fallback_k = max(min_k, int(duration / 15))  # one every ~15 seconds
        return np.linspace(0, duration, fallback_k + 2)[1:-1].tolist()

    timestamps = np.array(timestamps)
    scores = np.array(scores)

    # Normalize scores
    norm_scores = (scores - scores.min()) / \
        (scores.max() - scores.min() + 1e-8)

    # Compute base number of keyframes depending on duration
    base_k = int(duration / 5)  # 1 keyframe every ~15s
    base_k = max(base_k, min_k)

    # Influence from score quality and number of keyframes
    quality_boost = int(np.clip(norm_scores.mean() * len(timestamps), 0, 5))
    diversity_bonus = int(np.clip(np.std(timestamps) / duration * 10, 0, 5))

    # Final k (adaptive)
    k = base_k + quality_boost + diversity_bonus
    if max_k is not None:
        k = min(k, max_k)
    k = min(k, len(timestamps))  # don't exceed available keyframes
    k = max(k, min_k)

    # If scores are too uniformly bad -> fallback to uniform sampling
    if norm_scores.max() < score_thresh or norm_scores.std() < std_thresh:
        return np.linspace(0, duration, k + 2)[1:-1].tolist()

    # Score-based greedy selection with diversity
    selected = []
    while len(selected) < k:
        best_idx = -1
        best_score = -np.inf

        for i, t in enumerate(timestamps):
            if i in selected:
                continue
            score = norm_scores[i]

            # Diversity penalty
            penalty = 0
            if selected:
                dists = np.abs(t - timestamps[selected])
                min_dist = dists.min()
                penalty = 1 - (min_dist / duration)  # closer = more penalty

            combined = score - 0.6 * penalty  # increase weight for more diversity
            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx == -1:
            break
        selected.append(best_idx)

    return sorted(timestamps[i] for i in selected)


def extract_uniform_frames(
    video_path: str,
    num_frames: int
) -> List[Image.Image]:
    """Extract uniformly spaced frames using decord, based on video duration"""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        # Avoid starting exactly at 0. Space samples at segment centers
        segment_length = total_frames / num_frames
        frame_indices = [int((i + 0.5) * segment_length)
                         for i in range(num_frames)]
        frame_indices = [min(idx, total_frames - 1)
                         for idx in frame_indices]  # clamp

    timestamps = [idx / fps for idx in frame_indices]

    # Middle timestamp (best keyframe)
    main_keyframe_time = timestamps[len(timestamps) // 2]

    return main_keyframe_time, timestamps


def extract_keyframes_ffmpeg(
    video_path: str,
    max_frames: int = 16,
    num_frames: int = None
) -> List[Image.Image]:
    """Intelligently extracts keyframes up to max_frames based on content complexity"""
    try:
        # Get video metadata
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])

        # Detect all scene changes
        scene_changes, scene_thresholds = detect_scene_changes_direct(
            video_path)
        if len(scene_changes) != 0:
            best_idx = int(np.argmax(scene_thresholds))
            main_keyframe_time = scene_changes[best_idx]
        else:
            main_keyframe_time = duration / 2.0

        if num_frames is not None:
            selected_timestamps = select_keyframes_hybrid(
                scene_changes, scene_thresholds, duration, max_k=num_frames, min_k=num_frames)
        else:
            selected_timestamps = select_keyframes_hybrid(
                scene_changes, scene_thresholds, duration, max_k=max_frames)

        return main_keyframe_time, selected_timestamps

    except (ffmpeg.Error, subprocess.CalledProcessError) as e:
        logger.error(f"Keyframe extraction failed: {e}")
        return extract_uniform_frames(video_path, min(3, max_frames))


class CLIPVideoEmbedder:
    """Handles video embedding using CLIP via a server."""

    def __init__(
        self,
        frames_per_video_clip_max: int,
        server_url: str
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = server_url
        self.frames_per_video_clip_max = frames_per_video_clip_max

    def get_video_embedding_and_timestamps(
        self,
        video_path: str
    ) -> tuple[List[np.ndarray], List[str]]:
        """Extract keyframes and get their embeddings for a video."""
        video_name = Path(video_path).name

        num_frames = get_num_frames(video_path, self.frames_per_video_clip_max)
        frames = None

        _, timestamps = extract_keyframes_ffmpeg(
            video_path, max_frames=self.frames_per_video_clip_max, num_frames=num_frames)

        logger.debug(
            f"Selected {len(timestamps)} timestamps for video {video_name}")
        if frames is None:
            frames = extract_at_timestamps(video_path, timestamps)
        embeddings = embed_frames(self.server_url, frames)
        video_key_frame_codes = []
        for i, _ in enumerate(timestamps):
            video_key_frame_codes.append(
                f"{Path(video_name).stem}_{timestamps[i]}"
            )
        return embeddings, video_key_frame_codes
