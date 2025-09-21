import numpy as np


class MeanMatcher():
    """Matches images to keyframes by averaging similarity scores."""

    def __init__(
        self,
        img_emb: np.ndarray,
        key_emb: np.ndarray
    ) -> None:
        self.img_emb = img_emb
        self.key_emb = key_emb
        self.__similarity = img_emb @ key_emb.T

    def match(
        self,
        paths: list[str],
        k: int
    ) -> tuple[list[str], list[float]]:
        """Matches images to keyframes by averaging similarity scores."""
        means = np.mean(self.__similarity, axis=1)
        idcs = np.argsort(means)[::-1]  # Sort all indices in descending order

        # Create a dictionary to store the best (highest similarity) entry for each video
        seen_videos = set()
        top_paths = []
        top_similarities = []

        for idx in idcs:
            path = paths[idx]

            # Case 1: Video file (contains '.mp4_' in the path)
            if '.mp4_' in path:
                video_name = path.split('_')[0]  # Extract "video1234.mp4"
                if video_name in seen_videos:
                    continue  # Skip duplicate videos
                seen_videos.add(video_name)

            top_paths.append(path)
            top_similarities.append(means[idx])

            # Early exit if we've collected enough
            if len(top_paths) == k:
                break

        return top_paths, top_similarities
