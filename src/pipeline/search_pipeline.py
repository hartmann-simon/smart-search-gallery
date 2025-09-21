from pathlib import Path
import logging
import json

from matching.clip_matcher import CLIPMatcher
from verification.paligemma_client import PaliGemmaVerifier

logger = logging.getLogger(__name__)


class CLIPPaliGemmaPipeline:
    """Pipeline combining CLIP-based retrieval and PaliGemma verification."""

    def __init__(self,
                 config_path: str,
                 gallery_path: str,
                 frames_per_video_clip_max: int = 20,
                 port: int = 5000
                 ) -> None:

        self.gallery = Path(gallery_path)
        self.config_path = Path(config_path)
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.port = port

        # set by get_settings
        self.top_n = None
        self.keyframe_extractor = None
        self.use_vlm = None
        self.retr_imgs = None
        self.retr_vids = None
        self.verification_prompt = None
        self.chunk_size = None

        self.get_settings()

        self.clip_matcher = CLIPMatcher(
            image_video_folder=self.gallery,
            top_k=self.top_n,
            port=self.port,
            frames_per_video_clip_max=self.frames_per_video_clip_max,
        )

        self.verifier = PaliGemmaVerifier(port=self.port)

    def run(
        self,
        prompt: str
    ) -> dict:
        """Run the combined CLIP and PaliGemma pipeline."""
        self.get_settings()

        top_files, top_scores = self.clip_matcher.find_top_matches(
            prompt,
            self.top_n,
            self.retr_imgs,
            self.retr_vids
        )
        logger.debug(f"Top files: {top_files}")
        logger.debug(f"Top scores: {top_scores}")

        gallery = self.clip_matcher.image_video_folder
        top_filepaths = [str(gallery / file) for file in top_files]

        if self.use_vlm:
            prompt = self.verification_prompt.split(
                "<query>")[0] + prompt + self.verification_prompt.split("<query>")[1]
            logger.debug("Verifying matches with PaliGemma")
            verdict = self.verifier.verify_batch(
                top_filepaths, prompt, self.chunk_size)

            confirmed_matches, rejected_matches, unclear_matches = self.verifier.crossref_results(
                verdict, top_files)

            # Add a safety check to prevent errors if a match is not in the original list
            confirmed_scores = [
                top_scores[top_files.index(m)] for m in confirmed_matches if m in top_files]
        else:
            confirmed_matches = top_files
            rejected_matches = []
            unclear_matches = []
            confirmed_scores = top_scores

        return {
            "confirmed": confirmed_matches,
            "rejected": rejected_matches,
            "unclear": unclear_matches,
            "clip_matches": top_files,
            "confirmed_scores": confirmed_scores
        }

    def get_settings(self) -> None:
        """Load settings from the configuration file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            active_config = config.get("active", {})
            self.top_n = active_config.get("NUM_CLIPMATCHES", 30)
            self.use_vlm = bool(active_config.get("VERIFY", True))
            self.retr_imgs = bool(active_config.get("RETR_IMGS", True))
            self.retr_vids = bool(active_config.get("RETR_VIDS", True))
            self.verification_prompt = active_config.get("VERIFIC_PROMPT",
                                                         "Is <query> a fitting description of the image? Answer only with yes or no!"
                                                         )
            self.chunk_size = active_config.get("CHUNK_SIZE", 5)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Couldn't load config: {e}. Using defaults.")
            self.top_n = 30
            self.use_vlm = True
            self.retr_imgs = True
            self.retr_vids = True
            self.verification_prompt = "Is <query> a fitting description of the image? Answer only with yes or no!"
            self.chunk_size = 5
