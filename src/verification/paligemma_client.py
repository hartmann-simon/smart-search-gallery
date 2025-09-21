import requests
import time
import logging

logger = logging.getLogger(__name__)


class PaliGemmaVerifier:
    """Client-side Paligemma class

    args:
        port: int Port the server runs on.
    """

    def __init__(
        self,
        port: int = 5000
    ) -> None:
        self.server_url = f"http://localhost:{port}"

    def verify(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        return self.verify_batch([image_path], prompt)

    def verify_batch(
        self,
        image_paths: list[str],
        prompt: str,
        chunk_size: int = 5
    ) -> list[str]:
        """Verify images or video frames as a batch in one forward pass."""

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        # Check if server is responsive first
        try:
            health_response = requests.get(
                f"{self.server_url}/clip/model_name", timeout=5)
            logger.debug(f"Server health check: {health_response.status_code}")
        except Exception as e:
            logger.error(f"Server not responding: {e}")
            raise ConnectionError(
                f"Model server at {self.server_url} is not responding")

        start_time = time.time()
        logger.debug(
            f"Starting batch verification of {len(image_paths)} images with chunk_size={chunk_size}")

        items = [{"image_path": img, "prompt": prompt} for img in image_paths]

        test_timeout = 900  # 15 minutes to see server logs, for super slow machines
        logger.debug(
            f"Sending request to {self.server_url}/paligemma/verify_batch with {test_timeout}s timeout")

        response = requests.post(
            f"{self.server_url}/paligemma/verify_batch",
            json={"items": items, "chunk_size": chunk_size},
            timeout=test_timeout  # Reduced for debugging
        )
        response.raise_for_status()
        # gets back raw results, usually: <prompt>\n<answer>
        raw_results = response.json()["results"]

        elapsed_time = time.time() - start_time
        logger.debug(f"Batch processing completed in {elapsed_time:.2f}s")
        logger.debug(
            f"Average time per image: {elapsed_time/len(image_paths):.2f}s")

        # convert results to list format: [[<prompt>], [<answer>]]
        results = [r.split('\n') for r in raw_results]
        logger.debug(f"Results from client: {results}")
        # Fetch out answers with wrong structure
        return [res[-1] if isinstance(res, list) else "error" for res in results]

    def crossref_results(
        self,
        verdicts: list[str],
        image_paths: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Method for referencing PaliGemma's output with the verified images and videos

        """
        confirmed = []
        rejected = []
        unclear = []

        # Ensure that verdicts and image paths are lists
        if not isinstance(verdicts, list):
            list(verdicts)
        if not isinstance(image_paths, list):
            list(image_paths)

        # Check for results inputs mismatch
        assert len(verdicts) == len(image_paths)

        for r, image in zip(verdicts, image_paths):
            r = r.lower()
            if "yes" in r and "no" not in r:
                confirmed.append(image)
            elif "no" in r and "yes" not in r:
                rejected.append(image)
            else:
                unclear.append(image)

        return confirmed, rejected, unclear
