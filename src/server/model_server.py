from transformers import AutoProcessor, AutoModelForVision2Seq
from config.logging_config import setup_logging
from flask import Flask, request, jsonify
from concurrent.futures import Future
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import threading
import traceback
import logging
import base64
import queue
import torch
import clip
import time
import cv2
import os


load_dotenv()
# Setup logging for the all files in the application
# setup_logging(level="ERROR")
setup_logging(level="DEBUG")
logger = logging.getLogger(__name__)


class ModelServer:
    """
    Flask server for handling model inference requests.
    """

    def __init__(
        self,
        port: int = 5000,
        hf_token: str = None
    ) -> None:
        self.port = port
        self.hf_token = hf_token
        self.app = Flask(__name__)

        # Model variables
        self.clip_model_name = "ViT-L/14@336px"
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_device = None
        self.paligemma_model_name = "google/paligemma-3b-mix-224"
        self.paligemma_model = None
        self.paligemma_processor = None
        self.paligemma_device = None

        # Request queues
        self.clip_queue = queue.Queue(maxsize=100)
        self.paligemma_queue = queue.Queue(maxsize=100)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        self.app.route('/clip/embed_image',
                       methods=['POST'])(self.embed_image)
        self.app.route('/clip/embed_images',
                       methods=['POST'])(self.embed_images)
        self.app.route('/clip/embed_text',
                       methods=['POST'])(self.embed_text)
        self.app.route('/paligemma/verify_batch',
                       methods=['POST'])(self.paligemma_verify_batch)
        self.app.route('/clip/model_name',
                       methods=['GET'])(self.get_clip_model_name)
        self.app.route('/paligemma/model_name',
                       methods=['GET'])(self.get_paligemma_model_name)

    def initialize_models(self) -> None:
        """Initialize CLIP and PaliGemma models"""
        # Initialize CLIP
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model: {self.clip_model_name} ...")
        self.clip_model, self.clip_preprocess = clip.load(
            self.clip_model_name, device=self.clip_device)

        # Initialize PaliGemma
        self.paligemma_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading PaliGemma model: {self.paligemma_model_name} ...")
        self.paligemma_processor = AutoProcessor.from_pretrained(
            self.paligemma_model_name, token=self.hf_token)
        self.paligemma_model = AutoModelForVision2Seq.from_pretrained(
            self.paligemma_model_name,
            torch_dtype=torch.bfloat16,
            token=self.hf_token
        ).to(self.paligemma_device)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except RuntimeError:
                pass

    def clip_worker(self) -> None:
        """Worker thread for CLIP processing"""
        while True:
            task = self.clip_queue.get()
            if task is None:  # Sentinel value to stop the thread
                break
            try:
                image_path, prompt = task['image_path'], task['prompt']
                image = self.clip_preprocess(Image.open(
                    image_path)).unsqueeze(0).to(self.clip_device)
                with torch.no_grad():
                    image_feature = self.clip_model.encode_image(image).float()
                    text_tokens = clip.tokenize([prompt]).to(self.clip_device)
                    text_features = self.clip_model.encode_text(
                        text_tokens).float()
                    similarity = (image_feature @ text_features.T).item()
                task['future'].set_result(similarity)
            except (IOError, RuntimeError) as e:
                task['future'].set_exception(e)
            finally:
                self.clip_queue.task_done()

    def paligemma_worker(self) -> None:
        """Worker thread for PaliGemma processing"""
        try:
            logger.debug("PaliGemma worker thread started and running...")
            while True:
                logger.debug("PaliGemma worker: Waiting for task...")
                task = self.paligemma_queue.get()
                logger.debug(f"PaliGemma worker: Got task: {task is not None}")
                if task is None:
                    logger.debug(
                        "PaliGemma worker: Received stop signal, exiting...")
                    break
                try:
                    start_time = time.time()
                    logger.debug(
                        "PaliGemma worker: Starting batch processing...")
                    logger.debug(
                        f"PaliGemma worker: About to call _verify_batch with {len(task['batch'])} items")
                    chunk_size = task.get("chunk_size", 5)
                    result = self._verify_batch(task["batch"], chunk_size)
                    # Set result IMMEDIATELY after processing completes
                    task["future"].set_result(result)
                    end_time = time.time()
                    logger.debug(
                        f"PaliGemma worker: Batch processing complete in {end_time - start_time:.2f}s, result: {result}")
                    logger.debug("PaliGemma worker: Result set successfully")
                except (IOError, RuntimeError, ValueError) as e:
                    logger.error(f"PaliGemma worker: Exception occurred: {e}")
                    traceback.print_exc()
                    task['future'].set_exception(e)
                finally:
                    logger.debug(
                        "PaliGemma worker: Task done, marking queue task as done")
                    self.paligemma_queue.task_done()
        except (IOError, RuntimeError, ValueError) as e:
            logger.error(f"PaliGemma worker thread crashed: {e}")
            traceback.logger.debug_exc()

    def _verify_batch(
        self,
        batch: list[dict],
        chunk_size: int = 5
    ) -> list[str]:
        """Verify batch of images/video frames with PaliGemma"""
        batch_start_time = time.time()
        logger.debug("_verify_batch: Starting...")
        logger.debug(
            f"_verify_batch: Processing {len(batch)} items with chunk_size: {chunk_size}")

        # Use the provided chunk_size parameter
        logger.debug(f"_verify_batch: Using chunk size: {chunk_size}")
        all_results = []

        for chunk_start in range(0, len(batch), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batch))
            chunk = batch[chunk_start:chunk_end]
            chunk_num = chunk_start//chunk_size + 1
            total_chunks = (len(batch) + chunk_size - 1)//chunk_size
            logger.debug(
                f"_verify_batch: Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} items)")

            # Clear GPU cache before processing each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(
                    f"_verify_batch: Cleared GPU cache for chunk {chunk_num}")

            logger.debug(
                f"_verify_batch: Forced garbage collection for chunk {chunk_num}")

            logger.debug(
                f"_verify_batch: Recovery delay completed for chunk {chunk_num}")

            prompts = [item['prompt'] for item in chunk]
            logger.debug(f"First prompt in chunk: {prompts[0]}")

            logger.debug("_verify_batch: Loading images for chunk...")
            images_keyframes = []
            for i, item in enumerate(chunk):
                logger.debug(
                    f"_verify_batch: Processing item {i+1}/{len(chunk)} in chunk")
                img_keyframe_path = item["image_path"]

                if img_keyframe_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logger.debug(
                        f"_verify_batch: Loading image: {img_keyframe_path}")
                    images_keyframes.append(Image.open(
                        img_keyframe_path).convert("RGB"))
                    logger.debug(
                        f"_verify_batch: Successfully loaded image {i+1}")

                elif ".mp4" in img_keyframe_path.lower():
                    try:
                        logger.debug(
                            f"_verify_batch: Processing video: {img_keyframe_path}")
                        # keyframe items have the structure: <path>/<video_name>.mp4_timestamp
                        video_path, timestamp_str = img_keyframe_path.rsplit(
                            ".mp4_", 1)
                        timestamp = float(timestamp_str)
                        video_path = f"{video_path}.mp4"
                        images_keyframes.append(
                            self.extract_frame_at_timestamp(video_path, timestamp))
                        logger.debug(
                            f"_verify_batch: Successfully processed video {i+1}")
                    except (IOError, RuntimeError, ValueError) as e:
                        logger.error(
                            f"Failed to process video {img_keyframe_path}: {e}")

            logger.debug(
                f"_verify_batch: Loaded {len(images_keyframes)} images/frames for chunk")
            logger.debug(
                "_verify_batch: Preparing inputs for PaliGemma processor...")

            try:
                input_start = time.time()
                inputs = self.paligemma_processor(
                    images=images_keyframes,
                    text=prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.paligemma_device)
                input_time = time.time() - input_start
                logger.debug(
                    f"_verify_batch: Inputs prepared in {input_time:.2f}s, starting model generation for chunk {chunk_num}...")

                gen_start = time.time()
                with torch.no_grad():
                    outputs = self.paligemma_model.generate(
                        **inputs, max_new_tokens=5)
                gen_time = time.time() - gen_start
                logger.debug(
                    f"_verify_batch: Model generation complete in {gen_time:.2f}s, decoding results for chunk {chunk_num}...")

                decode_start = time.time()
                results = self.paligemma_processor.batch_decode(
                    outputs, skip_special_tokens=True)
                decode_time = time.time() - decode_start
                logger.debug(
                    f"PaliGemma chunk {chunk_num} results in {decode_time:.2f}s: {results}")
                logger.debug(f"PaliGemma chunk {chunk_num} results: {results}")

                chunk_results = [res.strip() for res in results]
                all_results.extend(chunk_results)

                logger.debug(f"_verify_batch: Chunk {chunk_num} complete!")

                # Clean up memory after each chunk
                del inputs, outputs, results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(
                        f"_verify_batch: Memory cleanup completed for chunk {chunk_num}")
                logger.debug(
                    f"_verify_batch: Additional garbage collection completed for chunk {chunk_num}")
                logger.debug(
                    f"_verify_batch: Post-processing delay completed for chunk {chunk_num}")

            except (IOError, RuntimeError, ValueError) as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                import traceback
                traceback.print_exc()
                chunk_results = ["error" for _ in chunk]
                all_results.extend(chunk_results)

        batch_total_time = time.time() - batch_start_time
        logger.debug(
            f"_verify_batch: All chunks processed in {batch_total_time:.2f}s! Total results: {len(all_results)}")
        logger.debug("_verify_batch: Batch processing complete!")
        return all_results

    def embed_image(self) -> None:
        """Embed single image with CLIP"""
        data = request.json
        try:
            image_path = data['image_path']
            image = self.clip_preprocess(Image.open(
                image_path)).unsqueeze(0).to(self.clip_device)
            with torch.no_grad():
                image_feature = self.clip_model.encode_image(image).float()
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
            return jsonify({'result': image_feature.cpu().numpy().tolist()})
        except (IOError, RuntimeError, ValueError) as e:
            return jsonify({"error": str(e)}), 500

    def embed_images(self) -> None:
        """Embed batch of images with CLIP"""
        data = request.json
        try:
            image_tensors = []

            if "image_paths" in data:
                for path in data["image_paths"]:
                    image = Image.open(path)
                    image_tensor = self.clip_preprocess(image).unsqueeze(0)
                    image_tensors.append(image_tensor)

            elif "images" in data:
                for image_data in data["images"]:
                    image = Image.open(
                        BytesIO(base64.b64decode(image_data))).convert("RGB")
                    image_tensor = self.clip_preprocess(image).unsqueeze(0)
                    image_tensors.append(image_tensor)
            else:
                raise ValueError(
                    "No valid image input found. Provide 'image_paths' or 'images'.")

            batch = torch.cat(image_tensors, dim=0).to(self.clip_device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(batch).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

            return jsonify({'result': image_features.cpu().numpy().tolist()})
        except (IOError, RuntimeError, ValueError) as e:
            return jsonify({"error": str(e)}), 500

    def embed_text(self) -> None:
        """Embed text with CLIP"""
        data = request.json
        try:
            text = data['text']
            text_tokens = clip.tokenize([text]).to(self.clip_device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(
                    text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return jsonify({'result': text_features.cpu().numpy().tolist()})
        except (IOError, RuntimeError, ValueError) as e:
            return jsonify({"error": str(e)}), 500

    def paligemma_verify_batch(self) -> None:
        """Verify batch with PaliGemma"""
        items = request.json.get("items", [])
        # Get chunk_size from request
        chunk_size = request.json.get("chunk_size", 5)
        if not isinstance(items, list):
            return jsonify({"error": "items must be a list"}), 400

        logger.debug(
            f"Received batch with {len(items)} items, chunk_size: {chunk_size}")
        future = Future()
        self.paligemma_queue.put(
            {"batch": items, "future": future, "chunk_size": chunk_size})

        try:
            logger.debug("Waiting for PaliGemma worker to complete...")
            # Increased timeout to 15 minutes for batch processing
            result = future.result(timeout=900)
            logger.debug(
                f"PaliGemma worker completed successfully with result: {result}")
            return jsonify({"results": result})
        except (IOError, RuntimeError, ValueError) as e:
            logger.error(f"Error in paligemma_verify_batch: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def extract_frame_at_timestamp(
        self,
        video_path: str,
        timestamp: float
    ) -> Image.Image:
        """Extract frame from video at specific timestamp"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            raise RuntimeError(f"Could not read frame at {timestamp:.2f}s")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_clip_model_name(self) -> None:
        """Get CLIP model name"""
        return jsonify({'clip_model_name': self.clip_model_name})

    def get_paligemma_model_name(self) -> None:
        """Get PaliGemma model name"""
        return jsonify({'paligemma_model_name': self.paligemma_model_name})

    def start_workers(self) -> None:
        """Start worker threads"""
        logger.debug("Starting worker threads...")
        try:
            for i in range(2):
                clip_thread = threading.Thread(
                    target=self.clip_worker, daemon=True)
                clip_thread.start()
                logger.debug(f"Started CLIP worker thread {i+1}")

            for i in range(1):
                paligemma_thread = threading.Thread(
                    target=self.paligemma_worker, daemon=True)
                paligemma_thread.start()
                logger.debug(f"Started PaliGemma worker thread {i+1}")

            logger.debug("All worker threads started successfully")

        except (RuntimeError, IOError, ValueError) as e:
            logger.error(f"Error starting worker threads: {e}")
            import traceback
            traceback.print_exc()

    def run(self) -> None:
        """Start the Flask server"""
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.initialize_models()
        self.start_workers()
        logger.debug(
            f"Successfully started model server on localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port)


def main() -> None:
    """Main entry point for programmatic startup"""
    port = int(os.getenv('MODEL_SERVER_PORT', '5000'))
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. See README.md for help.")

    server = ModelServer(port=port, hf_token=hf_token)
    server.run()


if __name__ == '__main__':
    main()
