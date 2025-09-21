from pipeline.search_pipeline import CLIPPaliGemmaPipeline
from dotenv import load_dotenv
from typing import Union
from pathlib import Path
import gradio as gr
import logging
import shutil
import uuid
import json
import os
import re


logger = logging.getLogger(__name__)


class ImageRetrievalApp:
    """
    Class for Gradio UI of the image/video retrieval pipeline.
    This version integrates the pipeline directly, removing the need for file-based communication.
    """

    def __init__(
        self,
        gradio_port: int = 7860
    ) -> None:
        load_dotenv()
        self.gallery_path = Path(os.getenv("GALLERY_PATH"))
        if not self.gallery_path or not self.gallery_path.is_dir():
            raise ValueError(
                "GALLERY_PATH environment variable is not set or is not a valid directory.")

        self.gradio_port = gradio_port
        self._suporrted_formats = ('.png', '.jpeg', '.jpg', '.mp4')
        self._batchsize = 20

        self.search_input = None
        self.search_btn = None
        self.batch_index = None
        self.btn_label = None
        self.selected_item = None
        self.gallery = None

        # Create a temp directory within gallery for uploads
        self.temp_upload_dir = self.gallery_path / ".temp_uploads"
        self.temp_upload_dir.mkdir(exist_ok=True)

        self._allowed_paths = [self.gallery_path]

        # Paths to all items in the gallery
        self._all_paths = sorted([
            item for item in self.gallery_path.iterdir()
            if item.is_file() and item.suffix.lower() in self._suporrted_formats
        ])

        logger.debug("Initializing the search pipeline...")
        server_port = os.getenv("MODEL_SERVER_PORT", "5000")
        self.pipeline = CLIPPaliGemmaPipeline(
            gallery_path=self.gallery_path,
            config_path="src/config/config.json",
            frames_per_video_clip_max=20,
            port=server_port
        )
        logger.debug("Pipeline initialized successfully.")

        self.app = self.build_interface()

    def _search_items(
        self,
        prompt: str,
        current_items: list
    ) -> tuple:
        """Central search function of the UI with a direct pipeline call."""
        # Input validation
        if not bool(prompt) or not prompt.isascii() or len(prompt) > 500 or not bool(re.search(r'[A-Za-z]', prompt)):
            gr.Warning(
                "⚠️ Please use a valid ASCII search query (1-500 chars).")
            initial_batch, new_batch_idx, new_title = self._reset_gallery()
            return initial_batch, new_batch_idx, new_title, {}

        gr.Info("ℹ️ Running search pipeline...")
        results = self.pipeline.run(prompt)
        logger.debug(f"Pipeline results: {results}")

        results_path = [str(self.gallery_path / Path(result))
                        for result in results['confirmed']]
        logger.debug(f"Results paths: {results_path}")
        scores = results['confirmed_scores']
        logger.debug(f"Scores: {scores}")

        # Check if no matches found
        if len(results_path) == 0:
            gr.Warning(
                f"❌ No matches found for '{prompt}'. Try a different search term.")
            initial_batch, new_batch_idx, new_title = self._reset_gallery()
            return initial_batch, new_batch_idx, new_title, {}

        gr.Info(f"✅ Found {len(results_path)} matches")

        scores_dict = self._scores_dict(results_path, scores)
        logger.debug(f"Scores dict: {scores_dict}")
        matches = []
        for match in results_path:
            if ".mp4" in match:
                filepath, _ = self._extract_videocodes(match)
                if filepath not in matches:
                    matches.append(filepath)
            else:
                matches.append(match)

        if any(".mp4" in match for match in matches):
            gr.Info(
                "ℹ️ Some matches are videos. See confidence section for exact timestamps.")

        gallery_title = f"## {len(results_path)} matches for '{prompt}'"
        logger.debug(f"Gallery title: {gallery_title}")
        return matches, 1, gallery_title, scores_dict

    def _load_batch(
        self,
        batch_idx: int,
        current_items: list
    ) -> tuple:
        """ Load the next batch of images/videos into the gallery. """
        start = batch_idx * self._batchsize
        end = start + self._batchsize
        new_batch = self._all_paths[start:end]
        status = f"✅ Loaded {len(new_batch)} more items" if len(
            new_batch) > 0 else "ℹ️ All items are already loaded"
        gr.Info(status)

        # Handle different formats that Gradio 3.5 might pass
        if current_items and isinstance(current_items[0], dict):
            # Extract file paths from Gradio's internal dict format
            current_paths = [item['name'] for item in current_items]
            return current_paths + new_batch, batch_idx + 1
        elif current_items and isinstance(current_items[0], tuple):
            # Convert new_batch to tuples to match the format
            new_batch_tuples = [(str(path), path.name) for path in new_batch]
            return current_items + new_batch_tuples, batch_idx + 1
        else:
            # current_items contains just paths or is empty
            return current_items + new_batch, batch_idx + 1

    def _reset_gallery(self) -> tuple:
        """ Reset the gallery to the initial state showing the first batch. """
        initial_batch = self._all_paths[:self._batchsize]
        gallery_title = "## 🎞️ Camera Roll"
        return initial_batch, 1, gallery_title

    def _on_search_or_cancel(
        self,
        prompt: str,
        current_items: list,
        btn_label: str
    ) -> tuple:
        """ Handle search or cancel button click. """
        if btn_label == "Search":
            results, new_batch_idx, new_title, scores_dict = self._search_items(
                prompt, current_items)
            if results is None:
                # _search_items already computed the reset state
                return results, new_batch_idx, new_title, gr.update(value="Search"), "Search", scores_dict
            results_filenames = [Path(p).name for p in results]
            return zip(results, results_filenames), new_batch_idx, new_title, gr.update(value="Cancel"), "Cancel", scores_dict
        else:
            initial_batch, new_batch_idx, new_title = self._reset_gallery()
            return initial_batch, new_batch_idx, new_title, gr.update(value="Search"), "Search", None

    def _scores_dict(
        self,
        results_path: list,
        scores: list
    ) -> dict:
        """ Create a dictionary mapping filenames (and timestamps for videos) to their scores. """
        scores_dict = {}
        for result, score in zip(results_path, scores):
            if ".mp4" in result:
                filepath, timestamp = self._extract_videocodes(result)
                minutes, seconds = divmod(int(float(timestamp)), 60)
                item = f"{Path(filepath).name} at {minutes:02}:{seconds:02}"
                scores_dict[item] = score
            else:
                scores_dict[Path(result).name] = score
        return scores_dict

    def _extract_videocodes(
        self,
        path: str
    ) -> tuple:
        """ Extract the original video path and timestamp from a video frame path. """
        assert ".mp4" in path, "Path must be a video!"
        filepath = f"{path.split('.mp4_')[0]}.mp4"
        timestamp = path.split(".mp4_")[1]
        return (filepath, timestamp)

    def _on_gallery_select(
        self,
        evt: gr.SelectData
    ) -> str:
        """ Handle gallery item selection to extract the file path. """
        return evt.value

    def _delete_selected_item(
        self,
        selected_item: Union[dict, str],
        current_gallery: list,
        current_batch_idx: int
    ) -> tuple:
        """ Delete the selected item from disk and update the gallery. """
        if not selected_item:
            gr.Warning("⚠️ No valid item selected.")
            return current_gallery, current_batch_idx, "", gr.update(), "", None

        # Handle different types of selection data
        if isinstance(selected_item, dict) and 'name' in selected_item:
            selected_item_path = selected_item['name']
        elif isinstance(selected_item, str):
            selected_item_path = selected_item
        else:
            gr.Warning("⚠️ Unable to determine selected item path.")
            return current_gallery, current_batch_idx, "", gr.update(), "", None

        if selected_item_path and Path(selected_item_path).exists():
            try:
                Path(selected_item_path).unlink()
                # Remove from all_paths
                self._all_paths = [p for p in self._all_paths if str(
                    p) != str(selected_item_path)]

                # Reset to photo roll gallery view to avoid showing deleted image
                initial_batch, new_batch_idx, gallery_title = self._reset_gallery()

                gr.Info(f"✅ Deleted {Path(selected_item_path).name}.")
                # Use gr.update with value to force gallery refresh and exit image view
                return gr.update(value=initial_batch), new_batch_idx, gallery_title, gr.update(value="Search"), "Search", None
            except Exception as e:
                gr.Error(f"❌ Failed to delete: {str(e)}")
            return current_gallery, current_batch_idx, "", gr.update(), "", None
        else:
            gr.Warning("⚠️ Selected item not found on disk.")
            return current_gallery, current_batch_idx, "", gr.update(), "", None

    def _handle_upload(
        self,
        uploaded_files: Union[list, None]
    ) -> tuple:
        """ Handle file uploads and update the gallery. """
        if not uploaded_files:
            gr.Warning("⚠️ No files selected.")
            return self._all_paths[:self._batchsize], 1

        uploaded_paths = []
        for file_obj in uploaded_files:
            # Get the file path from the file object
            if hasattr(file_obj, 'name'):
                file_path = file_obj.name
            else:
                # If it's already a string path, use it directly
                file_path = file_obj

            filename = Path(file_path).name
            name = Path(filename).stem
            fformat = Path(filename).suffix
            if fformat.lower() not in self._suporrted_formats:
                continue

            # Copy to temp directory within allowed paths to avoid Gradio security restrictions
            temp_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            temp_path = self.temp_upload_dir / temp_filename

            try:
                shutil.copy2(file_path, temp_path)
                uploaded_paths.append(temp_path)
            except (OSError, shutil.Error) as e:
                gr.Error(f"❌ Failed to process {filename}: {str(e)}")
                continue

        if uploaded_paths:
            self._all_paths.extend(uploaded_paths)
            self._all_paths = sorted(self._all_paths)
            gr.Info(f"✅ Added {len(uploaded_paths)} item(s) to gallery.")
            return self._all_paths[:self._batchsize], 1
        else:
            gr.Warning("⚠️ No valid items uploaded.")
            return self._all_paths[:self._batchsize], 1

    def _get_stats(self) -> dict:
        """ Get statistics about the number and size of images and videos in the gallery. """
        n_imgs, n_vids, s_imgs, s_vids = 0, 0, 0, 0
        for file in self.gallery_path.iterdir():
            if not file.is_file():
                continue
            size = file.stat().st_size
            if file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                n_imgs += 1
                s_imgs += size
            elif file.suffix.lower() == '.mp4':
                n_vids += 1
                s_vids += size
        return {"imgs": [n_imgs, s_imgs], "vids": [n_vids, s_vids]}

    def _save_config(
        self,
        img: bool,
        vid: bool,
        verify: bool,
        ver_prompt: str,
        chunk_size: int,
        n_clip: int,
    ) -> None:
        """ Save the current configuration to the config file. """
        if not img and not vid:
            gr.Warning(
                "⚠️ You excluded images and videos, no results possible.")
        config_path = "src/config/config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            config.setdefault("active", {})
            config["active"].update({
                "RETR_IMGS": int(img), "RETR_VIDS": int(vid), "VERIFY": int(verify),
                "VERIFIC_PROMPT": ver_prompt, "CHUNK_SIZE": int(chunk_size), "NUM_CLIPMATCHES": n_clip, })
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
            gr.Info("✅ Settings saved successfully")
        except (OSError, json.JSONDecodeError) as e:
            gr.Error(f"❌ Failed to save settings: {str(e)}")

    def _restore_defaults(self) -> list:
        """ Restore default settings from the config file. """
        config_path = "src/config/config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            if "default" not in config:
                gr.Error("❌ No 'default' configuration found.")
                return [True, True, True, "Is <query> a fitting description of the image? Answer only with yes or no!", 5, 30, "keyframe_k_frames"]

            defaults = config["default"]
            if "CHUNK_SIZE" not in defaults:
                defaults["CHUNK_SIZE"] = 5

            config["active"] = defaults.copy()
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
            gr.Info("✅ Defaults restored successfully")

            # Return values to update UI components
            return [
                bool(defaults.get("RETR_IMGS", 1)),
                bool(defaults.get("RETR_VIDS", 1)),
                bool(defaults.get("VERIFY", 1)),
                defaults.get(
                    "VERIFIC_PROMPT", "Is <query> a fitting description of the image? Answer only with yes or no!"),
                defaults.get("CHUNK_SIZE", 5),
                defaults.get("NUM_CLIPMATCHES", 30),
            ]
        except (OSError, json.JSONDecodeError) as e:
            gr.Error(f"❌ Failed to restore defaults: {str(e)}")
            return [True, True, True, "Is <query> a fitting description of the image? Answer only with yes or no!", 5, 30, "keyframe_k_frames"]

    def build_search_tab(self) -> None:
        """ Build the search tab of the Gradio interface. """
        with gr.Tab("Search"):
            gallery_title = gr.Markdown("## 🎞️ Camera Roll")
            with gr.Row():
                self.search_input = gr.Textbox(
                    placeholder="Search for anyting...", show_label=False, scale=4)
                self.search_btn = gr.Button("Search", scale=1, size="lg")

            initial_batch = self._all_paths[:self._batchsize]
            self.batch_index = gr.State(1)
            self.btn_label = gr.State("Search")
            self.selected_item = gr.State()

            self.gallery = gr.Gallery(
                label=" ", value=initial_batch, columns=4,
                height="auto", interactive=False
            )
            with gr.Accordion("See confidence", open=False):
                confidence_scores = gr.Label(
                    show_label=False)

            with gr.Row():
                self.load_more_btn = gr.Button("Load More Images")
                self.delete_btn = gr.Button("Delete Selected Item")

            self.load_more_btn.click(
                fn=self._load_batch,
                inputs=[self.batch_index, self.gallery],
                outputs=[self.gallery, self.batch_index]
            )
            self.gallery.select(
                fn=self._on_gallery_select,
                inputs=None,
                outputs=self.selected_item
            )
            self.delete_btn.click(
                fn=self._delete_selected_item,
                inputs=[self.selected_item, self.gallery, self.batch_index],
                outputs=[self.gallery, self.batch_index, gallery_title,
                         self.search_btn, self.btn_label, confidence_scores]
            )
            self.search_btn.click(
                fn=self._on_search_or_cancel,
                inputs=[self.search_input, self.gallery, self.btn_label],
                outputs=[self.gallery, self.batch_index, gallery_title,
                         self.search_btn, self.btn_label, confidence_scores]
            )

    def build_upload_tab(self) -> None:
        """ Build the upload tab of the Gradio interface. """
        with gr.Tab("Upload"):
            gr.Markdown("## ⬆️ Upload Images and Videos")
            upload_mask = gr.File(
                file_types=["image", ".mp4"], file_count="multiple",
                label="Select Media to Upload", interactive=True
            )
            upload_btn = gr.Button("Upload Selected Files")
            upload_btn.click(
                fn=self._handle_upload,
                inputs=[upload_mask],
                outputs=[self.gallery, self.batch_index]
            )

    def build_settings_tab(self) -> None:
        """ Build the settings tab of the Gradio interface. """
        # Load current config values
        config_path = "src/config/config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            active_config = config.get("active", {})
        except (OSError, json.JSONDecodeError):
            active_config = {}

        with gr.Tab("Settings"):
            gr.Markdown("## ⚙️ Settings")
            gr.Markdown("### General")
            with gr.Row():
                retrieve_videos = gr.Checkbox(
                    value=bool(active_config.get("RETR_VIDS", 1)), label="Retrieve Videos", info="Deactivate to exclude videos from the search results.")
                retrieve_images = gr.Checkbox(
                    value=bool(active_config.get("RETR_IMGS", 1)), label="Retrieve Images", info="Deactivate to exclude images from the search results.")
            gr.Markdown("### VLM")
            vlm_checkbox = gr.Checkbox(value=bool(active_config.get("VERIFY", 1)), label="Verify with VLM",
                                       info="A VLM is used to refine search results. Deactivating improves runtime but reduces quality.")
            verification_prompt = gr.Textbox(
                value=active_config.get(
                    "VERIFIC_PROMPT", "Is <query> a fitting description of the image? Answer only with yes or no!"),
                placeholder="Use <query> as a placeholder for your search query",
                label="VLM Verification Prompt", info="The VLM verifies items using a verification prompt. You can set a custom one here."
            )
            chunk_size_slider = gr.Slider(minimum=1, maximum=30, value=active_config.get("CHUNK_SIZE", 5), step=1, label="VLM Chunk Size",
                                          info="Images processed per batch by the VLM. Use lower values (1-3) for systems with limited RAM/GPU memory, higher values (10-30) for powerful machines. Lower = more reliable but slower.")
            gr.Markdown("### CLIP")
            top_k = gr.Slider(minimum=1, maximum=100, value=active_config.get("NUM_CLIPMATCHES", 30), step=1, label="Number of CLIP Matches",
                              info="CLIP retrieves items to be verified by the VLM. Increasing this negatively affects runtime but might improve results.")

            with gr.Row():
                save_btn = gr.Button("Save Settings")
                restore_btn = gr.Button("Restore Defaults")
            gr.Markdown('### Stats')
            stats = self._get_stats()
            gr.Markdown(
                f"{stats['imgs'][0]} Image(s): {stats['imgs'][1]/1e6:.1f}MB <br>{stats['vids'][0]} Video(s): {stats['vids'][1]/1e6:.1f}MB",
                container=True, line_breaks=True
            )
            save_btn.click(fn=self._save_config, inputs=[
                           retrieve_images, retrieve_videos, vlm_checkbox, verification_prompt, chunk_size_slider, top_k], outputs=[])
            restore_btn.click(fn=self._restore_defaults, inputs=[], outputs=[
                retrieve_images, retrieve_videos, vlm_checkbox, verification_prompt, chunk_size_slider, top_k])

    def build_interface(self) -> gr.Blocks:
        """ Build the complete Gradio interface with all tabs. """
        with gr.Blocks() as interface:
            gr.Markdown('# 🔍 Smart Search Gallery')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        self.build_search_tab()
                        self.build_upload_tab()
                        self.build_settings_tab()
        return interface

    def launch(self) -> None:
        """ Launch the Gradio app. """
        self.app.launch(server_name="127.0.0.1",
                        server_port=self.gradio_port,
                        share=False,
                        allowed_paths=self._allowed_paths)


def main() -> None:
    """Start the app after the model server is running."""
    app = ImageRetrievalApp(gradio_port=7864)
    app.launch()


if __name__ == "__main__":
    main()
