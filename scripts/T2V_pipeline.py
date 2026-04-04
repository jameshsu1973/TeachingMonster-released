"""
VideoGenerationPipeline Module

This module implements a monolithic orchestration layer for generating educational
or presentation-style videos from a simple text prompt. It follows a strictly 
sequential "waterfall" architecture:

Data Flow:
1. User Prompt -> [OutlineModule] -> Structured JSON Outline
2. Outline -> [Wrapper] -> Slide Content & Voiceover Scripts
3. Slide Content -> [SlidesModule] -> Static Images (.jpg)
4. Scripts -> [TTSModule] -> Audio files (.mp3) + Word-level Timestamps
5. Images/Scripts/Timestamps -> [CursorModule] -> X/Y Coordinate Trajectories
6. All Assets -> [MoviePy Renderer] -> Final MP4

Key Components:
- LLM Client: Handles all generative text tasks.
- AppConfig: Manages paths, slide types (PPT vs 3B1B), and model parameters.
- MoviePy: Used for the final compositing of layers (Slide + Cursor + Audio).
"""

import argparse
import os
from typing import List

import yaml
import numpy as np
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
import json

from src import *

class VideoGenerationPipeline:
    """
    End-to-end pipeline with built-in renderer.

    Attributes:
        app_config (AppConfig): Configuration object for directories and module settings.
        llm_client: The backend LLM provider (e.g., Gemini, OpenAI).
        video_output_path (str): Final destination of the rendered MP4.
        fps (int): Frames per second for the output video (default: 15).
    """

    def __init__(
        self,
        llm_client,
        app_config: AppConfig,
        output_video_name: str = "final_video.mp4",
        final_video_dir: str = "./output",
    ):
        self.app_config = app_config
        self.llm_client = llm_client

        # --- Directory Management ---
        # Prioritizes explicit arguments over config file defaults
        output_config = app_config.output
        tmp_dir = (output_config.tmp_dir if output_config else None) or "./tmp_pipeline"
        os.makedirs(tmp_dir, exist_ok=True)

        final_dir = (
            final_video_dir
            or (output_config.final_video_dir if output_config else None)
            or tmp_dir
        )

        # Define sub-directories for intermediate assets
        self.slides_output_root = os.path.join(tmp_dir, "slides")
        self.tts_output_root = os.path.join(tmp_dir, "tts")
        os.makedirs(final_dir, exist_ok=True)
        self.final_dir = final_dir

        self.video_output_path = os.path.join(final_dir, output_video_name)

        self.fps = 15

        # --- Module Initialization ---
        self.outline_module = T2VOutlineModule(llm_client)

        # Support for different slide engines (Standard PPT vs Manim-style 3B1B)
        if app_config.pipeline.slides_type == "PPT":
            self.slides_module = SlidesModule_PPT(
                llm_client, config=self.app_config.ppt, output_root=self.slides_output_root
            )
            self.wrapper = Wrapper_PPT(llm_client)
        else:
            self.slides_module = SlidesModule_3B1B(
                llm_client, output_root=self.slides_output_root
            )
            self.wrapper = Wrapper_3B1B(llm_client)
            
        self.tts_module = TTSModule(
            output_root=self.tts_output_root,
            tts_instruct="",  # 明確設為空字串，抑制非預期音效（如笑聲）
            slide_head_silence_ms=600.0,  # 投影片開頭停頓 0.6 秒
            slide_tail_silence_ms=1500.0,  # 投影片結尾停頓 1.5 秒，讓間隔更明確
            # ── 速度優化參數（與 tts.py 對齊）──
            use_torch_compile=True,      # torch.compile 加速首生成（預設開啟）
            # parallel_slides=4,           # 投影片 TTS + ASR 並行數
            # parallel_asr=True,           # ASR 與下一張 TTS 重疊執行
            # asr_beam_size=5,             # 調高 beam_size 加速 ASR
        )

    def load(self, skip_steps_123: bool = False):
        """
        Initializes heavy resources (e.g., local ML models, network connections).
        Must be called after __init__ and before run().

        Args:
            skip_steps_123: If True, skip loading outline/wrapper/slides modules.
        """
        if not skip_steps_123:
            self.outline_module.load()
            self.wrapper.load()
            self.slides_module.load()
        self.tts_module.load()

    def run(
        self,
        requirement_prompt: str,
        persona_prompt: str,
        use_existing_slides: bool = False,
        existing_slides_dir: str = "./tmp_template",
        skip_steps_123: bool = False,
        cache_dir: str = "./tmp",
    ) -> dict:
        """
        Orchestrates the full generation process from text to MP4.

        Args:
            requirement_prompt: The core topic or instruction.
            persona_prompt: The tone/style of the presentation (e.g., "Academic").
            use_existing_slides: If True, skip PPT generation and use existing JPGs.
            existing_slides_dir: Directory containing existing slide images.
            skip_steps_123: If True, load outlines/scripts/slides_struct from cache_dir.
            cache_dir: Directory containing cached JSON files (default: ./tmp).

        Returns:
            dict: A dictionary containing paths and data for all generated assets.
        """

        # =============================
        # Step 1: Content Planning
        # =============================
        outlines_path = os.path.join(cache_dir, "outlines.json")
        if skip_steps_123 and os.path.exists(outlines_path):
            print(f"[INFO] Loading cached outlines from: {outlines_path}")
            with open(outlines_path, encoding="utf-8") as f:
                outlines = json.load(f)
        else:
            print("[INFO] Generating outlines via API...")
            outlines: List[str] = self.outline_module.run(
                requirement_prompt=requirement_prompt,
                persona_prompt=persona_prompt,
            )
            os.makedirs(cache_dir, exist_ok=True)
            json.dump(outlines, open(outlines_path, "w+", encoding="utf-8"), ensure_ascii=False, indent=4)

        # =============================
        # Step 2: Content Interpretation
        # =============================
        slides_struct_path = os.path.join(cache_dir, "slides_struct.json")
        scripts_path = os.path.join(cache_dir, "scripts.json")
        if skip_steps_123 and os.path.exists(slides_struct_path) and os.path.exists(scripts_path):
            print(f"[INFO] Loading cached slides_struct from: {slides_struct_path}")
            print(f"[INFO] Loading cached scripts from: {scripts_path}")
            with open(slides_struct_path, encoding="utf-8") as f:
                slides_struct = json.load(f)
            with open(scripts_path, encoding="utf-8") as f:
                scripts = json.load(f)
        else:
            print("[INFO] Generating slides specs and scripts via API...")
            slides_struct, scripts = self.wrapper.run(outlines)
            os.makedirs(cache_dir, exist_ok=True)
            json.dump(slides_struct, open(slides_struct_path, "w+", encoding="utf-8"), ensure_ascii=False, indent=4)
            json.dump(scripts, open(scripts_path, "w+", encoding="utf-8"), ensure_ascii=False, indent=4)

        # =============================
        # Step 3: Visual Asset Generation
        # =============================
        if use_existing_slides or skip_steps_123:
            # When skip_steps_123 is True, always use existing slides
            slides_dir = existing_slides_dir if use_existing_slides else existing_slides_dir
            if not os.path.isdir(slides_dir):
                print(f"[ERROR] Slides directory '{slides_dir}' does not exist.")
                return None
            print(f"[INFO] Using existing slides from: {slides_dir}")
            slides_folder = slides_dir
            
            expected_count = len(slides_struct)
            existing_files = [f for f in os.listdir(slides_dir) if f.lower().endswith(('.jpg', '.png'))]
            print(f"[INFO] Found {len(existing_files)} image files in {slides_dir}")
            if len(existing_files) < expected_count:
                print(f"[WARNING] Expected {expected_count} slides but found {len(existing_files)}")
        else:
            print("[INFO] Generating slides via API...")
            slides_folder = self.slides_module.run(slides_struct)

        slide_images: List[Image.Image] = []
        slide_image_paths: List[str] = []
        for idx in range(1, len(slides_struct) + 1):
            img_path = os.path.join(slides_folder, f"{idx}.jpg")
            if not os.path.exists(img_path):
                png_path = os.path.join(slides_folder, f"{idx}.png")
                if os.path.exists(png_path):
                    img_path = png_path
            slide_image_paths.append(img_path)
            slide_images.append(Image.open(img_path))

        # =============================
        # Step 4: Audio Synthesis
        # =============================
        audio_folder, word_timings, *_extra = self.tts_module.run(scripts)
        json.dump(word_timings, open(os.path.join(cache_dir, "word_timings.json"), "w+", encoding="utf-8"), ensure_ascii=False, indent=4)

        audio_paths: List[str] = [
            os.path.join(audio_folder, f"{i}.mp3") for i in range(1, len(scripts) + 1)
        ]

        # =============================
        # Step 5: Animation Planning (DISABLED)
        # =============================

        # =============================
        # Step 6: Built-in Video Rendering (MoviePy)
        # =============================
        video_clips = []

        for slide_idx in range(len(slide_image_paths)):
            img_path = slide_image_paths[slide_idx]
            audio_path = audio_paths[slide_idx]
            
            audio_clip = AudioFileClip(audio_path).set_fps(44100)
            slide_duration = audio_clip.duration

            base_clip = ImageClip(img_path).set_duration(slide_duration)

            # --- USE ONLY BASE SLIDE + AUDIO (NO CURSOR) ---
            video_clips.append(base_clip.set_audio(audio_clip))

        # Merge all individual slide clips and encode to MP4
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(
            self.video_output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
        )

        # Cleanup memory/file handles
        final_video.close()
        for clip in video_clips:
            clip.close()
            if hasattr(clip, "audio") and clip.audio:
                clip.audio.close()

        # =============================
        # Step 7: Return bundle
        # =============================
        return {
            "outlines": outlines,
            "slides_struct": slides_struct,
            "slides_folder": slides_folder,
            "scripts": scripts,
            "audio_folder": audio_folder,
            "word_timings": word_timings,
            "final_video_path": self.video_output_path,
        }


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pipeline with prompts and video path"
    )

    parser.add_argument(
        "-r", "--requirement-prompt",
        type=str, default="Explain machine learning basics",
        help="Main requirement prompt",
    )
    parser.add_argument(
        "-p", "--persona-prompt",
        type=str, default="Friendly instructor",
        help="Persona prompt",
    )
    parser.add_argument(
        "-c", "--config",
        type=str, default="config/default.yaml",
        help="Generation config",
    )
    parser.add_argument(
        "-o", "--output-video-name",
        type=str, default="final_video.mp4",
        help="Output video file name",
    )
    parser.add_argument(
        "-d", "--final-video-dir",
        type=str, default=None,
        help="Directory for final video output",
    )
    parser.add_argument(
        "--use-existing-slides",
        action="store_true",
        help="Use existing slide images instead of generating new ones",
    )
    parser.add_argument(
        "--existing-slides-dir",
        type=str, default="./tmp_template",
        help="Directory containing existing slide images (default: ./tmp_template)",
    )
    parser.add_argument(
        "--skip-steps-123",
        action="store_true",
        help="Skip Steps 1-3 (outline, wrapper, slides) and use cached files from tmp/",
    )

    args = parser.parse_args()

    print("\n=== Input ===")
    print(f"Requirement prompt: {args.requirement_prompt}")
    print(f"Persona prompt: {args.persona_prompt}")
    print(f"Config path: {args.config}")
    print(f"Use existing slides: {args.use_existing_slides}")
    print(f"Skip steps 1-3: {args.skip_steps_123}")

    print("\n=== Loading ===")
    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        config = AppConfig(**data)

    client = GeminiClient(config.llm)

    pipeline = VideoGenerationPipeline(
        llm_client=client,
        app_config=config,
        output_video_name=args.output_video_name,
        final_video_dir=args.final_video_dir,
    )

    pipeline.load(skip_steps_123=args.skip_steps_123)

    print("\n=== Run ===")
    assets = pipeline.run(
        requirement_prompt=args.requirement_prompt,
        persona_prompt=args.persona_prompt,
        use_existing_slides=args.use_existing_slides,
        existing_slides_dir=args.existing_slides_dir,
        skip_steps_123=args.skip_steps_123,
    )

    if assets:
        print("\n=== Final Output ===")
        print("Final video:", assets["final_video_path"])
    else:
        print("\n[ERROR] Pipeline failed.")
