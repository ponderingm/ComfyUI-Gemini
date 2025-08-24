import logging
import os
import random
import json

import google.generativeai as genai
# スキーマ定義のためにTypeをインポート
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from torch import Tensor

from .utils import images_to_pillow, temporary_env_var


class GeminiNode:

    @classmethod
    def INPUT_TYPES(cls):  # noqa
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "prompt": ("STRING", {"default": "A girl in a sailor suit walking in Tokyo after the rain with a rainbow.", "multiline": True}),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "model": (
                    [
                        "gemini-1.5-flash-latest",
                        "gemini-1.5-pro-latest",
                    ],
                ),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "system_instruction": ("STRING", {}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 2.0, "step": 0.05}),
                "num_predict": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive_prompt", "negative_prompt",)
    FUNCTION = "ask_gemini"

    CATEGORY = "Gemini"

    def __init__(self):
        self.text_output: str | None = None

    def ask_gemini(self, **kwargs):
        positive_prompt = ""
        negative_prompt = ""

        if self.text_output:
            try:
                data = json.loads(self.text_output)
                positive_prompt = data.get("positive", "")
                negative_prompt = data.get("negative", "")
            except json.JSONDecodeError:
                print(f"ComfyUI-Gemini Error: Failed to decode JSON from response:\n{self.text_output}")
            except Exception as e:
                print(f"ComfyUI-Gemini Error: An unexpected error occurred during JSON parsing: {e}")

        return (positive_prompt, negative_prompt)

    def check_lazy_status(
        self,
        prompt: str,
        safety_settings: str,
        model: str,
        api_key: str | None = None,
        proxy: str | None = None,
        image_1: Tensor | list[Tensor] | None = None,
        image_2: Tensor | list[Tensor] | None = None,
        image_3: Tensor | list[Tensor] | None = None,
        system_instruction: str | None = None,
        error_fallback_value: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        **kwargs,
    ):
        self.text_output = None
        if not system_instruction:
            system_instruction = None
        images_to_send = []
        for image in [image_1, image_2, image_3]:
            if image is not None:
                images_to_send.extend(images_to_pillow(image))
        if "GOOGLE_API_KEY" in os.environ and not api_key:
            genai.configure(transport="rest")
        else:
            genai.configure(api_key=api_key, transport="rest")
        model = genai.GenerativeModel(model, safety_settings=safety_settings, system_instruction=system_instruction)

        # ▼▼▼ 変更点: ここから ▼▼▼
        # 1. 出力してほしいJSONの「スキーマ（構造）」を定義
        prompt_schema = {
            "type": "OBJECT",
            "properties": {
                "positive": {
                    "type": "STRING",
                    "description": "A detailed, high-quality positive prompt for image generation."
                },
                "negative": {
                    "type": "STRING",
                    "description": "A detailed, high-quality negative prompt for image generation."
                }
            },
            "required": ["positive", "negative"]
        }

        # 2. GenerationConfigに response_schema を設定
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=prompt_schema
        )
        # ▲▲▲ 変更点: ここまで ▲▲▲

        if temperature is not None and temperature >= 0:
            generation_config.temperature = temperature
        if num_predict is not None and num_predict > 0:
            generation_config.max_output_tokens = num_predict
        try:
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                response = model.generate_content([prompt, *images_to_send], generation_config=generation_config)
            self.text_output = response.text
        except Exception:
            if error_fallback_value is None:
                logging.getLogger("ComfyUI-Gemini").debug("ComfyUI-Gemini: exception occurred:", exc_info=True)
                return ["error_fallback_value"]
            if error_fallback_value == "":
                raise
        return []


NODE_CLASS_MAPPINGS = {
    "Ask_Gemini_Structured": GeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini_Structured": "My Gemini Auto-Prompter", 
}
