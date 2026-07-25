#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchaudio
from safetensors.torch import save as safetensors_save
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoConfig, HiggsAudioV2TokenizerModel

import omnivoice.models.omnivoice as omnivoice_module
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.text import chunk_text_punctuation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_REF_AUDIO = ROOT / "ref.wav"
DEFAULT_REF_TEXT = ROOT / "ref_text.txt"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "python_reference"
CASES_PATH = Path(__file__).with_name("cases.json")
MODEL_CACHE: dict[tuple[str, str, str], OmniVoice] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_case_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    case_ids = [item.strip() for item in raw.split(",") if item.strip()]
    return case_ids or None


def parse_capture_steps(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def parse_capture_layers(raw: str | None) -> list[int | str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return None
    layers: list[int | str] = []
    for value in values:
        if value == "final":
            layers.append("final")
        else:
            layers.append(int(value))
    return layers


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bundle_hash(tensors: dict[str, torch.Tensor]) -> str:
    serializable = {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(tensors.items())}
    payload = safetensors_save(serializable)
    return hashlib.sha256(payload).hexdigest()


def save_tensor_bundle(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    ensure_parent(path)
    cpu_tensors = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in tensors.items()
    }
    safetensors_save_file(cpu_tensors, str(path))


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    ensure_parent(path)
    torchaudio.save(str(path), audio.detach().cpu(), sample_rate)


def resolve_cases(
    ref_audio: Path,
    ref_text: str,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case_map = {case["id"]: case for case in data["cases"]}
    if case_ids is None:
        selected_ids = [
            case["id"]
            for case in data["cases"]
            if case.get("enabled_by_default", True)
        ]
    else:
        missing = [case_id for case_id in case_ids if case_id not in case_map]
        if missing:
            raise ValueError(f"unknown case ids: {missing}")
        selected_ids = case_ids

    cases = []
    for case_id in selected_ids:
        case = case_map[case_id]
        resolved = json.loads(json.dumps(case))
        if resolved.get("ref_audio") == "__CLI_REF_AUDIO__":
            resolved["ref_audio"] = str(ref_audio)
        if resolved.get("ref_text") == "__CLI_REF_TEXT__":
            resolved["ref_text"] = ref_text
        cases.append(resolved)
    return cases


def apply_debug_overrides(
    cases: list[dict[str, Any]],
    debug_case_ids: list[str] | None,
    debug_device: str | None,
    debug_dtype: str | None,
    capture_steps: list[int] | None,
    capture_layers: list[int | str] | None,
) -> None:
    if not debug_case_ids:
        return

    debug_case_id_set = set(debug_case_ids)
    for case in cases:
        if case["id"] not in debug_case_id_set:
            continue

        debug_cfg = case.setdefault("debug", {})
        debug_cfg["device"] = debug_device or debug_cfg.get("device", "cpu")
        debug_cfg["dtype"] = debug_dtype or debug_cfg.get("dtype", "float32")
        debug_cfg["capture_steps"] = capture_steps or debug_cfg.get(
            "capture_steps", [0, 15, 31]
        )
        debug_cfg["capture_layers"] = capture_layers or debug_cfg.get(
            "capture_layers", [0, 13, 27, "final"]
        )


def gather_runtime_info(model_dir: Path, device: str, dtype: str, seed: int) -> dict[str, Any]:
    package_names = [
        "omnivoice",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "numpy",
        "soundfile",
        "pydub",
        "safetensors",
    ]
    package_versions = {}
    for name in package_names:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "device_count": torch.cuda.device_count(),
        }
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gpu["nvidia_smi"] = result.stdout.strip().splitlines()
        except Exception as exc:
            gpu["nvidia_smi_error"] = str(exc)

    return {
        "created_at": utc_now(),
        "root": str(ROOT),
        "model_dir": str(model_dir),
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "gpu": gpu,
        "package_versions": package_versions,
        "model_hashes": {
            "model/model.safetensors": file_sha256(model_dir / "model.safetensors"),
            "model/audio_tokenizer/model.safetensors": file_sha256(model_dir / "audio_tokenizer" / "model.safetensors"),
        },
        "config_snapshots": {
            "model": json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
            "audio_tokenizer": json.loads((model_dir / "audio_tokenizer" / "config.json").read_text(encoding="utf-8")),
            "audio_preprocessor": json.loads((model_dir / "audio_tokenizer" / "preprocessor_config.json").read_text(encoding="utf-8")),
        },
    }


def verify_environment(model_dir: Path) -> dict[str, Any]:
    audio_cfg = AutoConfig.from_pretrained(str(model_dir / "audio_tokenizer"))
    return {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "higgs_audio_config_type": type(audio_cfg).__name__,
        "higgs_audio_model_type": audio_cfg.model_type,
        "higgs_audio_symbol": HiggsAudioV2TokenizerModel.__name__,
    }


def build_model(model_dir: Path, device: str, dtype: str) -> OmniVoice:
    cache_key = (str(model_dir), device, dtype)
    if cache_key not in MODEL_CACHE:
        torch_dtype = parse_dtype(dtype)
        model = OmniVoice.from_pretrained(str(model_dir), device_map=device, dtype=torch_dtype)
        model.eval()
        MODEL_CACHE[cache_key] = model
    return MODEL_CACHE[cache_key]


def patch_instrumented_generate_iterative(
    model: OmniVoice,
    capture_steps: set[int] | None = None,
    capture_layers: list[int | str] | None = None,
    capture_root: Path | None = None,
) -> None:
    capture_steps = capture_steps or set()
    capture_layers = capture_layers or [0, 13, 27, "final"]
    model._phase0_debug_invariants = None
    model._phase0_debug_payload = None
    model._phase0_debug_forward = None

    def instrumented_generate_iterative(self: OmniVoice, task, gen_config):
        inputs_list = [
            self._prepare_inference_inputs(
                task.texts[i],
                task.target_lens[i],
                task.ref_texts[i],
                task.ref_audio_tokens[i],
                task.langs[i],
                task.instructs[i],
                gen_config.denoise,
            )
            for i in range(task.batch_size)
        ]

        c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id
        batch_input_ids = torch.full(
            (2 * task.batch_size, self.config.num_audio_codebook, max_c_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_audio_mask = torch.zeros((2 * task.batch_size, max_c_len), dtype=torch.bool, device=self.device)
        batch_attention_mask = torch.zeros(
            (2 * task.batch_size, 1, max_c_len, max_c_len),
            dtype=torch.bool,
            device=self.device,
        )

        for i, inp in enumerate(inputs_list):
            c_len, u_len = c_lens[i], task.target_lens[i]
            batch_input_ids[i, :, :c_len] = inp["input_ids"]
            batch_audio_mask[i, :c_len] = inp["audio_mask"]
            batch_attention_mask[i, :, :c_len, :c_len] = True

            batch_input_ids[task.batch_size + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[task.batch_size + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[task.batch_size + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[task.batch_size + i, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (task.batch_size, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )
        tokens_init = tokens.clone()

        timesteps = omnivoice_module._get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        schedules = []
        for target_len in task.target_lens:
            total_mask = target_len * self.config.num_audio_codebook
            remaining = total_mask
            schedule = []
            for step in range(gen_config.num_step):
                value = (
                    remaining
                    if step == gen_config.num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        remaining,
                    )
                )
                schedule.append(int(value))
                remaining -= int(value)
            schedules.append(schedule)

        layer_ids = torch.arange(self.config.num_audio_codebook, device=self.device).view(1, -1, 1)

        if capture_root is not None and task.batch_size == 1:
            self._phase0_debug_invariants = {
                "batch_input_ids_before_step_00": batch_input_ids.detach().cpu().contiguous(),
                "batch_audio_mask": batch_audio_mask.detach().cpu().contiguous(),
                "batch_attention_mask": batch_attention_mask.detach().cpu().contiguous(),
                "tokens_init": tokens_init.detach().cpu().contiguous(),
            }
            self._phase0_debug_payload = {
                "mode": determine_generation_mode(task.ref_audio_tokens[0], task.instructs[0]),
                "capture_steps": sorted(capture_steps),
                "capture_layers": capture_layers,
                "c_lens": c_lens,
                "target_lens": task.target_lens,
                "max_c_len": max_c_len,
                "timesteps": timesteps,
                "schedules": schedules,
            }

            inputs_embeds = self._prepare_embed_inputs(batch_input_ids, batch_audio_mask)
            llm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = llm_outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("expected hidden states for debug capture")

            forward_bundle = {
                "inputs_embeds": inputs_embeds.detach().cpu().contiguous(),
            }
            for layer in capture_layers:
                if layer == "final":
                    forward_bundle["final_hidden"] = llm_outputs[0].detach().cpu().contiguous()
                    continue
                layer_index = int(layer)
                forward_bundle[f"hidden_layer_{layer_index:02d}"] = (
                    hidden_states[layer_index + 1].detach().cpu().contiguous()
                )
            self._phase0_debug_forward = forward_bundle

        for step in range(gen_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            for i in range(task.batch_size):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len = c_lens[i]
                target_len = task.target_lens[i]
                c_logits = batch_logits[i : i + 1, :, c_len - target_len : c_len, :]
                u_logits = batch_logits[task.batch_size + i : task.batch_size + i + 1, :, :target_len, :]

                pred_tokens, confidence_scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
                selection_scores = confidence_scores.clone()
                selection_scores = selection_scores - (layer_ids * gen_config.layer_penalty_factor)

                if gen_config.position_temperature > 0.0:
                    selection_scores = omnivoice_module._gumbel_sample(
                        selection_scores, gen_config.position_temperature
                    )

                sample_tokens = tokens[i : i + 1, :, :target_len]
                selection_scores.masked_fill_(sample_tokens != self.config.audio_mask_id, -float("inf"))
                _, topk_idx = torch.topk(selection_scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                tokens[i : i + 1, :, :target_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - target_len : c_len] = sample_tokens
                batch_input_ids[task.batch_size + i : task.batch_size + i + 1, :, :target_len] = sample_tokens

                if capture_root is not None and step in capture_steps and task.batch_size == 1:
                    step_path = capture_root / "steps" / f"step_{step:02d}.safetensors"
                    save_tensor_bundle(
                        step_path,
                        {
                            "batch_input_ids_before_step": batch_input_ids,
                            "c_logits": c_logits,
                            "u_logits": u_logits,
                            "pred_tokens": pred_tokens,
                            "confidence_scores": confidence_scores,
                            "tokens_after_step": sample_tokens,
                        },
                    )

        return [tokens[i, :, : task.target_lens[i]] for i in range(task.batch_size)]

    model._generate_iterative = MethodType(instrumented_generate_iterative, model)


def serialize_generation_config(gen_config: OmniVoiceGenerationConfig) -> dict[str, Any]:
    return asdict(gen_config)


def determine_generation_mode(
    ref_audio_tokens: torch.Tensor | None,
    instruct: str | None,
) -> str:
    if ref_audio_tokens is not None:
        return "clone"
    if instruct is not None:
        return "design"
    return "auto"


def build_prompt_contract(
    model: OmniVoice,
    text: str,
    num_target_tokens: int,
    ref_text: str | None,
    ref_audio_tokens: torch.Tensor | None,
    lang: str | None,
    instruct: str | None,
    denoise: bool,
    prepared_input_key: str,
    audio_mask_key: str,
) -> dict[str, Any]:
    style_text = ""
    if denoise and ref_audio_tokens is not None:
        style_text += "<|denoise|>"
    lang_str = lang if lang else "None"
    instruct_str = instruct if instruct else "None"
    style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
    style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

    full_text = omnivoice_module._combine_text(ref_text=ref_text, text=text)
    style_token_ids = model.text_tokenizer(style_text, return_tensors="pt").input_ids.squeeze(0)
    text_token_ids = model.text_tokenizer(
        f"<|text_start|>{full_text}<|text_end|>",
        return_tensors="pt",
    ).input_ids.squeeze(0)

    style_length = int(style_token_ids.numel())
    text_length = int(text_token_ids.numel())
    ref_audio_length = int(ref_audio_tokens.size(-1)) if ref_audio_tokens is not None else 0
    target_length = int(num_target_tokens)
    target_start_idx = style_length + text_length + ref_audio_length

    return {
        "mode": determine_generation_mode(ref_audio_tokens, instruct),
        "style_text": style_text,
        "full_text": full_text,
        "style_token_ids": style_token_ids.tolist(),
        "text_token_ids": text_token_ids.tolist(),
        "segments": {
            "style_length": style_length,
            "text_length": text_length,
            "ref_audio_length": ref_audio_length,
            "target_length": target_length,
        },
        "target_start_idx": target_start_idx,
        "total_length": target_start_idx + target_length,
        "prepared_input_key": prepared_input_key,
        "audio_mask_key": audio_mask_key,
    }


def decode_raw_audio(model: OmniVoice, tokens: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    tokenizer_device = model.audio_tokenizer.device
    if isinstance(tokens, list):
        chunk_audios = [
            model.audio_tokenizer.decode(chunk.to(tokenizer_device).unsqueeze(0)).audio_values[0].cpu()
            for chunk in tokens
        ]
        return omnivoice_module.cross_fade_chunks(chunk_audios, model.sampling_rate)
    return model.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0)).audio_values[0].cpu()


def ensure_tensor3(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise RuntimeError(f"expected 3D tensor for stage1 debug, got {tuple(tensor.shape)}")
    return tensor.detach().cpu().contiguous()


def capture_stage1_debug_single(
    model: OmniVoice,
    tokens: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tokenizer = model.audio_tokenizer
    tokenizer_device = tokenizer.device
    audio_codes = tokens.unsqueeze(0).to(tokenizer_device).transpose(0, 1)
    projected_sum = None
    for index, indices in enumerate(audio_codes):
        quantizer = tokenizer.quantizer.quantizers[index]
        quantized = quantizer.codebook.decode(indices)
        projected = quantizer.project_out(quantized)
        projected_sum = projected if projected_sum is None else projected_sum + projected
    if projected_sum is None:
        raise RuntimeError("stage1 quantizer trace is empty")

    quantizer_output = projected_sum.transpose(1, 2)
    fc2_output = tokenizer.fc2(quantizer_output.transpose(1, 2))
    decoder_input = fc2_output.transpose(1, 2)

    tensors = {
        "project_out": ensure_tensor3(projected_sum),
        "quantizer_output": ensure_tensor3(quantizer_output),
        "fc2_output": ensure_tensor3(fc2_output),
        "decoder_input": ensure_tensor3(decoder_input),
    }

    hidden = tokenizer.acoustic_decoder.conv1(decoder_input)
    for index, block in enumerate(tokenizer.acoustic_decoder.block):
        hidden = block(hidden)
        tensors[f"decoder_block_{index:02d}"] = ensure_tensor3(hidden)
    raw_waveform = tokenizer.acoustic_decoder.conv2(tokenizer.acoustic_decoder.snake1(hidden))
    return tensors, ensure_tensor3(raw_waveform)


def capture_stage1_debug(
    model: OmniVoice,
    generated_tokens: torch.Tensor | list[torch.Tensor],
    final_audio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if isinstance(generated_tokens, list):
        tensors: dict[str, torch.Tensor] = {}
        raw_chunks: list[torch.Tensor] = []
        for index, chunk_tokens in enumerate(generated_tokens):
            chunk_tensors, chunk_raw = capture_stage1_debug_single(model, chunk_tokens)
            for name, tensor in chunk_tensors.items():
                tensors[f"chunk_{index:02d}_{name}"] = tensor
            tensors[f"chunk_{index:02d}_raw_waveform"] = chunk_raw
            raw_chunks.append(chunk_raw.squeeze(0))
        tensors["raw_waveform"] = ensure_tensor3(
            omnivoice_module.cross_fade_chunks(raw_chunks, model.sampling_rate)
        )
    else:
        tensors, raw_waveform = capture_stage1_debug_single(model, generated_tokens)
        tensors["raw_waveform"] = raw_waveform
    tensors["final_waveform"] = ensure_tensor3(final_audio)
    return tensors


def compute_chunk_texts(model: OmniVoice, task, gen_config: OmniVoiceGenerationConfig) -> list[str]:
    avg_tokens_per_char = task.target_lens[0] / len(task.texts[0])
    text_chunk_len = int(
        gen_config.audio_chunk_duration * model.audio_tokenizer.config.frame_rate / avg_tokens_per_char
    )
    return chunk_text_punctuation(
        text=task.texts[0],
        chunk_len=text_chunk_len,
        min_chunk_len=3,
    )


def build_case_inputs_and_prompt_contracts(
    model: OmniVoice,
    task,
    gen_config: OmniVoiceGenerationConfig,
    generated_tokens: torch.Tensor | list[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if isinstance(generated_tokens, list):
        result: dict[str, torch.Tensor] = {}
        prompt_contracts: list[dict[str, Any]] = []
        chunk_texts = compute_chunk_texts(model, task, gen_config)
        first_chunk_tokens = generated_tokens[0]
        for index, chunk_tokens in enumerate(generated_tokens):
            if task.ref_audio_tokens[0] is not None:
                ref_text = task.ref_texts[0]
                ref_audio_tokens = task.ref_audio_tokens[0]
            elif index == 0:
                ref_text = None
                ref_audio_tokens = None
            else:
                ref_text = chunk_texts[0]
                ref_audio_tokens = first_chunk_tokens
            prepared = model._prepare_inference_inputs(
                text=chunk_texts[index],
                num_target_tokens=chunk_tokens.size(-1),
                ref_text=ref_text,
                ref_audio_tokens=ref_audio_tokens,
                lang=task.langs[0],
                instruct=task.instructs[0],
                denoise=gen_config.denoise,
            )
            prepared_key = f"prepared_input_ids_chunk_{index:02d}"
            audio_mask_key = f"audio_mask_chunk_{index:02d}"
            result[prepared_key] = prepared["input_ids"]
            result[audio_mask_key] = prepared["audio_mask"]
            prompt_contracts.append(
                build_prompt_contract(
                    model=model,
                    text=chunk_texts[index],
                    num_target_tokens=chunk_tokens.size(-1),
                    ref_text=ref_text,
                    ref_audio_tokens=ref_audio_tokens,
                    lang=task.langs[0],
                    instruct=task.instructs[0],
                    denoise=gen_config.denoise,
                    prepared_input_key=prepared_key,
                    audio_mask_key=audio_mask_key,
                )
            )
        if task.ref_audio_tokens[0] is not None:
            result["ref_audio_tokens"] = task.ref_audio_tokens[0]
        return result, prompt_contracts

    prepared = model._prepare_inference_inputs(
        text=task.texts[0],
        num_target_tokens=generated_tokens.size(-1),
        ref_text=task.ref_texts[0],
        ref_audio_tokens=task.ref_audio_tokens[0],
        lang=task.langs[0],
        instruct=task.instructs[0],
        denoise=gen_config.denoise,
    )
    result = {
        "prepared_input_ids": prepared["input_ids"],
        "audio_mask": prepared["audio_mask"],
    }
    if task.ref_audio_tokens[0] is not None:
        result["ref_audio_tokens"] = task.ref_audio_tokens[0]
    prompt_contract = build_prompt_contract(
        model=model,
        text=task.texts[0],
        num_target_tokens=generated_tokens.size(-1),
        ref_text=task.ref_texts[0],
        ref_audio_tokens=task.ref_audio_tokens[0],
        lang=task.langs[0],
        instruct=task.instructs[0],
        denoise=gen_config.denoise,
        prepared_input_key="prepared_input_ids",
        audio_mask_key="audio_mask",
    )
    return result, [prompt_contract]


def build_prepared_json(
    model: OmniVoice,
    task,
    gen_config: OmniVoiceGenerationConfig,
    generated_tokens: torch.Tensor | list[torch.Tensor],
    prompt_contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared = {
        "batch_size": task.batch_size,
        "texts": task.texts,
        "langs": task.langs,
        "instructs": task.instructs,
        "ref_texts": task.ref_texts,
        "target_lens_estimated": task.target_lens,
        "ref_rms": task.ref_rms,
        "frame_rate": model.audio_tokenizer.config.frame_rate,
        "generation_config": serialize_generation_config(gen_config),
        "prompt_contracts": prompt_contracts,
    }
    if isinstance(generated_tokens, list):
        chunk_texts = compute_chunk_texts(model, task, gen_config)
        prepared["chunk_plan"] = {
            "kind": "chunked",
            "chunk_texts": chunk_texts,
            "chunk_target_lens_actual": [chunk.size(-1) for chunk in generated_tokens],
            "chunks": [
                {
                    "index": index,
                    "text": chunk_texts[index],
                    "target_len_actual": int(chunk.size(-1)),
                    "prepared_input_key": prompt_contracts[index]["prepared_input_key"],
                    "audio_mask_key": prompt_contracts[index]["audio_mask_key"],
                    "segments": prompt_contracts[index]["segments"],
                    "target_start_idx": prompt_contracts[index]["target_start_idx"],
                    "total_length": prompt_contracts[index]["total_length"],
                }
                for index, chunk in enumerate(generated_tokens)
            ],
        }
    else:
        prepared["chunk_plan"] = {
            "kind": "single",
            "chunk_target_lens_actual": [generated_tokens.size(-1)],
        }
        prepared["prompt_contract"] = prompt_contracts[0]
        for key in [
            "mode",
            "style_text",
            "full_text",
            "style_token_ids",
            "text_token_ids",
            "segments",
            "target_start_idx",
            "total_length",
        ]:
            prepared[key] = prompt_contracts[0][key]
    return prepared


def tokens_to_bundle(tokens: torch.Tensor | list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(tokens, list):
        return {f"chunk_{index:02d}": chunk for index, chunk in enumerate(tokens)}
    return {"tokens": tokens}


def verify_case_outputs(
    case_id: str,
    token_bundle: dict[str, torch.Tensor],
    audio_mask_id: int,
    final_audio: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    for name, tensor in token_bundle.items():
        if tensor.dim() != 2 or tensor.size(0) != 8:
            raise RuntimeError(f"{case_id}: unexpected token shape for {name}: {tuple(tensor.shape)}")
        if torch.any(tensor == audio_mask_id):
            raise RuntimeError(f"{case_id}: audio_mask_id remains in {name}")
    if final_audio.dim() != 2 or final_audio.size(0) != 1:
        raise RuntimeError(f"{case_id}: unexpected final audio shape {tuple(final_audio.shape)}")
    if sample_rate != 24000:
        raise RuntimeError(f"{case_id}: expected 24000 Hz output, got {sample_rate}")
    return {
        "token_shapes": {name: list(tensor.shape) for name, tensor in token_bundle.items()},
        "audio_shape": list(final_audio.shape),
        "sample_rate": sample_rate,
    }


def run_case_once(
    model_dir: Path,
    base_device: str,
    base_dtype: str,
    seed: int,
    case: dict[str, Any],
    case_dir: Path | None = None,
    capture_stage1_debug_artifacts: bool = False,
) -> dict[str, Any]:
    device = case.get("debug", {}).get("device", base_device)
    dtype = case.get("debug", {}).get("dtype", base_dtype)
    if case.get("debug"):
        MODEL_CACHE.pop((str(model_dir), device, dtype), None)
    model = build_model(model_dir, device=device, dtype=dtype)
    set_determinism(seed)
    if case_dir is not None and case.get("debug"):
        patch_instrumented_generate_iterative(
            model,
            capture_steps=set(case.get("debug", {}).get("capture_steps", [])),
            capture_layers=case.get("debug", {}).get("capture_layers"),
            capture_root=case_dir,
        )
    else:
        patch_instrumented_generate_iterative(model)

    generation = dict(case.get("generation", {}))
    gen_config = OmniVoiceGenerationConfig.from_dict(generation)
    raw_inputs = {
        "id": case["id"],
        "kind": case["kind"],
        "text": case["text"],
        "language": case.get("language"),
        "instruct": case.get("instruct"),
        "ref_audio": case.get("ref_audio"),
        "ref_text": case.get("ref_text"),
        "generation": generation,
    }
    with torch.inference_mode():
        task = model._preprocess_all(
            text=case["text"],
            language=case.get("language"),
            ref_text=case.get("ref_text"),
            ref_audio=case.get("ref_audio"),
            instruct=case.get("instruct"),
            preprocess_prompt=generation.get("preprocess_prompt", True),
            speed=generation.get("speed"),
            duration=generation.get("duration"),
        )

        short_idx, long_idx = task.get_indices(gen_config, model.audio_tokenizer.config.frame_rate)
        if short_idx and long_idx:
            raise RuntimeError(f"{case['id']}: expected a single path, got mixed short/long indices")

        if short_idx:
            generated_tokens: torch.Tensor | list[torch.Tensor] = model._generate_iterative(task, gen_config)[0]
        else:
            generated_tokens = model._generate_chunked(task, gen_config)[0]

        raw_audio = decode_raw_audio(model, generated_tokens)
        final_audio = model._decode_and_post_process(generated_tokens, task.ref_rms[0], gen_config)
        stage1_debug = (
            capture_stage1_debug(model, generated_tokens, final_audio)
            if capture_stage1_debug_artifacts
            else None
        )
        token_bundle = tokens_to_bundle(generated_tokens)
        inputs_bundle, prompt_contracts = build_case_inputs_and_prompt_contracts(
            model,
            task,
            gen_config,
            generated_tokens,
        )
        prepared = build_prepared_json(
            model,
            task,
            gen_config,
            generated_tokens,
            prompt_contracts,
        )
        verification = verify_case_outputs(
            case["id"],
            token_bundle,
            model.config.audio_mask_id,
            final_audio,
            model.sampling_rate,
        )

    debug_payload = getattr(model, "_phase0_debug_payload", None)
    debug_invariants = getattr(model, "_phase0_debug_invariants", None)
    debug_forward = getattr(model, "_phase0_debug_forward", None)
    if case.get("debug") and case_dir is not None:
        if debug_payload is None or debug_invariants is None or debug_forward is None:
            raise RuntimeError(f"{case['id']}: missing debug capture artifacts")
        inputs_bundle = {
            **inputs_bundle,
            **debug_invariants,
        }

    return {
        "raw_inputs": raw_inputs,
        "prepared": prepared,
        "inputs_bundle": inputs_bundle,
        "token_bundle": token_bundle,
        "raw_audio": raw_audio,
        "final_audio": final_audio,
        "verification": verification,
        "runtime": {
            "device": str(model.device),
            "dtype": dtype,
            "seed": seed,
        },
        "token_hash": tensor_bundle_hash(token_bundle),
        "debug_payload": debug_payload,
        "debug_forward": debug_forward,
        "stage1_debug": stage1_debug,
    }


def write_case_artifacts(case_dir: Path, result: dict[str, Any], sample_rate: int) -> dict[str, str]:
    write_json(case_dir / "case.json", result["raw_inputs"])
    write_json(case_dir / "prepared.json", result["prepared"])
    save_tensor_bundle(case_dir / "inputs.safetensors", result["inputs_bundle"])
    save_tensor_bundle(case_dir / "final_tokens.safetensors", result["token_bundle"])
    save_audio(case_dir / "decoded_raw.wav", result["raw_audio"], sample_rate)
    save_audio(case_dir / "final.wav", result["final_audio"], sample_rate)
    if result["debug_payload"] is not None:
        write_json(case_dir / "debug.json", result["debug_payload"])
    if result["debug_forward"] is not None:
        save_tensor_bundle(case_dir / "forward_step_00.safetensors", result["debug_forward"])
    if result["stage1_debug"] is not None:
        save_tensor_bundle(case_dir / "stage1_debug.safetensors", result["stage1_debug"])

    hashes = {}
    for relative in [
        "case.json",
        "prepared.json",
        "inputs.safetensors",
        "final_tokens.safetensors",
        "decoded_raw.wav",
        "final.wav",
    ]:
        hashes[relative] = file_sha256(case_dir / relative)
    if result["debug_payload"] is not None:
        hashes["debug.json"] = file_sha256(case_dir / "debug.json")
    if result["debug_forward"] is not None:
        hashes["forward_step_00.safetensors"] = file_sha256(case_dir / "forward_step_00.safetensors")
    if result["stage1_debug"] is not None:
        hashes["stage1_debug.safetensors"] = file_sha256(case_dir / "stage1_debug.safetensors")

    steps_dir = case_dir / "steps"
    if steps_dir.exists():
        for step_file in sorted(steps_dir.glob("step_*.safetensors")):
            relative = str(step_file.relative_to(case_dir)).replace("\\", "/")
            hashes[relative] = file_sha256(step_file)
    return hashes


def collect_case_manifest(
    case_dir: Path,
    case: dict[str, Any],
    result: dict[str, Any],
    determinism: dict[str, Any],
    sample_rate: int,
) -> dict[str, Any]:
    hashes = write_case_artifacts(case_dir, result, sample_rate)
    return {
        "id": case["id"],
        "kind": case["kind"],
        "path": str(case_dir.relative_to(case_dir.parent.parent)).replace("\\", "/"),
        "files": hashes,
        "verification": result["verification"],
        "runtime": result["runtime"],
        "determinism": determinism,
    }


def run_determinism_check(
    model_dir: Path,
    device: str,
    dtype: str,
    seed: int,
    case: dict[str, Any],
) -> dict[str, Any]:
    first = run_case_once(model_dir, device, dtype, seed, case)
    second = run_case_once(model_dir, device, dtype, seed, case)
    report = {
        "device": case.get("debug", {}).get("device", device),
        "dtype": case.get("debug", {}).get("dtype", dtype),
        "seed": seed,
        "hashes": [first["token_hash"], second["token_hash"]],
        "matched": first["token_hash"] == second["token_hash"],
    }
    if not report["matched"] and str(report["device"]).startswith("cuda"):
        cpu_case = json.loads(json.dumps(case))
        cpu_case.setdefault("debug", {})
        cpu_case["debug"]["device"] = "cpu"
        cpu_case["debug"]["dtype"] = "float32"
        cpu_first = run_case_once(model_dir, device, dtype, seed, cpu_case)
        cpu_second = run_case_once(model_dir, device, dtype, seed, cpu_case)
        report["cpu_fallback"] = {
            "device": "cpu",
            "dtype": "float32",
            "hashes": [cpu_first["token_hash"], cpu_second["token_hash"]],
            "matched": cpu_first["token_hash"] == cpu_second["token_hash"],
        }
    return report


def smoke_test(model_dir: Path, device: str, dtype: str, seed: int) -> dict[str, Any]:
    case = {
        "id": "smoke",
        "kind": "smoke",
        "text": "Short smoke test for OmniVoice.",
        "language": "English",
        "generation": {
            "num_step": 2,
            "guidance_scale": 2.0,
            "t_shift": 0.1,
            "denoise": True,
            "position_temperature": 5.0,
            "class_temperature": 0.0,
            "layer_penalty_factor": 5.0,
            "preprocess_prompt": True,
            "postprocess_output": True,
        },
    }
    result = run_case_once(model_dir, device, dtype, seed, case)
    return {
        "token_hash": result["token_hash"],
        "audio_shape": result["verification"]["audio_shape"],
        "sample_rate": result["verification"]["sample_rate"],
    }


def build_manifest(
    out_dir: Path,
    runtime_info: dict[str, Any],
    environment: dict[str, Any],
    smoke: dict[str, Any],
    cases: list[dict[str, Any]],
    case_manifests: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "root": str(ROOT),
        "out_dir": str(out_dir),
        "runtime": "runtime.json",
        "environment_check": environment,
        "smoke_test": smoke,
        "cases_requested": [case["id"] for case in cases],
        "cases": case_manifests,
        "summary": {
            "case_count": len(case_manifests),
            "main_cases": [case["id"] for case in cases if case["kind"] == "main"],
            "debug_cases": [case["id"] for case in cases if case.get("debug")],
        },
        "runtime_snapshot": runtime_info,
    }


def should_run_determinism(case_id: str) -> bool:
    return case_id in {
        "auto_en_short",
        "clone_user_ref",
        "debug_auto_en_short",
        "debug_clone_user_ref",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Phase 0 Python golden reference for OmniVoice.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--ref-text-file", default=str(DEFAULT_REF_TEXT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--debug-case-ids", default=None)
    parser.add_argument("--debug-device", default=None)
    parser.add_argument("--debug-dtype", default=None)
    parser.add_argument("--capture-steps", default=None)
    parser.add_argument("--capture-layers", default=None)
    parser.add_argument("--capture-stage1-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ref_audio = Path(args.ref_audio).resolve()
    ref_text_file = Path(args.ref_text_file).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ref_text = read_text_file(ref_text_file)
    cases = resolve_cases(
        ref_audio,
        raw_ref_text,
        case_ids=parse_case_ids(args.case_ids),
    )
    apply_debug_overrides(
        cases,
        debug_case_ids=parse_case_ids(args.debug_case_ids),
        debug_device=args.debug_device,
        debug_dtype=args.debug_dtype,
        capture_steps=parse_capture_steps(args.capture_steps),
        capture_layers=parse_capture_layers(args.capture_layers),
    )

    runtime_info = gather_runtime_info(model_dir, args.device, args.dtype, args.seed)
    environment = verify_environment(model_dir)
    smoke = smoke_test(model_dir, args.device, args.dtype, args.seed)

    case_manifests: dict[str, Any] = {}
    for case in cases:
        case_dir = out_dir / case["id"]
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        determinism = (
            run_determinism_check(model_dir, args.device, args.dtype, args.seed, case)
            if should_run_determinism(case["id"])
            else {"skipped": True}
        )
        result = run_case_once(
            model_dir,
            args.device,
            args.dtype,
            args.seed,
            case,
            case_dir=case_dir,
            capture_stage1_debug_artifacts=args.capture_stage1_debug,
        )
        case_manifests[case["id"]] = collect_case_manifest(
            case_dir,
            case,
            result,
            determinism,
            sample_rate=24000,
        )

    write_json(out_dir / "runtime.json", runtime_info)
    manifest = build_manifest(out_dir, runtime_info, environment, smoke, cases, case_manifests)
    write_json(out_dir / "manifest.json", manifest)

    required_files = [out_dir / "runtime.json", out_dir / "manifest.json"]
    for case in cases:
        case_dir = out_dir / case["id"]
        required_files.extend(
            [
                case_dir / "case.json",
                case_dir / "prepared.json",
                case_dir / "inputs.safetensors",
                case_dir / "final_tokens.safetensors",
                case_dir / "decoded_raw.wav",
                case_dir / "final.wav",
            ]
        )
        if case.get("debug"):
            required_files.append(case_dir / "debug.json")
            required_files.append(case_dir / "forward_step_00.safetensors")
            for step in case.get("debug", {}).get("capture_steps", []):
                required_files.append(case_dir / "steps" / f"step_{step:02d}.safetensors")
        if args.capture_stage1_debug:
            required_files.append(case_dir / "stage1_debug.safetensors")
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "manifest": str(out_dir / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
