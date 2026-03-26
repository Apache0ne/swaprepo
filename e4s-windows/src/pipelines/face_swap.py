import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, UnidentifiedImageError
from skimage.transform import resize
from torch.nn import functional as F

from src.datasets.dataset import TO_TENSOR, NORMALIZE, __celebAHQ_masks_to_faceParser_mask_detailed
from src.models.networks import Net3
from src.options.swap_options import SwapFacePipelineOptions
from src.pretrained.face_parsing.face_parsing_demo import (
    faceParsing_demo,
    init_faceParsing_pretrained_model,
    vis_parsing_maps,
)
from src.pretrained.face_vid2vid.driven_demo import (
    drive_source_demo,
    init_facevid2vid_pretrained_model,
)
from src.pretrained.gpen.gpen_demo import GPEN_demo, init_gpen_pretrained_model
from src.utils import torch_utils
from src.utils.alignmengt import calc_alignment_coefficients, compute_transform, crop_image
from src.utils.morphology import dilation, erosion
from src.utils.multi_band_blending import blending
from src.utils.swap_face_mask import faceParser_label_list_detailed, swap_head_mask_revisit_considerGlass


DEFAULT_FACE_VID2VID_CFG = "./pretrained_ckpts/facevid2vid/vox-256.yaml"
DEFAULT_FACE_VID2VID_CKPT = "./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
DEFAULT_GPEN_BASE_DIR = "./pretrained_ckpts/gpen/"
DEFAULT_GPEN_PARAMS = {
    "base_dir": DEFAULT_GPEN_BASE_DIR,
    "in_size": 512,
    "model": "GPEN-BFR-512",
    "use_sr": True,
    "sr_model": "realesrnet",
    "sr_scale": 4,
    "channel_multiplier": 2,
    "narrow": 1,
}
DEFAULT_FACE_PARSING_CKPTS = {
    "default": ("./pretrained_ckpts/face_parsing/79999_iter.pth", ""),
    "segnext": (
        "./pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth",
        "./pretrained_ckpts/face_parsing/segnext.small.512x512.celebamaskhq.160k.py",
    ),
}
ALIGN_OUTPUT_SIZE = 1024
REENACTMENT_CHUNK_SIZE = 16
OUTPUT_ROOT = Path("./example/output/gradio_faceswap_runs")
FACE_SWAP_REGION_NAMES = tuple(faceParser_label_list_detailed)
FACE_SWAP_REGION_INDEX = {name: idx for idx, name in enumerate(FACE_SWAP_REGION_NAMES)}
FACE_SWAP_DEFAULT_REGION_NAMES = ("lip", "eyebrows", "eyes", "nose", "skin", "mouth")
FACE_SWAP_DEFAULT_REGION_INDICES = tuple(FACE_SWAP_REGION_INDEX[name] for name in FACE_SWAP_DEFAULT_REGION_NAMES)
FACE_SWAP_SWAPPABLE_REGION_NAMES = tuple(name for name in FACE_SWAP_REGION_NAMES if name != "background")
FACE_SWAP_SWAPPABLE_REGION_INDICES = tuple(FACE_SWAP_REGION_INDEX[name] for name in FACE_SWAP_SWAPPABLE_REGION_NAMES)


@dataclass
class MediaProbeResult:
    media_type: str
    total_frames: int
    fps: float | None
    duration_seconds: float | None
    frame_durations_ms: list[int] | None
    selected_indices: list[int]
    selected_count: int
    output_fps: float | None
    estimated_output_duration_seconds: float | None


@dataclass
class FaceSwapRunResult:
    media_type: str
    output_path: str
    preview_path: str
    status_text: str
    artifacts_path: str | None = None
    gallery_items: list[tuple[str, str]] | None = None


@dataclass
class AlignedImageFrame:
    name: str
    aligned_image: Image.Image
    original_image: Image.Image
    inverse_transform: np.ndarray | None


def create_masks(mask, outer_dilation=0, operation="dilation"):
    radius = outer_dilation
    temp = copy.deepcopy(mask)
    if operation == "dilation":
        full_mask = dilation(
            temp,
            torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
            engine="convolution",
        )
        border_mask = full_mask - temp
    elif operation == "erosion":
        full_mask = erosion(
            temp,
            torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
            engine="convolution",
        )
        border_mask = temp - full_mask
    elif operation == "expansion":
        full_mask = dilation(
            temp,
            torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
            engine="convolution",
        )
        erosion_mask = erosion(
            temp,
            torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
            engine="convolution",
        )
        border_mask = full_mask - erosion_mask
    else:
        raise ValueError(f"Unsupported mask operation: {operation}")

    border_mask = border_mask.clip(0, 1)
    content_mask = mask
    return content_mask, border_mask, full_mask


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert("RGBA")
    pasted_image = dst_image.copy().convert("RGBA")
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image


def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=None, belowFace_interpolation=False):
    if comp_indices is None:
        comp_indices = []
    selected_indices = set(comp_indices)

    style_vectors = copy.deepcopy(style_vectors1)

    for comp_idx in comp_indices:
        style_vectors[:, comp_idx, :] = style_vectors2[:, comp_idx, :]

    if 7 in selected_indices and torch.sum(style_vectors2[:, 7, :]) == 0:
        style_vectors[:, 7, :] = (style_vectors1[:, 7, :] + style_vectors2[:, 7, :]) / 2

    if 9 in selected_indices and torch.sum(style_vectors2[:, 9, :]) == 0:
        style_vectors[:, 9, :] = style_vectors1[:, 9, :]

    if belowFace_interpolation and 8 in selected_indices:
        style_vectors[:, 8, :] = (style_vectors1[:, 8, :] + style_vectors2[:, 8, :]) / 2

    return style_vectors


def build_swap_options(**overrides):
    parser = SwapFacePipelineOptions().parser
    opts = parser.parse_args([])
    for key, value in overrides.items():
        setattr(opts, key, value)
    return opts


def clone_opts(opts, **overrides):
    cloned = SimpleNamespace(**vars(opts))
    for key, value in overrides.items():
        setattr(cloned, key, value)
    return cloned


def normalize_start_frame(value):
    if value in (None, ""):
        return 0
    return max(0, int(value))


def normalize_frame_step(value):
    if value in (None, ""):
        return 1
    return max(1, int(value))


def normalize_max_frames(value):
    if value in (None, ""):
        return None
    return max(1, int(value))


def select_frame_indices(total_frames, start_frame=0, frame_step=1, max_frames=None):
    start_frame = normalize_start_frame(start_frame)
    frame_step = normalize_frame_step(frame_step)
    max_frames = normalize_max_frames(max_frames)

    indices = list(range(start_frame, max(total_frames, 0), frame_step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def media_display_name(path_or_image, fallback):
    if isinstance(path_or_image, (str, os.PathLike)):
        return Path(path_or_image).stem
    return fallback


def load_rgb_image(path_or_image):
    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    if isinstance(path_or_image, np.ndarray):
        if path_or_image.ndim == 2:
            return Image.fromarray(path_or_image).convert("RGB")
        return Image.fromarray(path_or_image[..., :3]).convert("RGB")

    path = str(path_or_image)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def parse_target_mask_path(path):
    if not path:
        return None
    target_mask = Image.open(path).convert("L")
    return __celebAHQ_masks_to_faceParser_mask_detailed(target_mask)


def load_gif_frames(path):
    with Image.open(path) as image:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
        durations = [int(frame.info.get("duration", image.info.get("duration", 100))) for frame in ImageSequence.Iterator(image)]
    return frames, durations


def load_video_frames(path, selected_indices):
    if len(selected_indices) == 0:
        return []

    selected_set = set(selected_indices)
    last_index = selected_indices[-1]
    frames = []

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_index = 0
    while frame_index <= last_index:
        success, frame = cap.read()
        if not success:
            break
        if frame_index in selected_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_index += 1

    cap.release()
    return frames


def find_ffmpeg():
    candidates = []
    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(which_path)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "Library", "bin", "ffmpeg.exe"),
                os.path.join(conda_prefix, "bin", "ffmpeg"),
                os.path.join(conda_prefix, "ffmpeg.exe"),
            ]
        )

    python_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            str(python_dir / "ffmpeg.exe"),
            str(python_dir / "ffmpeg"),
            str(python_dir.parent / "Library" / "bin" / "ffmpeg.exe"),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def detect_media_type(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return "gif"

    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) > 1:
                return "gif"
            return "image"
    except UnidentifiedImageError:
        pass
    except OSError:
        pass

    cap = cv2.VideoCapture(path)
    is_video = cap.isOpened()
    cap.release()
    if is_video:
        return "video"

    return "image"


def compute_gif_selected_durations(frame_durations_ms, selected_indices, frame_step):
    selected_durations = []
    for index, selected_index in enumerate(selected_indices):
        if index + 1 < len(selected_indices):
            next_index = selected_indices[index + 1]
        else:
            next_index = min(selected_index + frame_step, len(frame_durations_ms))
        duration = sum(frame_durations_ms[selected_index:next_index]) or frame_durations_ms[selected_index]
        selected_durations.append(duration)
    return selected_durations


def probe_media(path, start_frame=0, frame_step=1, max_frames=None):
    media_type = detect_media_type(path)
    start_frame = normalize_start_frame(start_frame)
    frame_step = normalize_frame_step(frame_step)
    max_frames = normalize_max_frames(max_frames)

    if media_type == "image":
        return MediaProbeResult(
            media_type="image",
            total_frames=1,
            fps=None,
            duration_seconds=None,
            frame_durations_ms=None,
            selected_indices=[0],
            selected_count=1,
            output_fps=None,
            estimated_output_duration_seconds=None,
        )

    if media_type == "gif":
        _, frame_durations_ms = load_gif_frames(path)
        total_frames = len(frame_durations_ms)
        selected_indices = select_frame_indices(total_frames, start_frame, frame_step, max_frames)
        selected_durations = compute_gif_selected_durations(frame_durations_ms, selected_indices, frame_step)
        total_duration_seconds = sum(frame_durations_ms) / 1000.0 if total_frames else 0.0
        selected_duration_seconds = sum(selected_durations) / 1000.0 if selected_durations else 0.0
        return MediaProbeResult(
            media_type="gif",
            total_frames=total_frames,
            fps=None,
            duration_seconds=total_duration_seconds,
            frame_durations_ms=frame_durations_ms,
            selected_indices=selected_indices,
            selected_count=len(selected_indices),
            output_fps=None,
            estimated_output_duration_seconds=selected_duration_seconds,
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open target media: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0

    selected_indices = select_frame_indices(total_frames, start_frame, frame_step, max_frames)
    output_fps = fps / frame_step if frame_step > 1 else fps
    duration_seconds = total_frames / fps if total_frames else 0.0
    selected_duration_seconds = len(selected_indices) / output_fps if selected_indices else 0.0

    return MediaProbeResult(
        media_type="video",
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        frame_durations_ms=None,
        selected_indices=selected_indices,
        selected_count=len(selected_indices),
        output_fps=output_fps,
        estimated_output_duration_seconds=selected_duration_seconds,
    )


def format_media_summary(probe_result):
    lines = [f"Detected target type: {probe_result.media_type}"]
    if probe_result.media_type == "image":
        lines.append("Still image target")
        lines.append("Frame controls: ignored")
        return "\n".join(lines)

    lines.append(f"Total frames: {probe_result.total_frames}")
    if probe_result.fps is not None:
        lines.append(f"Input fps: {probe_result.fps:.3f}")
    if probe_result.duration_seconds is not None:
        lines.append(f"Input duration: {probe_result.duration_seconds:.2f}s")
    lines.append(f"Selected frames: {probe_result.selected_count}")
    if probe_result.output_fps is not None:
        lines.append(f"Output fps: {probe_result.output_fps:.3f}")
    if probe_result.estimated_output_duration_seconds is not None:
        lines.append(f"Estimated output duration: {probe_result.estimated_output_duration_seconds:.2f}s")
    return "\n".join(lines)


def ensure_output_dir(output_dir=None):
    if output_dir is None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = OUTPUT_ROOT / run_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def archive_directory(directory):
    archive_base = str(directory)
    archive_path = shutil.make_archive(archive_base, "zip", root_dir=directory)
    return archive_path


def save_preview_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return str(output_path)


def append_gallery_item(items, path, caption):
    if path and Path(path).exists():
        items.append((str(path), caption))


def normalize_region_selection(selection, default_indices=FACE_SWAP_DEFAULT_REGION_INDICES):
    if selection is None:
        return list(default_indices)

    normalized = []
    for item in selection:
        if isinstance(item, str):
            if item not in FACE_SWAP_REGION_INDEX:
                raise ValueError(f"Unknown face-swap region: {item}")
            index = FACE_SWAP_REGION_INDEX[item]
        else:
            index = int(item)
        if index == 0 or index not in FACE_SWAP_REGION_INDEX.values():
            continue
        if index not in normalized:
            normalized.append(index)
    return [index for index in FACE_SWAP_SWAPPABLE_REGION_INDICES if index in normalized]


def region_indices_to_names(indices):
    return [FACE_SWAP_REGION_NAMES[index] for index in indices]


def format_region_caption(prefix, indices):
    region_names = region_indices_to_names(indices)
    suffix = ", ".join(region_names) if region_names else "none"
    return f"{prefix}: {suffix}"


def create_region_selection_overlay(image, mask, selected_indices):
    image_np = np.array(image.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask, dtype=np.uint8)
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(
            mask_np,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if len(selected_indices) == 0:
        dimmed = image_np * 0.3 + 24
        return Image.fromarray(np.clip(dimmed, 0, 255).astype(np.uint8))

    colors = np.array(torch_utils.get_colors(), dtype=np.float32)
    selected_mask = np.isin(mask_np, np.array(selected_indices))
    overlay = image_np.copy()
    overlay[~selected_mask] = overlay[~selected_mask] * 0.2 + 36
    color_layer = colors[np.clip(mask_np, 0, len(colors) - 1)]
    overlay[selected_mask] = overlay[selected_mask] * 0.35 + color_layer[selected_mask] * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def save_video_frames_mp4(frames, output_path, fps):
    if len(frames) == 0:
        raise ValueError("No frames available to save.")

    first_frame = np.array(frames[0].convert("RGB"))
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 0.1),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        for frame in frames:
            frame_bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def save_gif_frames(frames, durations_ms, output_path):
    if len(frames) == 0:
        raise ValueError("No frames available to save.")
    pil_frames = [frame.convert("RGB") for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,
        loop=0,
    )


def mux_video_audio(ffmpeg_path, target_video_path, silent_video_path, output_path, start_frame, selected_count, fps):
    start_time = start_frame / fps
    duration = selected_count / fps
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(target_video_path),
        "-i",
        str(silent_video_path),
        "-map",
        "1:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg audio mux failed")


class FaceSwapRuntime:
    def __init__(self, opts):
        self.base_opts = clone_opts(opts)
        self.generator, self.kp_detector, self.he_estimator, self.estimate_jacobian = init_facevid2vid_pretrained_model(
            DEFAULT_FACE_VID2VID_CFG,
            DEFAULT_FACE_VID2VID_CKPT,
        )
        self.gpen_model = init_gpen_pretrained_model(model_params=DEFAULT_GPEN_PARAMS)
        self.net = Net3(self.base_opts).to(self.base_opts.device)
        save_dict = torch.load(self.base_opts.checkpoint_path)
        self.net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
        self.net.latent_avg = save_dict["latent_avg"].to(self.base_opts.device)
        self.net.eval()
        print("Load E4S pre-trained model success!")
        self._face_parsers = {}
        self._aligner = None

    def get_face_parser(self, face_parser_name):
        if face_parser_name not in DEFAULT_FACE_PARSING_CKPTS:
            raise NotImplementedError(
                "Please choose a valid face parser, "
                f"the current supported models are [ default | segnext ], but {face_parser_name} is given."
            )
        if face_parser_name not in self._face_parsers:
            ckpt_path, config_path = DEFAULT_FACE_PARSING_CKPTS[face_parser_name]
            self._face_parsers[face_parser_name] = init_faceParsing_pretrained_model(
                face_parser_name,
                ckpt_path,
                config_path,
            )
        return self._face_parsers[face_parser_name]

    def get_aligner(self):
        if self._aligner is None:
            import dlib

            predictor = dlib.shape_predictor("./pretrained_ckpts/shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            self._aligner = (predictor, detector)
        return self._aligner

    def align_source_image(self, source, source_name, need_crop=True, only_target_crop=False):
        if only_target_crop:
            return load_rgb_image(source).resize((ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE))
        if not need_crop:
            return load_rgb_image(source).resize((ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE))
        aligned_frame = self._align_inputs([(source_name, source)])[0]
        return aligned_frame.aligned_image

    def align_target_image(self, target, target_name, need_crop=True):
        if not need_crop:
            target_image = load_rgb_image(target).resize((ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE))
            return AlignedImageFrame(
                name=target_name,
                aligned_image=target_image,
                original_image=target_image.copy(),
                inverse_transform=None,
            )
        return self._align_inputs([(target_name, target)])[0]

    def _align_inputs(self, inputs):
        predictor, detector = self.get_aligner()
        aligned_frames = []
        for name, path_or_image in inputs:
            c, x, y = compute_transform(path_or_image, predictor, detector=detector, scale=1.0)
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            aligned_image = crop_image(path_or_image, ALIGN_OUTPUT_SIZE, quad.copy())
            original_image = load_rgb_image(path_or_image)
            inverse_transform = calc_alignment_coefficients(
                quad + 0.5,
                [[0, 0], [0, ALIGN_OUTPUT_SIZE], [ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE], [ALIGN_OUTPUT_SIZE, 0]],
            )
            aligned_frames.append(
                AlignedImageFrame(
                    name=name,
                    aligned_image=aligned_image.convert("RGB"),
                    original_image=original_image.convert("RGB"),
                    inverse_transform=inverse_transform,
                )
            )
        return aligned_frames

    def run_still_swap(
        self,
        source,
        target,
        opts=None,
        output_dir=None,
        target_mask=None,
        need_crop=True,
        verbose=False,
        only_target_crop=False,
        result_name=None,
        shape_regions=None,
        texture_regions=None,
    ):
        opts = clone_opts(self.base_opts) if opts is None else clone_opts(opts)
        opts.verbose = verbose
        output_dir = ensure_output_dir(output_dir)
        shape_region_indices = normalize_region_selection(shape_regions)
        texture_region_indices = normalize_region_selection(texture_regions)

        source_name = media_display_name(source, "source")
        target_name = media_display_name(target, "target")
        if result_name is None:
            result_name = f"swap_{source_name}_to_{target_name}.png"

        gallery_items = []
        source_input = load_rgb_image(source)
        target_input = load_rgb_image(target)
        source_aligned = self.align_source_image(source, source_name, need_crop=need_crop, only_target_crop=only_target_crop)
        target_frame = self.align_target_image(target, target_name, need_crop=need_crop or only_target_crop)
        source_input_path = save_preview_image(source_input, output_dir / "preview_source_input.png")
        target_input_path = save_preview_image(target_input, output_dir / "preview_target_input.png")
        source_aligned_path = save_preview_image(source_aligned, output_dir / "preview_source_aligned.png")
        target_aligned_path = save_preview_image(target_frame.aligned_image, output_dir / "preview_target_aligned.png")
        result_image = self._swap_aligned_target(
            source_aligned,
            target_frame,
            opts,
            target_mask=target_mask,
            verbose=verbose,
            save_dir=output_dir,
            source_name=source_name,
            target_name=target_name,
            result_name=result_name,
            shape_regions=shape_region_indices,
            texture_regions=texture_region_indices,
        )
        output_path = output_dir / result_name
        if not output_path.exists():
            result_image.save(output_path)
        append_gallery_item(gallery_items, source_input_path, "Source input")
        append_gallery_item(gallery_items, target_input_path, "Target input")
        append_gallery_item(gallery_items, source_aligned_path, "Aligned source")
        append_gallery_item(gallery_items, target_aligned_path, "Aligned target")
        append_gallery_item(gallery_items, output_dir / "T_mask_vis.png", "Target mask")
        append_gallery_item(gallery_items, output_dir / "shape_selection_vis.png", format_region_caption("Shape regions", shape_region_indices))
        append_gallery_item(gallery_items, output_dir / "texture_selection_vis.png", format_region_caption("Texture regions", texture_region_indices))
        append_gallery_item(gallery_items, output_dir / f"D_{source_name}_to_{target_name}.png", "Driven face")
        append_gallery_item(gallery_items, output_dir / "D_mask_vis.png", "Driven mask")
        append_gallery_item(gallery_items, output_dir / "swappedMaskVis.png", "Swapped mask")
        artifacts_path = archive_directory(output_dir) if verbose else None
        return FaceSwapRunResult(
            media_type="image",
            output_path=str(output_path),
            preview_path=str(output_path),
            status_text="Face swap completed.",
            artifacts_path=artifacts_path,
            gallery_items=gallery_items,
        )

    def run_media_swap(
        self,
        source,
        target_media_path,
        opts=None,
        output_dir=None,
        start_frame=0,
        frame_step=1,
        max_frames=None,
        verbose=False,
        shape_regions=None,
        texture_regions=None,
    ):
        opts = clone_opts(self.base_opts) if opts is None else clone_opts(opts)
        opts.verbose = verbose
        output_dir = ensure_output_dir(output_dir)
        shape_region_indices = normalize_region_selection(shape_regions)
        texture_region_indices = normalize_region_selection(texture_regions)
        probe_result = probe_media(target_media_path, start_frame=start_frame, frame_step=frame_step, max_frames=max_frames)

        if probe_result.media_type == "image":
            return self.run_still_swap(
                source,
                target_media_path,
                opts=opts,
                output_dir=output_dir,
                target_mask=None,
                need_crop=True,
                verbose=verbose,
                shape_regions=shape_region_indices,
                texture_regions=texture_region_indices,
            )

        if probe_result.selected_count == 0:
            raise ValueError("No frames selected to process.")

        source_name = media_display_name(source, "source")
        target_name = media_display_name(target_media_path, "target")
        gallery_items = []
        source_input = load_rgb_image(source)
        source_aligned = self.align_source_image(source, source_name, need_crop=True)
        source_input_path = save_preview_image(source_input, output_dir / "preview_source_input.png")
        source_aligned_path = save_preview_image(source_aligned, output_dir / "preview_source_aligned.png")
        append_gallery_item(gallery_items, source_input_path, "Source input")
        append_gallery_item(gallery_items, source_aligned_path, "Aligned source")
        source_256 = resize(np.array(source_aligned) / 255.0, (256, 256))

        if probe_result.media_type == "gif":
            all_target_frames, all_durations_ms = load_gif_frames(target_media_path)
            selected_target_frames = [all_target_frames[index] for index in probe_result.selected_indices]
            selected_durations_ms = compute_gif_selected_durations(
                all_durations_ms,
                probe_result.selected_indices,
                normalize_frame_step(frame_step),
            )
        else:
            selected_target_frames = load_video_frames(target_media_path, probe_result.selected_indices)
            selected_durations_ms = None
        first_frame_index = probe_result.selected_indices[0]
        first_frame_label = f"Selected target frame {first_frame_index}"
        first_frame_input_path = save_preview_image(
            selected_target_frames[0],
            output_dir / f"preview_target_input_{first_frame_index:06d}.png",
        )
        append_gallery_item(gallery_items, first_frame_input_path, first_frame_label)

        generated_frames = []
        warnings = []
        frame_output_root = output_dir / "frames"
        preview_frame_dir = frame_output_root / f"frame_{first_frame_index:06d}"
        frame_output_root.mkdir(parents=True, exist_ok=True)

        for chunk_start in range(0, len(selected_target_frames), REENACTMENT_CHUNK_SIZE):
            chunk_frames = selected_target_frames[chunk_start:chunk_start + REENACTMENT_CHUNK_SIZE]
            chunk_indices = probe_result.selected_indices[chunk_start:chunk_start + REENACTMENT_CHUNK_SIZE]
            aligned_targets = [
                self.align_target_image(frame, f"frame_{frame_index:06d}", need_crop=True)
                for frame, frame_index in zip(chunk_frames, chunk_indices)
            ]
            target_frames_256 = [
                resize(np.array(aligned_target.aligned_image) / 255.0, (256, 256))
                for aligned_target in aligned_targets
            ]
            predictions = drive_source_demo(
                source_256,
                target_frames_256,
                self.generator,
                self.kp_detector,
                self.he_estimator,
                self.estimate_jacobian,
            )

            for aligned_target, prediction, frame_index in zip(aligned_targets, predictions, chunk_indices):
                frame_save_dir = None
                if verbose:
                    frame_save_dir = frame_output_root / f"frame_{frame_index:06d}"
                    frame_save_dir.mkdir(parents=True, exist_ok=True)
                elif frame_index == first_frame_index:
                    frame_save_dir = preview_frame_dir
                    frame_save_dir.mkdir(parents=True, exist_ok=True)
                if frame_index == first_frame_index:
                    preview_target_aligned_path = save_preview_image(
                        aligned_target.aligned_image,
                        output_dir / f"preview_target_aligned_{frame_index:06d}.png",
                    )
                    append_gallery_item(gallery_items, preview_target_aligned_path, "Aligned target")
                frame_result = self._swap_aligned_target(
                    source_aligned,
                    aligned_target,
                    opts,
                    target_mask=None,
                    verbose=verbose,
                    save_dir=frame_save_dir,
                    source_name=source_name,
                    target_name=f"frame_{frame_index:06d}",
                    result_name=f"{frame_index:06d}.png",
                    prediction_override=prediction,
                    shape_regions=shape_region_indices,
                    texture_regions=texture_region_indices,
                )
                generated_frames.append(frame_result.convert("RGB"))

        base_name = f"swap_{source_name}_to_{target_name}"
        append_gallery_item(gallery_items, preview_frame_dir / "T_mask_vis.png", "Target mask")
        append_gallery_item(gallery_items, preview_frame_dir / "shape_selection_vis.png", format_region_caption("Shape regions", shape_region_indices))
        append_gallery_item(gallery_items, preview_frame_dir / "texture_selection_vis.png", format_region_caption("Texture regions", texture_region_indices))
        append_gallery_item(gallery_items, preview_frame_dir / f"D_{source_name}_to_frame_{first_frame_index:06d}.png", "Driven face")
        append_gallery_item(gallery_items, preview_frame_dir / "D_mask_vis.png", "Driven mask")
        append_gallery_item(gallery_items, preview_frame_dir / "swappedMaskVis.png", "Swapped mask")
        artifacts_path = archive_directory(output_dir) if verbose else None

        if probe_result.media_type == "gif":
            output_path = output_dir / f"{base_name}.gif"
            save_gif_frames(generated_frames, selected_durations_ms, output_path)
            status_lines = [
                "Face swap completed.",
                format_media_summary(probe_result),
            ]
            return FaceSwapRunResult(
                media_type="gif",
                output_path=str(output_path),
                preview_path=str(output_path),
                status_text="\n".join(status_lines),
                artifacts_path=artifacts_path,
                gallery_items=gallery_items,
            )

        silent_output_path = output_dir / f"{base_name}_silent.mp4"
        save_video_frames_mp4(generated_frames, silent_output_path, probe_result.output_fps or probe_result.fps or 25.0)
        final_output_path = output_dir / f"{base_name}.mp4"
        shutil.copyfile(silent_output_path, final_output_path)

        frame_step = normalize_frame_step(frame_step)
        ffmpeg_path = find_ffmpeg()
        if frame_step == 1 and ffmpeg_path:
            try:
                mux_video_audio(
                    ffmpeg_path,
                    target_media_path,
                    silent_output_path,
                    final_output_path,
                    normalize_start_frame(start_frame),
                    probe_result.selected_count,
                    probe_result.fps or 25.0,
                )
            except Exception as exc:
                warnings.append(f"Audio mux failed, returning silent video instead: {exc}")
                shutil.copyfile(silent_output_path, final_output_path)
        elif frame_step != 1:
            warnings.append("Audio is disabled when Frame Step is greater than 1.")
        else:
            warnings.append("ffmpeg was not found, returning silent video.")

        status_lines = ["Face swap completed.", format_media_summary(probe_result)]
        if warnings:
            status_lines.append("Warnings:")
            status_lines.extend(warnings)

        return FaceSwapRunResult(
            media_type="video",
            output_path=str(final_output_path),
            preview_path=str(final_output_path),
            status_text="\n".join(status_lines),
            artifacts_path=artifacts_path,
            gallery_items=gallery_items,
        )

    @torch.no_grad()
    def _swap_aligned_target(
        self,
        source_aligned,
        target_frame,
        opts,
        target_mask=None,
        verbose=False,
        save_dir=None,
        source_name="source",
        target_name="target",
        result_name="swap.png",
        prediction_override=None,
        shape_regions=None,
        texture_regions=None,
    ):
        parser = self.get_face_parser(opts.faceParser_name)
        shape_region_indices = normalize_region_selection(shape_regions)
        texture_region_indices = normalize_region_selection(texture_regions)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        S = source_aligned.convert("RGB")
        T = target_frame.aligned_image.convert("RGB")
        T_mask = faceParsing_demo(parser, T, convert_to_seg12=True, model_name=opts.faceParser_name) if target_mask is None else target_mask

        if save_dir is not None:
            if verbose:
                Image.fromarray(T_mask).save(save_dir / "T_mask.png")
            Image.fromarray(vis_parsing_maps(T, T_mask)).save(save_dir / "T_mask_vis.png")

        S_256 = resize(np.array(S) / 255.0, (256, 256))
        T_256 = resize(np.array(T) / 255.0, (256, 256))
        if prediction_override is None:
            prediction = drive_source_demo(
                S_256,
                [T_256],
                self.generator,
                self.kp_detector,
                self.he_estimator,
                self.estimate_jacobian,
            )[0]
        else:
            prediction = prediction_override
        prediction = (prediction * 255).astype(np.uint8)

        driven_bgr = GPEN_demo(prediction[:, :, ::-1], self.gpen_model, aligned=False)
        D = Image.fromarray(driven_bgr[:, :, ::-1])
        if save_dir is not None:
            D.save(save_dir / f"D_{source_name}_to_{target_name}.png")

        D_mask = faceParsing_demo(parser, D, convert_to_seg12=True, model_name=opts.faceParser_name)
        if save_dir is not None:
            if verbose:
                Image.fromarray(D_mask).save(save_dir / "D_mask.png")
            Image.fromarray(vis_parsing_maps(D, D_mask)).save(save_dir / "D_mask_vis.png")

        driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D).to(opts.device).float().unsqueeze(0)
        driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
        driven_mask = (driven_mask * 255).long().to(opts.device).unsqueeze(0)
        driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls=opts.num_seg_cls)

        target = transforms.Compose([TO_TENSOR, NORMALIZE])(T).to(opts.device).float().unsqueeze(0)
        target_mask_tensor = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
        target_mask_tensor = (target_mask_tensor * 255).long().to(opts.device).unsqueeze(0)
        target_onehot = torch_utils.labelMap2OneHot(target_mask_tensor, num_cls=opts.num_seg_cls)

        driven_style_vector, _ = self.net.get_style_vectors(driven, driven_onehot)
        target_style_vector, _ = self.net.get_style_vectors(target, target_onehot)

        if verbose and save_dir is not None:
            torch.save(driven_style_vector, save_dir / "D_style_vec.pt")
            driven_style_codes = self.net.cal_style_codes(driven_style_vector)
            driven_face, _, _ = self.net.gen_img(
                torch.zeros(1, 512, 32, 32).to(opts.device),
                driven_style_codes,
                driven_onehot,
            )
            torch_utils.tensor2im(driven_face[0]).save(save_dir / "D_recon.png")

            torch.save(target_style_vector, save_dir / "T_style_vec.pt")
            target_style_codes = self.net.cal_style_codes(target_style_vector)
            target_face, _, _ = self.net.gen_img(
                torch.zeros(1, 512, 32, 32).to(opts.device),
                target_style_codes,
                target_onehot,
            )
            torch_utils.tensor2im(target_face[0]).save(save_dir / "T_recon.png")

        swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(
            D_mask,
            T_mask,
            selected_indices=shape_region_indices,
        )
        if save_dir is not None:
            create_region_selection_overlay(T, swapped_msk, shape_region_indices).save(save_dir / "shape_selection_vis.png")
            create_region_selection_overlay(T, T_mask, texture_region_indices).save(save_dir / "texture_selection_vis.png")
            if verbose:
                cv2.imwrite(str(save_dir / "swappedMask.png"), swapped_msk)
            swapped_one_hot = torch_utils.labelMap2OneHot(
                torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(),
                num_cls=12,
            )
            torch_utils.tensor2map(swapped_one_hot[0]).save(save_dir / "swappedMaskVis.png")

        swapped_style_vectors = swap_comp_style_vector(
            target_style_vector,
            driven_style_vector,
            list(texture_region_indices),
            belowFace_interpolation=False,
        )
        if verbose and save_dir is not None:
            torch.save(swapped_style_vectors, save_dir / "swapped_style_vec.pt")

        swapped_msk_tensor = transforms.Compose([TO_TENSOR])(Image.fromarray(swapped_msk).convert("L"))
        swapped_msk_tensor = (swapped_msk_tensor * 255).long().to(opts.device).unsqueeze(0)
        swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk_tensor, num_cls=opts.num_seg_cls)
        swapped_style_codes = self.net.cal_style_codes(swapped_style_vectors)
        swapped_face, _, _ = self.net.gen_img(
            torch.zeros(1, 512, 32, 32).to(opts.device),
            swapped_style_codes,
            swapped_onehot,
        )
        swapped_face_image = torch_utils.tensor2im(swapped_face[0])

        foreground_indices = set(shape_region_indices) | set(texture_region_indices)
        if foreground_indices:
            is_foreground = logical_or_reduce(*[swapped_msk_tensor == clz for clz in sorted(foreground_indices)])
        else:
            is_foreground = torch.zeros_like(swapped_msk_tensor, dtype=torch.bool)
        hole_index = torch.from_numpy(hole_map == 255).to(opts.device).unsqueeze(0).unsqueeze(0)
        is_foreground = torch.logical_or(is_foreground, hole_index)
        foreground_mask = is_foreground.float()

        outer_dilation = 5
        if opts.lap_bld:
            content_mask, border_mask, full_mask = create_masks(
                foreground_mask,
                outer_dilation=outer_dilation,
                operation="expansion",
            )
        else:
            content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation)

        content_mask = F.interpolate(content_mask, (ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE), mode="bilinear", align_corners=False)
        content_mask_image = Image.fromarray(255 * content_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))
        full_mask = F.interpolate(full_mask, (ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE), mode="bilinear", align_corners=False)
        full_mask_image = Image.fromarray(255 * full_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))

        if opts.lap_bld:
            content_mask_np = content_mask[0, 0, :, :, None].cpu().numpy()
            border_mask = F.interpolate(border_mask, (ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE), mode="bilinear", align_corners=False)
            border_mask_np = border_mask[0, 0, :, :, None].cpu().numpy()
            border_mask_np = np.repeat(border_mask_np, 3, axis=-1)
            swapped_and_pasted = np.array(swapped_face_image) * content_mask_np + np.array(T) * (1 - content_mask_np)
            swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
            swapped_and_pasted = Image.fromarray(blending(np.array(T), np.array(swapped_and_pasted), mask=border_mask_np))
        else:
            if outer_dilation == 0:
                swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, content_mask_image, radius=outer_dilation)
            else:
                swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, full_mask_image, radius=outer_dilation)

        if target_frame.inverse_transform is not None:
            swapped_rgba = swapped_and_pasted.convert("RGBA")
            pasted_image = target_frame.original_image.convert("RGBA")
            swapped_rgba.putalpha(255)
            projected = swapped_rgba.transform(
                target_frame.original_image.size,
                Image.PERSPECTIVE,
                target_frame.inverse_transform,
                Image.BILINEAR,
            )
            pasted_image.alpha_composite(projected)
            result_image = pasted_image.convert("RGB")
        else:
            result_image = swapped_and_pasted.convert("RGB")

        if save_dir is not None:
            result_image.save(save_dir / result_name)
        return result_image


def load_face_swap_runtime(opts=None):
    if opts is None:
        opts = build_swap_options()
    runtime = FaceSwapRuntime(opts)
    return runtime


def run_cli_face_swap(opts):
    runtime = load_face_swap_runtime(opts)
    target_mask_seg12 = None
    if len(opts.target_mask) != 0:
        target_mask_seg12 = parse_target_mask_path(opts.target_mask)
    return runtime.run_still_swap(
        opts.source,
        opts.target,
        opts=opts,
        output_dir=opts.output_dir,
        target_mask=target_mask_seg12,
        need_crop=True,
        verbose=opts.verbose,
    )
