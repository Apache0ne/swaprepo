import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from demo.gradio_utils import *
from src.datasets.dataset import TO_TENSOR, NORMALIZE
from src.models.networks import Net3
from src.options.edit_options import EditOptions
from src.pipelines.face_swap import (
    build_swap_options,
    format_media_summary,
    load_face_swap_runtime,
    parse_target_mask_path,
    probe_media,
)
from src.pretrained.face_parsing.face_parsing_demo import (
    faceParsing_demo,
    init_faceParsing_pretrained_model,
    is_segnext_available,
)
from src.utils import torch_utils


FACE_SWAP_PARSER_CHOICES = ["default"] + (["segnext"] if is_segnext_available() else [])
FACE_SWAP_REGION_CHOICES = [region for region in COMP if region != "background"]
FACE_SWAP_DEFAULT_REGIONS = ["lip", "eyebrows", "eyes", "nose", "skin", "mouth"]
FACE_SWAP_REGION_PRESETS = {
    "Current Behavior": FACE_SWAP_DEFAULT_REGIONS,
    "Include Hair": FACE_SWAP_DEFAULT_REGIONS + ["hair"],
    "Full Head": FACE_SWAP_REGION_CHOICES,
}


class EditDemoHelper:
    def __init__(self):
        self.opt = EditOptions().parse()

        self.faceParsing_model = init_faceParsing_pretrained_model(
            self.opt.faceParser_name,
            self.opt.faceParsing_ckpt,
            self.opt.segnext_config,
        )
        assert self.opt.faceParsing_ckpt is not None, "please fetch the pre-trained faceParsing model checkpoint!"

        self.rgi_ckpt = self.opt.checkpoint_path
        assert self.opt.checkpoint_path is not None, "please fetch the pre-trained E4S model checkpoint!"

        self.net = Net3(self.opt).eval().to(self.opt.device)
        ckpt_dict = torch.load(self.opt.checkpoint_path)
        self.net.latent_avg = ckpt_dict["latent_avg"].to(self.opt.device) if self.opt.start_from_latent_avg else None
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"], prefix="module."))
        print("Loading Done!")

        self.src_img = None
        self.initial_label_map = None
        self.initial_colored_map = None
        self.ref_img = None
        self.ref_label_map = None
        self.src_texture_vectors = None
        self.ref_texture_vectors = None

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * 2,
            128: 128 * 2,
            256: 64 * 2,
            512: 32 * 2,
            1024: 16 * 2,
        }
        self.noise = [torch.randn(1, 512, 4, 4).to(self.opt.device)]
        for size in [8, 16, 32, 64, 128, 256, 512, 1024]:
            self.noise.append(torch.randn(1, channels[size], size, size).to(self.opt.device))
            self.noise.append(torch.randn(1, channels[size], size, size).to(self.opt.device))

    def esitmate_mask(self, image):
        return faceParsing_demo(
            self.faceParsing_model,
            image,
            convert_to_seg12=True,
            model_name=self.opt.faceParser_name,
        )

    @torch.no_grad()
    def extract_src_texture_vectors(self):
        if self.initial_label_map is None:
            return

        src = transforms.Compose([TO_TENSOR, NORMALIZE])(self.src_img)
        src = src.to(self.opt.device).float().unsqueeze(0)
        src_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.initial_label_map))
        src_mask = (src_mask * 255).long().to(self.opt.device).unsqueeze(0)
        src_onehot = torch_utils.labelMap2OneHot(src_mask, num_cls=self.opt.num_seg_cls)
        self.src_texture_vectors, _ = self.net.get_style_vectors(src, src_onehot)

    @torch.no_grad()
    def extract_ref_texture_vectors(self):
        if self.ref_label_map is None:
            return

        ref = transforms.Compose([TO_TENSOR, NORMALIZE])(self.ref_img)
        ref = ref.to(self.opt.device).float().unsqueeze(0)
        ref_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.ref_label_map))
        ref_mask = (ref_mask * 255).long().to(self.opt.device).unsqueeze(0)
        ref_onehot = torch_utils.labelMap2OneHot(ref_mask, num_cls=self.opt.num_seg_cls)
        self.ref_texture_vectors, _ = self.net.get_style_vectors(ref, ref_onehot)


class FaceSwapDemoHelper:
    def __init__(self):
        self.base_opts = build_swap_options()
        self.runtime = None

    def get_runtime(self):
        if self.runtime is None:
            self.runtime = load_face_swap_runtime(self.base_opts)
        return self.runtime

    def build_run_opts(self, lap_bld, verbose, face_parser_name):
        return build_swap_options(
            lap_bld=lap_bld,
            verbose=verbose,
            faceParser_name=face_parser_name,
            checkpoint_path=self.base_opts.checkpoint_path,
            device=self.base_opts.device,
        )


edit_helper = EditDemoHelper()
swap_helper = FaceSwapDemoHelper()


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _as_numpy_image(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return np.array(value)
    return value


def _load_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return Image.fromarray(value).convert("RGB")
        return Image.fromarray(value[..., :3]).convert("RGB")
    if isinstance(value, str):
        try:
            return Image.open(value).convert("RGB")
        except Exception:
            data = np.fromfile(value, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    raise TypeError(f"Unsupported image input type: {type(value)!r}")


def _make_editor_value(image):
    image = _as_numpy_image(image)
    return {"background": image, "layers": [], "composite": image}


def _normalize_face_swap_regions(regions):
    if not regions:
        return []
    return [region for region in FACE_SWAP_REGION_CHOICES if region in regions]


def apply_face_swap_region_preset(preset_name):
    regions = FACE_SWAP_REGION_PRESETS.get(preset_name, FACE_SWAP_DEFAULT_REGIONS)
    normalized = _normalize_face_swap_regions(regions)
    return normalized, normalized


def format_face_swap_region_summary(shape_regions, texture_regions):
    shape_regions = _normalize_face_swap_regions(shape_regions)
    texture_regions = _normalize_face_swap_regions(texture_regions)
    shape_summary = ", ".join(shape_regions) if shape_regions else "none"
    texture_summary = ", ".join(texture_regions) if texture_regions else "none"
    return "\n".join(
        [
            f"Shape regions: {shape_summary}",
            f"Texture regions: {texture_summary}",
        ]
    )


def _extract_editor_state(edited_mask):
    if isinstance(edited_mask, dict):
        background = _as_numpy_image(edited_mask.get("background"))
        composite = _as_numpy_image(
            _first_not_none(
                edited_mask.get("composite"),
                edited_mask.get("image"),
                edited_mask.get("background"),
            )
        )
        layers = edited_mask.get("layers") or []
        layer = _as_numpy_image(layers[-1] if layers else edited_mask.get("mask"))
    else:
        background = None
        composite = _as_numpy_image(edited_mask)
        layer = None
    return background, composite, layer


def _layer_to_binary_mask(layer, background, composite):
    if layer is not None:
        if layer.ndim == 2:
            return layer != 0
        if layer.shape[-1] == 4:
            return layer[..., 3] != 0
        return np.any(layer[..., :3] != 0, axis=-1)

    if background is not None and composite is not None:
        bg_rgb = background[..., :3] if background.ndim == 3 else background
        comp_rgb = composite[..., :3] if composite.ndim == 3 else composite
        return np.any(comp_rgb != bg_rgb, axis=-1)

    if composite is not None:
        comp_rgb = composite[..., :3] if composite.ndim == 3 else composite
        return np.any(comp_rgb != 0, axis=-1)

    return None


def esitimate_init_mask_fn(image):
    image = _load_image(image)
    label_map = edit_helper.esitmate_mask(image)

    edit_helper.initial_label_map = label_map
    edit_helper.initial_colored_map = label_map_to_colored_mask(edit_helper.initial_label_map)
    edit_helper.src_img = image
    edit_helper.extract_src_texture_vectors()

    return _make_editor_value(edit_helper.initial_colored_map), "Load input image success!"


def esitimate_referece_mask_fn(image):
    image = _load_image(image)
    label_map = edit_helper.esitmate_mask(image)

    edit_helper.ref_label_map = label_map
    edit_helper.ref_img = image
    edit_helper.extract_ref_texture_vectors()

    return "Load reference image success!"


def edit_mask_fn(region_radio, edited_mask):
    if region_radio is None:
        return _make_editor_value(edit_helper.initial_colored_map), "Please choose the region you want to edit on, and try again."

    background, composite, layer = _extract_editor_state(edited_mask)
    mask = _layer_to_binary_mask(layer, background, composite)
    if mask is None:
        return _make_editor_value(edit_helper.initial_colored_map), "Please draw on the mask and try again."

    comp_idx = COMP2INDEX[region_radio]
    current_mask = composite if composite is not None else edit_helper.initial_colored_map
    label_map = colored_mask_to_label_map(current_mask)
    label_map[mask] = comp_idx
    colored_mask_edited = label_map_to_colored_mask(label_map)

    return _make_editor_value(colored_mask_edited), f"Edit {region_radio} region success!"


@torch.no_grad()
def face_shape_edit_fn(edited_mask):
    _, mask, _ = _extract_editor_state(edited_mask)
    mask = mask if mask is not None else edit_helper.initial_colored_map
    label_map = colored_mask_to_label_map(mask)

    onehot = torch_utils.labelMap2OneHot(
        (TO_TENSOR(label_map) * 255).long().to(edit_helper.opt.device).unsqueeze(0),
        num_cls=edit_helper.opt.num_seg_cls,
    )

    style_codes = edit_helper.net.cal_style_codes(edit_helper.src_texture_vectors)
    generated, _, _ = edit_helper.net.gen_img(
        torch.zeros(1, 512, 32, 32).to(onehot.device),
        style_codes,
        onehot,
        randomize_noise=False,
        noise=edit_helper.noise,
    )

    return torch_utils.tensor2im(generated[0]), "Edit shape success!"


@torch.no_grad()
def face_texture_edit_fn(region_groups, alpha):
    regions = region_groups
    if len(regions) == 0:
        return edit_helper.src_img, "Please choose the region you want to mix, and try again."

    mixed_texture_vectors = edit_helper.src_texture_vectors.clone()
    for region in regions:
        idx = COMP2INDEX[region]
        mixed_texture_vectors[0, idx, :] = (
            (1 - alpha) * edit_helper.src_texture_vectors[0, idx, :]
            + alpha * edit_helper.ref_texture_vectors[0, idx, :]
        )

    mixed_style_codes = edit_helper.net.cal_style_codes(mixed_texture_vectors)
    onehot = torch_utils.labelMap2OneHot(
        (TO_TENSOR(edit_helper.initial_label_map) * 255).long().to(edit_helper.opt.device).unsqueeze(0),
        num_cls=edit_helper.opt.num_seg_cls,
    )

    generated, _, _ = edit_helper.net.gen_img(
        torch.zeros(1, 512, 32, 32).to(onehot.device),
        mixed_style_codes,
        onehot,
        randomize_noise=False,
        noise=edit_helper.noise,
    )

    return torch_utils.tensor2im(generated[0]), f"Edit {' '.join(regions)} region(s) success!"


def summarize_target_media(target_media, start_frame, frame_step, max_frames):
    if not target_media:
        return "Upload target media to see summary.", gr.update(visible=False, interactive=False, value=None)

    try:
        probe_result = probe_media(
            target_media,
            start_frame=start_frame,
            frame_step=frame_step,
            max_frames=max_frames,
        )
    except Exception as exc:
        return f"Failed to inspect target media: {exc}", gr.update(visible=False, interactive=False, value=None)

    mask_update = gr.update(visible=probe_result.media_type == "image", interactive=probe_result.media_type == "image")
    if probe_result.media_type != "image":
        mask_update = gr.update(visible=False, interactive=False, value=None)
    return format_media_summary(probe_result), mask_update


def run_face_swap(
    source_image,
    target_media,
    shape_regions,
    texture_regions,
    target_mask,
    lap_bld,
    verbose,
    face_parser_name,
    start_frame,
    frame_step,
    max_frames,
):
    if not source_image:
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=[]),
            "Please upload a source image.",
        )

    if not target_media:
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=[]),
            "Please upload target media.",
        )

    try:
        runtime = swap_helper.get_runtime()
        opts = swap_helper.build_run_opts(lap_bld=lap_bld, verbose=verbose, face_parser_name=face_parser_name)
        shape_regions = _normalize_face_swap_regions(shape_regions)
        texture_regions = _normalize_face_swap_regions(texture_regions)
        probe_result = probe_media(
            target_media,
            start_frame=start_frame,
            frame_step=frame_step,
            max_frames=max_frames,
        )

        if probe_result.media_type == "image":
            target_mask_seg12 = parse_target_mask_path(target_mask) if target_mask else None
            result = runtime.run_still_swap(
                source_image,
                target_media,
                opts=opts,
                target_mask=target_mask_seg12,
                need_crop=True,
                verbose=verbose,
                shape_regions=shape_regions,
                texture_regions=texture_regions,
            )
            status_text = "\n".join(
                [
                    result.status_text,
                    format_face_swap_region_summary(shape_regions, texture_regions),
                    format_media_summary(probe_result),
                ]
            )
        else:
            result = runtime.run_media_swap(
                source_image,
                target_media,
                opts=opts,
                start_frame=start_frame,
                frame_step=frame_step,
                max_frames=max_frames,
                verbose=verbose,
                shape_regions=shape_regions,
                texture_regions=texture_regions,
            )
            status_text = "\n".join([result.status_text, format_face_swap_region_summary(shape_regions, texture_regions)])
    except Exception as exc:
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=[]),
            f"Face swap failed: {exc}",
        )

    image_update = gr.update(
        visible=result.media_type in {"image", "gif"},
        value=result.preview_path if result.media_type in {"image", "gif"} else None,
    )
    video_update = gr.update(
        visible=result.media_type == "video",
        value=result.preview_path if result.media_type == "video" else None,
    )
    result_file_update = gr.update(visible=True, value=result.output_path)
    artifacts_file_update = gr.update(visible=bool(result.artifacts_path), value=result.artifacts_path)
    gallery_update = gr.update(visible=bool(result.gallery_items), value=result.gallery_items or [])

    return image_update, video_update, result_file_update, artifacts_file_update, gallery_update, status_text


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <img src="file/assets/cvpr_banner_homepage.svg" alt="CVPR2023" style="width:250px">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
             Welcome to E4S demo page!
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem">
        Zhian Liu<sup>1*</sup>, Maomao Li<sup>2*</sup>,
        <a href="https://yzhang2016.github.io" style="color:blue;">Yong Zhang</a><sup>2*</sup>,
        Cairong Wang</a><sup>3</sup>,
        <a href="https://qzhang-cv.github.io" style="color:blue;">Qi Zhang</a><sup>2</sup>,
        <a href="https://juewang725.github.io" style="color:blue;">Jue Wang</a><sup>2</sup>,
        <a href="https://nieyongwei.net" style="color:blue;">Yongwei Nie</a><sup>1</sup><br>
        [<a href="https://arxiv.org/abs/2211.14068" style="color:red;">arXiv</a>]
        [<a href="https://e4s2022.github.io" style="color:red;">Project page</a>]
        [<a href="https://github.com/e4s2022/e4s" style="color:red;">GitHub</a>]
        </h2>
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        <sup>1</sup> South China University of Technology, <sup>2</sup>Tencent AI Lab, <sup>3</sup>Tsinghua Shenzhen International Graduate School
        </h3>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Face Swap"):
            with gr.Row():
                swap_source_img = gr.File(
                    label="Source Image",
                    type="filepath",
                    file_types=["image", ".webp"],
                )
                with gr.Column():
                    swap_target_media = gr.File(
                        label="Target Media",
                        type="filepath",
                        file_types=["image", "video", ".gif", ".webp"],
                    )
                    swap_media_summary = gr.Textbox(
                        label="Target Media Summary",
                        value="Upload target media to see summary.",
                        lines=6,
                        interactive=False,
                    )
                    swap_run_btn = gr.Button("Run Face Swap", variant="primary")
                    swap_status = gr.Textbox(
                        label="Run Status / Warnings",
                        value="Ready to run face swap.",
                        lines=8,
                        interactive=False,
                    )

            with gr.Accordion("Region Controls", open=True):
                swap_region_preset = gr.Radio(
                    ["Current Behavior", "Include Hair", "Full Head"],
                    value="Current Behavior",
                    label="Region Preset",
                )
                with gr.Row():
                    swap_shape_regions = gr.CheckboxGroup(
                        choices=FACE_SWAP_REGION_CHOICES,
                        value=FACE_SWAP_DEFAULT_REGIONS,
                        label="Shape Regions",
                        info="Choose which regions inherit source structure/mask.",
                    )
                    swap_texture_regions = gr.CheckboxGroup(
                        choices=FACE_SWAP_REGION_CHOICES,
                        value=FACE_SWAP_DEFAULT_REGIONS,
                        label="Texture Regions",
                        info="Choose which regions inherit source appearance/style.",
                    )

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    swap_face_parser = gr.Radio(
                        FACE_SWAP_PARSER_CHOICES,
                        value="default",
                        label="Face Parser",
                        info=None if "segnext" in FACE_SWAP_PARSER_CHOICES else "SegNeXt is unavailable in this environment.",
                    )
                    swap_lap_bld = gr.Checkbox(label="Use Laplacian Blending", value=False)
                    swap_verbose = gr.Checkbox(label="Verbose Artifacts", value=False)
                swap_target_mask = gr.Image(
                    label="Target Mask (still-image targets only)",
                    type="filepath",
                    height=256,
                    width=256,
                    visible=False,
                )

            with gr.Accordion("Frame Processing", open=False):
                with gr.Row():
                    swap_start_frame = gr.Number(label="Start Frame", value=0, precision=0)
                    swap_frame_step = gr.Number(label="Frame Step", value=1, precision=0)
                    swap_max_frames = gr.Textbox(label="Max Frames", value="", placeholder="Blank = unlimited")

            with gr.Row():
                swap_output_image = gr.Image(
                    label="Face Swap Result",
                    type="filepath",
                    height=400,
                    visible=False,
                )
                swap_output_video = gr.Video(
                    label="Face Swap Video Result",
                    height=400,
                    visible=False,
                    show_download_button=True,
                )

            with gr.Row():
                swap_result_file = gr.File(label="Download Result", visible=False)
                swap_artifacts_file = gr.File(label="Verbose Artifacts", visible=False)
            swap_debug_gallery = gr.Gallery(
                label="Inputs / Masks / Intermediate Views",
                columns=4,
                height=320,
                visible=False,
            )

        with gr.Tab("Face Edit"):
            with gr.Row():
                input_img = gr.Image(label="Input Image", type="filepath", height=400, width=400)
                input_mask = gr.ImageEditor(
                    label="Mask",
                    type="numpy",
                    sources="upload",
                    height=400,
                    width=400,
                    brush=gr.Brush(default_size=24, colors=["#FFFFFF"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=24),
                )

            with gr.Row():
                with gr.Tab("Shape Editing"):
                    region_radio = gr.Radio(COMP, value="hair", label="Facial Regions", info="Which region(s) are you interested in?")
                    shape_edit_logging_text = gr.Textbox(
                        label="Operations Logging",
                        value="Ready to edit shape...",
                        lines=2,
                        interactive=False,
                    )
                    with gr.Row():
                        edit_mask_btn = gr.Button("Confirm Mask Editing")
                        face_shape_edit_btn = gr.Button("Get Edited Face")

                with gr.Tab("Texture Editing"):
                    region_groups = gr.CheckboxGroup(
                        choices=COMP,
                        label="Facial Regions",
                        info="Which region(s) are you interested in?",
                    )
                    with gr.Row():
                        reference_img = gr.Image(label="Reference Image", type="filepath", height=256, width=256)
                        with gr.Column():
                            alpha = gr.Slider(0, 1, value=1.0, label="Editing Extent", info="Choose between 0 and 1")
                            texture_edit_logging_text = gr.Textbox(
                                label="Operations Logging",
                                value="Ready to edit texture...",
                                lines=2,
                                interactive=False,
                            )
                            face_texture_edit_btn = gr.Button("Get Edited Face")

                output_img = gr.Image(label="Result", type="pil", height=400, width=400)

    input_img.change(
        fn=esitimate_init_mask_fn,
        inputs=[input_img],
        outputs=[input_mask, shape_edit_logging_text],
        queue=False,
    )
    reference_img.change(
        fn=esitimate_referece_mask_fn,
        inputs=[reference_img],
        outputs=[texture_edit_logging_text],
        queue=False,
    )
    edit_mask_btn.click(
        fn=edit_mask_fn,
        inputs=[region_radio, input_mask],
        outputs=[input_mask, shape_edit_logging_text],
    )
    face_texture_edit_btn.click(
        fn=face_texture_edit_fn,
        inputs=[region_groups, alpha],
        outputs=[output_img, texture_edit_logging_text],
    )
    face_shape_edit_btn.click(
        fn=face_shape_edit_fn,
        inputs=[input_mask],
        outputs=[output_img, shape_edit_logging_text],
    )

    swap_target_media.change(
        fn=summarize_target_media,
        inputs=[swap_target_media, swap_start_frame, swap_frame_step, swap_max_frames],
        outputs=[swap_media_summary, swap_target_mask],
        queue=False,
    )
    swap_start_frame.change(
        fn=summarize_target_media,
        inputs=[swap_target_media, swap_start_frame, swap_frame_step, swap_max_frames],
        outputs=[swap_media_summary, swap_target_mask],
        queue=False,
    )
    swap_frame_step.change(
        fn=summarize_target_media,
        inputs=[swap_target_media, swap_start_frame, swap_frame_step, swap_max_frames],
        outputs=[swap_media_summary, swap_target_mask],
        queue=False,
    )
    swap_max_frames.change(
        fn=summarize_target_media,
        inputs=[swap_target_media, swap_start_frame, swap_frame_step, swap_max_frames],
        outputs=[swap_media_summary, swap_target_mask],
        queue=False,
    )
    swap_region_preset.change(
        fn=apply_face_swap_region_preset,
        inputs=[swap_region_preset],
        outputs=[swap_shape_regions, swap_texture_regions],
        queue=False,
    )
    swap_run_btn.click(
        fn=run_face_swap,
        inputs=[
            swap_source_img,
            swap_target_media,
            swap_shape_regions,
            swap_texture_regions,
            swap_target_mask,
            swap_lap_bld,
            swap_verbose,
            swap_face_parser,
            swap_start_frame,
            swap_frame_step,
            swap_max_frames,
        ],
        outputs=[
            swap_output_image,
            swap_output_video,
            swap_result_file,
            swap_artifacts_file,
            swap_debug_gallery,
            swap_status,
        ],
    )


if __name__ == "__main__":
    demo.launch()
