from src.options.swap_options import SwapFacePipelineOptions
from src.pipelines.face_swap import run_cli_face_swap


if __name__ == "__main__":
    options = SwapFacePipelineOptions().parse()
    run_cli_face_swap(options)
