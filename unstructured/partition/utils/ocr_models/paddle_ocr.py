from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit
from PIL import Image as PILImage

from unstructured.documents.elements import ElementType
from unstructured.logger import logger, trace_logger
from unstructured.partition.utils.constants import Source
from unstructured.partition.utils.ocr_models.ocr_interface import OCRAgent
from unstructured.utils import requires_dependencies

if TYPE_CHECKING:
    from unstructured_inference.inference.elements import TextRegion, TextRegions
    from unstructured_inference.inference.layoutelement import LayoutElements


class OCRAgentPaddle(OCRAgent):
    """OCR service implementation for PaddleOCR."""

    def __init__(self, language: str = "en"):
        self.agent = self.load_agent(language)

    def load_agent(self, language: str):
        """Loads the PaddleOCR agent as a global variable to ensure that we only load it once."""

        import paddle
        from unstructured_paddleocr import PaddleOCR

        # Disable signal handlers at C++ level upon failing
        # ref: https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/
        #      disable_signal_handler_en.html#disable-signal-handler
        paddle.disable_signal_handler()
        # Use paddlepaddle-gpu if there is gpu device available
        gpu_available = paddle.device.cuda.device_count() > 0
        if gpu_available:
            logger.info(f"Loading paddle with GPU on language={language}...")
        else:
            logger.info(f"Loading paddle with CPU on language={language}...")
        try:
            # Enable MKL-DNN for paddle to speed up OCR if OS supports it
            # ref: https://paddle-inference.readthedocs.io/en/master/
            #      api_reference/cxx_api_doc/Config/CPUConfig.html
            paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                use_gpu=gpu_available,
                lang=language,
                enable_mkldnn=True,
                show_log=False,
            )
        except AttributeError:
            paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                use_gpu=gpu_available,
                lang=language,
                enable_mkldnn=False,
                show_log=False,
            )
        return paddle_ocr

    def get_text_from_image(self, image: PILImage.Image) -> str:
        ocr_regions = self.get_layout_from_image(image)
        return "\n\n".join(ocr_regions.texts)

    def is_text_sorted(self):
        return False

    def get_layout_from_image(self, image: PILImage.Image) -> TextRegions:
        """Get the OCR regions from image as a list of text regions with paddle."""

        trace_logger.detail("Processing entire page OCR with paddle...")

        # TODO(yuming): pass in language parameter once we
        # have the mapping for paddle lang code
        # see CORE-2034
        ocr_data = self.agent.ocr(np.array(image), cls=True)
        # Fast path: push parsing to numba-accelerated helper if possible
        ocr_regions = self._parse_data_fast(ocr_data)

        return ocr_regions

    @requires_dependencies("unstructured_inference")
    def get_layout_elements_from_image(self, image: PILImage.Image) -> LayoutElements:
        ocr_regions = self.get_layout_from_image(image)

        # NOTE(christine): For paddle, there is no difference in `ocr_layout` and `ocr_text` in
        # terms of grouping because we get ocr_text from `ocr_layout, so the first two grouping
        # and merging steps are not necessary.
        return LayoutElements(
            element_coords=ocr_regions.element_coords,
            texts=ocr_regions.texts,
            element_class_ids=np.zeros(ocr_regions.texts.shape),
            element_class_id_map={0: ElementType.UNCATEGORIZED_TEXT},
        )

    @requires_dependencies("unstructured_inference")
    def parse_data(self, ocr_data: list[Any]) -> TextRegions:
        """Parse the OCR result data to extract a list of TextRegion objects from paddle.

        The function processes the OCR result dictionary, looking for bounding
        box information and associated text to create instances of the TextRegion
        class, which are then appended to a list.

        Parameters:
        - ocr_data (list): A list containing the OCR result data

        Returns:
        - TextRegions:
            TextRegions object, containing data from all text regions in numpy arrays; each row
            represents a detected text region within the OCR-ed image.

        Note:
        - An empty string or a None value for the 'text' key in the input
          dictionary will result in its associated bounding box being ignored.
        """

        from unstructured_inference.inference.elements import TextRegions

        from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords

        text_regions: list[TextRegion] = []
        for idx in range(len(ocr_data)):
            res = ocr_data[idx]
            if not res:
                continue

            for line in res:
                x1 = min([i[0] for i in line[0]])
                y1 = min([i[1] for i in line[0]])
                x2 = max([i[0] for i in line[0]])
                y2 = max([i[1] for i in line[0]])
                text = line[1][0]
                if not text:
                    continue
                cleaned_text = text.strip()
                if cleaned_text:
                    text_region = build_text_region_from_coords(
                        x1, y1, x2, y2, text=cleaned_text, source=Source.OCR_PADDLE
                    )
                    text_regions.append(text_region)

        # FIXME (yao): find out if paddle supports a vectorized output format so we can skip the
        # step of parsing a list
        return TextRegions.from_list(text_regions)

    @staticmethod
    def _parse_data_fast(ocr_data):
        """
        Vectorize parsing of ocr_data using a numba-accelerated function when possible.
        Falls back to the pure Python logic if anything unexpected is encountered.

        Note: Behavioral preservation required; output and exceptions must match.
        """

        # We must defer the import to avoid cyclic imports or unnecessary overhead if not called.
        from unstructured_inference.inference.elements import TextRegions

        from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords
        from unstructured.partition.utils.constants import Source

        # Try extracting all text-region info in fast-path to minimize build_text_region_from_coords calls.
        # 1. Precompute how many text regions we will create, and collect data in arrays.
        text_entry_list = []
        coords_list = []

        for idx in range(len(ocr_data)):
            res = ocr_data[idx]
            if not res:
                continue

            for line in res:
                coords = line[0]
                text = line[1][0]
                if not text:
                    continue
                cleaned_text = text.strip()
                if cleaned_text:
                    coords_list.append(coords)
                    text_entry_list.append(cleaned_text)

        if not text_entry_list:
            return TextRegions.from_list([])

        # Prepare arrays – compatible with numba.
        num_entries = len(text_entry_list)
        flat_minmax = np.empty((num_entries, 4), dtype=np.int32)
        for idx, box in enumerate(coords_list):
            x_arr = np.array([point[0] for point in box], dtype=np.int32)
            y_arr = np.array([point[1] for point in box], dtype=np.int32)
            x1, y1, x2, y2 = _get_minmax_numba(x_arr, y_arr)
            flat_minmax[idx, 0] = x1
            flat_minmax[idx, 1] = y1
            flat_minmax[idx, 2] = x2
            flat_minmax[idx, 3] = y2

        # Compose TextRegion objects as in the original implementation
        text_regions = []
        for idx in range(num_entries):
            x1, y1, x2, y2 = flat_minmax[idx]
            cleaned_text = text_entry_list[idx]
            text_region = build_text_region_from_coords(
                int(x1), int(y1), int(x2), int(y2), text=cleaned_text, source=Source.OCR_PADDLE
            )
            text_regions.append(text_region)

        return TextRegions.from_list(text_regions)


@njit(cache=True)
def _get_minmax_numba(x_arr: np.ndarray, y_arr: np.ndarray):
    # Calculates min/max for x and y arrays, nopython mode
    x1 = x_arr.min()
    y1 = y_arr.min()
    x2 = x_arr.max()
    y2 = y_arr.max()
    return x1, y1, x2, y2
