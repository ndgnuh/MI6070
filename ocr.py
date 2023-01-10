from vietocr.tool.config import get_config, list_configs
from vietocr.tool.predictor import Predictor
from vietocr import sample_images
from PIL import Image
from functools import lru_cache
from io import BytesIO
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
from itertools import product
from eyeball.predictor import Predictor as DetectionPredictor
from eyeball.config import read_yaml
import numpy as np
import networkx as nx


def rect_overlap(rec1, rec2):
    inter_x = max(0, min(rec1[2], rec2[2]) - max(rec1[0], rec2[0]))
    inter_y = max(0, min(rec1[3], rec2[3]) - max(rec1[1], rec2[1]))
    return inter_x, inter_y


def image_to_bytes(image):
    io = BytesIO()
    image.save(io, "JPEG")
    return io.getvalue()


@lru_cache
def vietocr_model(config, device):
    available_configs = list_configs()
    assert config in available_configs, available_configs
    config = get_config(config)
    config['device'] = device
    return Predictor(config)


@lru_cache
def detector_model(config):
    return DetectionPredictor.from_config(config)


def detect_text(model, image):
    w, h = image.size
    results = model.predict_single(image)
    # for (x1, y1, x2, y2, score) in tqdm(outputs):
    #     x1 = int(x1 * w)
    #     y1 = int(y1 * h)
    #     x2 = int(x2 * w)
    #     y2 = int(y2 * h)
    #     boxes.append((x1, y1, x2, y2))
    #     scores.append(score)

    # n = len(boxes)
    # should_merge = np.zeros((n, n))
    # for (i, b1), (j, b2) in product(enumerate(boxes), enumerate(boxes)):
    #     if i == j:
    #         continue
    #     inter_x, inter_y = rect_overlap(b1, b2)
    #     h = (b1[3] - b1[1] + b2[3] - b2[1]) // 2
    #     if inter_x > 0 and inter_y > h * 0.5:
    #         should_merge[i, j] = True
    #         should_merge[j, i] = True
    # should_merge = nx.Graph(should_merge)
    # partitions = nx.connected_components(should_merge)

    # final_boxes = []
    # for p in partitions:
    #     x1, y1, x2, y2 = 9999, 9999, 0, 0
    #     for i in p:
    #         x1 = min(x1, boxes[i][0])
    #         y1 = min(y1, boxes[i][1])
    #         x2 = max(x2, boxes[i][2])
    #         y2 = max(y2, boxes[i][3])

    #     final_boxes.append((x1, y1, x2, y2))

    return results


def transcribe_text(model, image, boxes):
    texts = []
    for box in tqdm(boxes):
        text = model(image.crop(box))
        texts.extend(text)
    return texts


def reconstruct(image, texts, boxes):
    w, h = image.size
    output = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(output)
    for (text, box) in zip(texts, boxes):
        print(text, box)
        draw.rectangle(box, outline=(0, 0, 0))
        x1, y1, x2, y2 = box
        fsize = (y2 - y1) * 0.5
        font = ImageFont.truetype(r'fonts/Play-Regular.ttf', size=int(fsize))
        draw.text((box[0], box[1]), text, font=font, fill=(0, 0, 0))
    return output
