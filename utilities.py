import torch, torchvision

import torchvision.transforms as T

# standard PyTorch mean-std input image normalization
transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


test = []


def filter_bboxes_from_outputs(im, outputs, threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs["logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], im.size)
    print(bboxes_scaled[0])
    for p, (xmin, ymin, xmax, ymax) in zip(probas_to_keep, bboxes_scaled.tolist()):
        cl = p.argmax()
        test.append(
            [
                CLASSES[cl],
                xmin,
                ymin,
                xmax - xmin,
                ymax - ymin,
                change_origin(im, xmin, xmax, ymin, ymax),
            ]
        )
    print(test)
    # return probas_to_keep, bboxes_scaled
    return test


CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


import matplotlib.pyplot as plt


def change_origin(im, xmin, xmax, ymin, ymax):
    width, height = im.size
    print(width,height)
    origin_x = width // 2
    origin_y = -height // 2

    center_x = int((xmax + xmin) // 2)
    center_y = int(-(ymax + ymin) // 2)

    center_x = center_x - origin_x
    center_y = center_y - origin_y

    return [center_x, center_y]


test_origin = []


def plot_results(pil_img, prob=None, boxes=None):
    global test, test_origin
    # plt.figure(figsize=(16, 10))
    # plt.imshow(pil_img)
    # ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            # ax.add_patch(
            #     plt.Rectangle(
            #         (xmin, ymin),
            #         xmax - xmin,
            #         ymax - ymin,
            #         fill=False,
            #         color=c,
            #         linewidth=3,
            #     )
            # )
            cl = p.argmax()
            test.append(
                [
                    CLASSES[cl],
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    change_origin(xmin, xmax, ymin, ymax),
                ]
            )
            # test_origin.append([change_origin(xmin,xmax,ymin,ymax)])
            text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
            # ax.text(
            #     xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5)
            # )
    # plt.axis("off")
    # plt.show()
    
def get_image_region(point):
    x, y = point  # Extract x and y coordinates from the point
    
    # Define the dimensions and boundaries of each region
    regions = {
        'Top-Left': (-960, 180, -280, 920),
        'Top-Center': (-280, 180, 280, 920),
        'Top-Right': (280, 180, 960, 920),
        'Left': (-960, -180, -280, 180),
        'Center': (-280, -180, 280, 180),
        'Right': (280, -180, 960, 180),
        'Bottom-Left': (-960, -920, -280, -180),
        'Bottom-Center': (-280, -920, 280, -180),
        'Bottom-Right': (280, -920, 960, -180)
    }
    
    # Iterate through the regions and check if the point falls within any of them
    for region, (x1, y1, x2, y2) in regions.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return region  # Return the region if the point falls within it
    
    return 'Unknown'  # Return 'Unknown' if the point is outside all regions or if the input is invalid