import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response
from utilities import transform, filter_bboxes_from_outputs, get_image_region
from transformers import (
    DetrForObjectDetection,
    DPTForDepthEstimation,
    DPTFeatureExtractor,
)
from PIL import Image

app = Flask(__name__)
detrmodel = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
dptmodel = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
detrmodel.eval()
dptmodel.eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    r = request
    file = request.files["data"]
    # Read the image via file.stream
    im = Image.open(file.stream)
    img = transform(im).unsqueeze(0)

    outputs = detrmodel(img)
    final_outputs = filter_bboxes_from_outputs(im, outputs, threshold=0.95)
    # plot_results(im, probas_to_keep, bboxes_scaled)

    object_list = [
        [int(item) if isinstance(item, float) else item for item in sublist]
        for sublist in final_outputs
    ]

    pixel_values = feature_extractor(im, return_tensors="pt").pixel_values
    print("WEWEFDV")
    with torch.no_grad():
        outputs = dptmodel(pixel_values)
        predicted_depth = outputs.predicted_depth
    print("Hi")
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=im.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    print("Hello!")
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_image = formatted  # cv2.imread('/content/Depth Image', cv2.IMREAD_GRAYSCALE)
    normal_image = im
    # Perform object detection or segmentation on the normal image to obtain object bounding boxes and names
    # and store them in a list called 'objects' with elements of the form: (object_name, x, y, w, h)
    objects = object_list
    # Calculate average depth value for each object
    objects_with_depth = []
    for idx, obj in enumerate(objects):
        (
            object_name,
            x,
            y,
            w,
            h,
            origin,
        ) = obj  # Object bounding box coordinates and name
        depth_values = depth_image[y : y + h, x : x + w]
        average_depth = np.mean(depth_values)
        objects_with_depth.append(([object_name, idx, origin], average_depth))

    # Sort objects based on depth (lower depth values have higher priority)
    objects_with_depth.sort(key=lambda x: x[1])

    # Create a priority list of objects with object names and initial indices
    priority_list = [obj[0] for obj in objects_with_depth]
    priority_list = sorted(priority_list, key=lambda x: x[1])
    print(priority_list)
    return Response("success", status=200)


@app.route("/results", methods=["POST"])
def results():
    data = request.get_json(force=True)
    prediction = detrmodel.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


app.run(debug=True)
