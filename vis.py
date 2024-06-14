import argparse
from models.albef.engine import ALBEF
import json

import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        default="images/marinedet/images/crab_Google_0018.jpeg",
        type=str,
    )
    parser.add_argument(
        "--query",
        default="This is a graphical illustration of a blue crab shown against a white background. It has an oval-shaped carapace and blue legs. The tips of the claws appear reddish in color.",
        type=str,
    )
    args = parser.parse_args()

    json_file = "/homes/susan/workspace/david/Documents/GroundVLP/data/class_level_val_seen_grounding_info.json"

    with open(json_file, "r") as file:
        data = json.load(file)

    samples = random.sample(data, 100)

    image_folder = "/homes/susan/workspace/david/Documents/GroundVLP/images/marinedet"

    output_folder = "/homes/susan/workspace/david/Documents/GroundVLP/output"
    # Clear the output folder
    os.system(f"rm -rf {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    engine = ALBEF(model_id="ALBEF", device="cuda", templates="there is a {}")
    for sample in samples:
        image_path = os.path.join(image_folder, sample["file_name"])
        caption = sample["sentences"][0]

        engine.visualize_groundvlp(image_path, query=caption)

        print(f"Image path: {image_path}")
        print(f"Caption: {caption}")
        print("_" * 30)
