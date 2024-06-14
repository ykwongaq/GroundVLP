import json
import argparse


def read_json(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json(data, json_file: str):
    with open(json_file, "w") as f:
        json.dump(data, f)


def main(json_file: str):

    json_data = read_json(json_file)

    ID_TO_CATEGORY = {}
    for category in json_data["categories"]:
        ID_TO_CATEGORY[category["id"]] = category["name"]

    ID_TO_IMAGE = {}
    for image in json_data["images"]:
        ID_TO_IMAGE[image["id"]] = image["file_name"]

    formatted_jsons = []

    # Target format: {"file_name": "COCO_train2014_000000581563.jpg", "sentences": ["lower left corner darkness", "bpttom left dark", "black van in front of cab"], "gt_bbox": [0.0, 373.89, 137.59, 126.11], "category": "car"}
    for annotation in json_data["annotations"]:
        formatted_json = {}
        formatted_json["file_name"] = ID_TO_IMAGE[annotation["image_id"]]
        formatted_json["sentences"] = [annotation["caption"]]
        formatted_json["gt_bbox"] = annotation["bbox"]
        formatted_json["category"] = ID_TO_CATEGORY[annotation["category_id"]]
        formatted_jsons.append(formatted_json)

    output_file = json_file.replace(".json", "_info.json")
    print(f"Writing to {output_file}")
    write_json(formatted_jsons, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True, type=str)
    args = parser.parse_args()
    main(args.json_file)
