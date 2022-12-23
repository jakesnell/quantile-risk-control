import json
import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()

batch = None
classes = None

COCO_LABEL_JSON_FILENAME = "data/coco_2014/val_2k/labels.json"


class CocoLoader:
    def __init__(self):
        with open(COCO_LABEL_JSON_FILENAME, "r") as f:
            self.label_data = json.load(f)

        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> dict[int, list[int]]:
        annos = self.label_data["annotations"]

        labels: dict[int, set[int]] = {
            image_id: set() for (image_id, _) in self.images()
        }

        for (image_id, category_id) in map(
            lambda anno: (anno["image_id"], anno["category_id"]), annos
        ):
            labels[image_id].add(category_id)

        return {
            image_id: list(category_ids) for (image_id, category_ids) in labels.items()
        }

    def num_classes(self) -> int:
        return len(self.categories())

    def categories(self) -> list[tuple[int, str]]:
        return [
            (category["id"], category["name"])
            for category in self.label_data["categories"]
            if category["supercategory"] is not None
        ]

    def images(self) -> list[tuple[int, str]]:
        return [
            (image["id"], image["file_name"]) for image in self.label_data["images"]
        ]

    def category_mapping(self) -> dict[int, int]:
        return {category_id: i for i, (category_id, _) in enumerate(self.categories())}

    @typechecked
    def one_hot_labels(
        self, image_ids: TensorType["batch", int]
    ) -> TensorType["batch", "classes", int]:
        num_examples = image_ids.size(0)
        labels = torch.zeros(num_examples, self.num_classes(), dtype=torch.long)
        cat_map = self.category_mapping()

        for i, image_id in enumerate(image_ids.tolist()):
            for category_id in self.ground_truth[image_id]:
                labels[i][cat_map[category_id]] = 1

        return labels
