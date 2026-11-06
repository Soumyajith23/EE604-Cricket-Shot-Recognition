
import os
import cv2
import torch
import numpy as np

from torchvision import transforms
from model import ResNetLSTM


# ==================================================
# MODEL CONFIGURATION
# ==================================================

NUM_FRAMES = 16
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# ACTUAL CRICKET SHOT CLASSES
# Extracted from Kaggle training notebook
# ==================================================

CLASS_NAMES = [
    "Cover Drive",      # Class 0
    "Defense",          # Class 1
    "Flick",            # Class 2
    "Hook",             # Class 3
    "Late Cut",         # Class 4
    "Lofted Shot",      # Class 5
    "Pull Shot",        # Class 6
    "Square Cut",       # Class 7
    "Straight Drive",   # Class 8
    "Sweep"             # Class 9
]


# ==================================================
# VIDEO PREPROCESSING
# ==================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# EXTRACT VIDEO FRAMES
# ==================================================

def extract_frames(
    video_path,
    num_frames=NUM_FRAMES
):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open the uploaded video."
        )

    frames = []

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise ValueError(
                "Video contains no readable frames."
            )

        # Uniformly sample 16 frames
        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames
        ).astype(int)

        last_valid_frame = None

        for idx in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            success, frame = cap.read()

            if success:
                last_valid_frame = frame.copy()

            elif last_valid_frame is not None:
                frame = last_valid_frame.copy()

            else:
                raise ValueError(
                    f"Unable to read frame {idx}"
                )

            # BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Apply training preprocessing
            frame = transform(frame)

            frames.append(frame)

    finally:
        cap.release()

    # Shape: [1, 16, 3, 224, 224]
    return torch.stack(
        frames
    ).unsqueeze(0)


# ==================================================
# EXTRACT CHECKPOINT WEIGHTS
# ==================================================

def get_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError(
            "Checkpoint does not contain valid weights."
        )

    # Remove DataParallel prefix
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }

    # Match CNN layer naming
    state = {
        (
            "cnn." + key[len("resnet."):]
            if key.startswith("resnet.")
            else key
        ): value
        for key, value in state.items()
    }

    return state


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

def load_model(
    model_path,
    device=DEVICE,
    class_names=CLASS_NAMES
):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    state = get_state_dict(checkpoint)

    required_keys = [
        "lstm.weight_hh_l0",
        "lstm.weight_ih_l0",
        "fc.weight"
    ]

    for key in required_keys:
        if key not in state:
            raise ValueError(
                f"Missing model parameter: {key}"
            )

    # Detect architecture from saved weights
    hidden_size = int(
        state["lstm.weight_hh_l0"].shape[1]
    )

    num_classes = int(
        state["fc.weight"].shape[0]
    )

    num_layers = sum(
        key.startswith("lstm.weight_ih_l")
        and not key.endswith("_reverse")
        for key in state
    )

    bidirectional = (
        "lstm.weight_ih_l0_reverse" in state
    )

    if len(class_names) != num_classes:
        raise ValueError(
            f"Model has {num_classes} classes, "
            f"but CLASS_NAMES contains {len(class_names)}."
        )

    # Initialize model
    model = ResNetLSTM(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        pretrained=False
    )

    # Load trained weights
    model.load_state_dict(
        state,
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model


# ==================================================
# RESOLVE SHOT NAME
# ==================================================

def resolve_class_name(
    index,
    class_names=CLASS_NAMES
):

    index = int(index)

    if index < 0 or index >= len(class_names):
        raise ValueError(
            f"Invalid predicted class index: {index}"
        )

    name = class_names[index]

    if name is None or not str(name).strip():
        return f"Class {index}"

    return str(name)


# ==================================================
# PREDICT CRICKET SHOT
# ==================================================

@torch.inference_mode()
def predict_video(
    video_path,
    model,
    device=DEVICE,
    class_names=CLASS_NAMES
):

    # Extract frames
    video_tensor = extract_frames(
        video_path,
        NUM_FRAMES
    ).to(device)

    # Run model inference
    outputs = model(video_tensor)

    if outputs.ndim != 2 or outputs.shape[0] != 1:
        raise ValueError(
            f"Unexpected model output shape: "
            f"{tuple(outputs.shape)}"
        )

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0].cpu().numpy()

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Model returned invalid probabilities."
        )

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Number of model outputs does not "
            "match the class names."
        )

    # Find predicted class
    predicted_index = int(
        np.argmax(probabilities)
    )

    # Convert class index to actual shot name
    predicted_shot = resolve_class_name(
        predicted_index,
        class_names
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # Probabilities for every shot
    all_probabilities = {}

    for i, probability in enumerate(probabilities):

        shot_name = resolve_class_name(
            i,
            class_names
        )

        all_probabilities[shot_name] = float(
            probability
        )

    # Sort probabilities
    all_probabilities = dict(
        sorted(
            all_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "class_index": predicted_index,
        "shot": predicted_shot,
        "confidence": confidence,
        "probabilities": all_probabilities
    }
