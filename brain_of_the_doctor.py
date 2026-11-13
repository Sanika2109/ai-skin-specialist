# Step 1: Install and import dependencies

import base64
import os
import mimetypes
import re

import cv2
from dotenv import load_dotenv
from groq import Groq


# Step 2: Load environment variables

folder = os.path.dirname(__file__)
env_path = os.path.join(folder, ".env")

load_dotenv(env_path)


# Step 3: Helper functions

def encode_file(filepath):
    """Encode a file as base64."""
    with open(filepath, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_media_type(filepath, fallback):
    """Get MIME type of a file."""
    media_type, _ = mimetypes.guess_type(filepath)
    return media_type or fallback


def encode_frame(frame):
    """Convert an OpenCV frame to a base64 JPEG string."""

    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Could not encode video frame.")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# Step 4: Extract representative frames from video

def extract_video_frames(video_filepath, number_of_frames=5):

    cap = cv2.VideoCapture(video_filepath)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_filepath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    # If video has fewer frames than requested
    number_of_frames = min(number_of_frames, total_frames)

    # Select frames throughout the entire video
    frame_indices = [
        int(i * (total_frames - 1) / (number_of_frames - 1))
        if number_of_frames > 1
        else 0
        for i in range(number_of_frames)
    ]

    frames = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if success:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Could not extract any frames from the video.")

    return frames


# Step 5: Brain of the Doctor

def brain_of_the_doctor(
    patient_text,
    image_filepath=None,
    video_filepath=None
):

    # Get Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "Missing GROQ_API_KEY in .env or environment."
        )

    # Groq vision model
    model = os.environ.get(
        "GROQ_MODEL",
        "qwen/qwen3.6-27b"
    )

    # Create Groq client
    client = Groq(api_key=groq_api_key)


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    if not patient_text:
        patient_text = "The patient provided media without additional text description."

    prompt = (
        "You are a confident and helpful doctor specializing in "
        "general skin care information. "
        "Speak naturally, clearly, and reassuringly. "
        "Limit your entire response to two or three sentences maximum. "
        "Do not provide a definitive medical diagnosis. "
        "Describe only what is visibly observable and provide general "
        "skin care guidance. "
        "If the patient has provided only an image and no video, "
        "check if the image is sufficient for analysis. "
        "If not, ask them to provide a video first because "
        "more visual details may be needed. "
        "Do not use special characters, symbols, asterisks, "
        "or markdown formatting because the response will be "
        "converted directly to audio.\n\n"
        f"Patient text: {patient_text}"
    )


    # ==================================================
    # CASE 1: VIDEO
    # ==================================================

    if video_filepath:

        print("Processing video...")

        frames = extract_video_frames(
            video_filepath,
            number_of_frames=5
        )

        content = [
            {
                "type": "text",
                "text": (
                    "These images are consecutive representative frames "
                    "from a skin-related video. "
                    "Analyze the visible skin features across the frames. "
                    "Pay attention to changes, texture, redness, swelling, "
                    "spots, irritation, or other visible features. "
                    "Give general skincare information and do not provide "
                    "a definitive diagnosis.\n\n"
                    + prompt
                ),
            }
        ]

        # Add extracted frames
        for frame in frames:

            image_data = encode_frame(frame)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:image/jpeg;base64,{image_data}"
                        )
                    },
                }
            )


        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1000,
            extra_body={"reasoning_format": "hidden"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical assistant "
                        "specializing in general skincare information. "
                        "Never provide a definitive diagnosis."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )


    # ==================================================
    # CASE 2: IMAGE
    # ==================================================

    elif image_filepath:

        print("Processing image...")

        image_data = encode_file(image_filepath)

        media_type = get_media_type(
            image_filepath,
            "image/jpeg"
        )

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1000,
            extra_body={"reasoning_format": "hidden"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical assistant "
                        "specializing in general skincare information. "
                        "Never provide a definitive diagnosis."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{media_type};base64,"
                                    f"{image_data}"
                                ),
                            },
                        },
                    ],
                },
            ],
        )


    # ==================================================
    # CASE 3: TEXT ONLY
    # ==================================================

    else:

        print("Processing text...")

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1000,
            extra_body={"reasoning_format": "hidden"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical assistant "
                        "specializing in general skincare information. "
                        "Never provide a definitive diagnosis."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )


    # Step 6: Get doctor's response

    doctors_response_in_text = (
        response.choices[0].message.content or ""
    )

    # Strip internal reasoning blocks if present (<think>...</think> or unclosed <think>...)
    if "<think>" in doctors_response_in_text:
        if "</think>" in doctors_response_in_text:
            doctors_response_in_text = re.sub(
                r'<think>.*?</think>',
                '',
                doctors_response_in_text,
                flags=re.DOTALL
            ).strip()
        else:
            doctors_response_in_text = re.sub(
                r'<think>.*$',
                '',
                doctors_response_in_text,
                flags=re.DOTALL
            ).strip()

    return doctors_response_in_text


# ======================================================
# Step 7: Test the function
# ======================================================

if __name__ == "__main__":

    sample_image = os.path.join(
        folder,
        "sample_image.jpeg"
    )

    sample_video = os.path.join(
        folder,
        "test-video.mp4"
    )

    patient_text = (
        "I have noticed this skin problem recently. "
        "Can you tell me what I should do?"
    )


    # ----------------------------------------------
    # IMAGE TEST
    # ----------------------------------------------

    if os.path.exists(sample_image):

        result = brain_of_the_doctor(
            patient_text=patient_text,
            image_filepath=sample_image
        )

        print("\nDoctor response:")
        print(result)


    # ----------------------------------------------
    # VIDEO TEST
    # ----------------------------------------------

    elif os.path.exists(sample_video):

        result = brain_of_the_doctor(
            patient_text=patient_text,
            video_filepath=sample_video
        )

        print("\nDoctor response:")
        print(result)


    else:

        result = brain_of_the_doctor(
            patient_text=patient_text
        )

        print("\nDoctor response:")
        print(result)