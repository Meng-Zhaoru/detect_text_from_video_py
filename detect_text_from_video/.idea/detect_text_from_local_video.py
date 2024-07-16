import io
import os
import cv2
import numpy as np

from google.cloud import videointelligence

def preprocess_frame(frame):
    """Preprocess a single grayscale frame to enhance text visibility."""
    # # Apply Gaussian Blur to reduce noise
    # blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Perform morphological operations (optional)
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.erode(binary, kernel, iterations=1)
    # binary = cv2.dilate(binary, kernel, iterations=1)

    return binary

    # # Apply histogram equalization to enhance contrast
    # equalized_frame = cv2.equalizeHist(frame)

    # Apply adaptive thresholding to create a binary image
    # binary_frame = cv2.adaptiveThreshold(equalized_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 11, 2)
    #
    # # Optionally, adjust contrast and brightness
    # alpha = 1.5  # Contrast control (1.0-3.0)
    # beta = 0  # Brightness control (0-100)
    # adjusted_frame = cv2.convertScaleAbs(binary_frame, alpha=alpha, beta=beta)
    #
    # # Perform morphological operations (optional)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # processed_frame = cv2.morphologyEx(adjusted_frame, cv2.MORPH_CLOSE, kernel)
    #
    # return processed_frame

def video_speech_transcript(video_file_path):
    """Transcribe speech from a video in local."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]

    with io.open(video_file_path, "rb") as file:
        input_content = file.read()

    file_sz_mb = os.path.getsize(video_file_path) / (1024 * 1024)
    timeoutDuration = max(600, 60 * file_sz_mb)

    config = videointelligence.SpeechTranscriptionConfig(
        language_code="zh-HK",
        enable_automatic_punctuation=True,
    )

    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": video_context,
        }
    )

    print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=timeoutDuration)

    annotation_results = result.annotation_results[0]
    transcribed_speech = "Transcript:\n"
    for speech_transcription in annotation_results.speech_transcriptions:
        transcribed_speech += speech_transcription.alternatives[0].transcript
    print("\n" + transcribed_speech + ".\n")

def video_detect_text(video_file_path):
    """Detect text in a binary mode grayscale video file."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.TEXT_DETECTION]
    video_context = videointelligence.VideoContext()

    with io.open(video_file_path, "rb") as file:
        input_content = file.read()

    file_sz_mb = os.path.getsize(video_file_path) / (1024 * 1024)
    timeoutDuration = max(600, 60 * file_sz_mb)

    # Get the size of the buffer
    # file_sz_mb = video_file_path.getbuffer().nbytes / (1024 * 1024)
    # timeoutDuration = max(600, 60 * file_sz_mb)

    # file_size_mb = len(video_file_path) / (1024 * 1024)
    # timeoutDuration = max(600, 60 * file_size_mb)

    operation = video_client.annotate_video(
        request = {
            "features": features,
            "input_content": input_content,
            "video_context": video_context,
        }
    )

    print("\nProcessing video for text detection.")
    result = operation.result(timeout=timeoutDuration)

    # The first result is retrieved becasue a single video was processed.
    annotation_result = result.annotation_results[0]
    # print(len(result.annotation_results))

    text_recognized = ""
    for text_annotation in annotation_result.text_annotations:
        text_recognized += text_annotation.text

    print("Text:\n" + text_recognized + "\n")


def preprocess_video(input_video_path, output_folder):
    """Covert a video to grayscale and save it."""
    print("\nConverting the video to grayscale and setup threshold.")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(input_video_path)

    # Extract the video file name and create the output path
    video_name = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_folder, f"preprocessed_{video_name}")

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to write the grayscale video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    # Loop until the video is completely read
    while cap.isOpened():
        # Read a frame from the input video
        ret, frame = cap.read()
        # If no frame is returned, break the loop
        if not ret:
            break
        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Preprocess the frame to enhance text visibility
        # Apply adaptive thresholding to create a binary image
        processed_frame = preprocess_frame(gray_frame)
        # Write the grayscale frame to the output video
        out.write(processed_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"\nPreprocessed video saved to {output_video_path}")
    return output_video_path


def main():
    input_video_path = input("Please enter the path to the video file to analyze:\n")
    output_folder = input("Please enter the folder where you want to save the converted grayscale video files:\n")
    #
    # convert_video_to_grayscale_and_save(input_video_path, output_folder)
    #
    # gray_video_b = convert_video_to_grayscale(input_video_path)

    preprocessed_video = preprocess_video(input_video_path, output_folder)
    video_detect_text(preprocessed_video)
    video_speech_transcript(input_video_path)

if __name__ == "__main__":
    main()


