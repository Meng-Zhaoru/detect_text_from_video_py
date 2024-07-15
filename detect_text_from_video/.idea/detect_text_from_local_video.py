import io
import os
import cv2

from google.cloud import videointelligence

def video_speech_transcript(video_file_path):
    """Transcribe speech from a video in local."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]
    file_sz_mb = os.path.getsize(video_file_path) / (1024 * 1024)
    timeoutDuration = max(600, 60 * file_sz_mb)

    config = videointelligence.SpeechTranscriptionConfig(
        language_code="zh-HK",
        max_alternatives=0,
    )

    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": video_file_path,
            "video_context": video_context,
        }
    )

    print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=timeoutDuration)

    annotation_results = result.annotation_results[0]
    for speech_transcription in annotation_results.speech_transcriptions:
        print("Transcript: {}".format(speech_transcription.alternatives[0].transcript))

def video_detect_text(video_file_b):
    """Detect text in a binary mode grayscale video file."""
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.TEXT_DETECTION]
    video_context = videointelligence.VideoContext()

    file_size_mb = len(video_file_b) / (1024 * 1024)
    timeoutDuration = max(600, 60 * file_size_mb)

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

    for text_annotation in annotation_result.text_annotations:
        print("\nText: {}.format(text_annotation.text)")

def convert_video_to_grayscale(input_video_path):
    """Convert a video to grayscale. Return the video in binary mode."""
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    gray_frames = []

    # Loop until the video is completely read
    while cap.isOpend():
        print("\nConverting the video to grayscale using OpenCV.")
        # Read a frame from the input video
        ret, frame = cap.read()
        # If no frame is returned, break the loop
        if not ret:
            break
        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLORBGR2GRAY)
        gray_frames.append(gray_frame)

    cap.release() # Realease the input capture object

    # Combine frames into a video_like byte stream for input to the videointelligence API
    # Use imencode function in OpenCV to encode a frame into .mp4 format
    # imencode() returns a tuple: (success of the operation, buffer containing encoded frame).
    _, buffer = cv2.imencode('.mp4', gray_frames[0])
    # Covert the buffer object into a bytes object
    video_file_b = buffer.tobytes()
    # Loop start from the 2nd frame
    for gray_frame in gray_frames[1:]:
        _, buffer = cv2.imencode('.mp4', gray_frame)
        video_file_b += buffer.tobytes()

    return video_file_b


def main():
    input_video_path = input("Please enter the path to the video file to analyze: ")

    gray_video_b = convert_video_to_grayscale(input_video_path)

    video_detect_text(gray_video_b)
    video_speech_transcript(input_video_path)

if __name__ == "__main__":
    main()


