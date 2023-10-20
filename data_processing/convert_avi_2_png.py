import cv2
import os


def avi_to_png_frames(input_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get information about the video (frame width, height, and frames per second)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as a PNG image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Frames extracted: {frame_count}")


if __name__ == "__main__":

    seq_no = 7
    seq_no = str(seq_no)
    input_video_path = f"/media/atr17/HDD Storage/US_axial_data_shallower_V2/Catheter-{seq_no}/Catheter-{seq_no}.avi"  # Change this to your input video file path
    output_folder_path = f"/media/atr17/HDD Storage/US_axial_data_shallower_V2/Images/Val/Catheter/Catheter-{seq_no}"  # Change this to the desired output folder path
    avi_to_png_frames(input_video_path, output_folder_path)