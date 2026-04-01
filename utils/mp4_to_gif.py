from moviepy import VideoFileClip, concatenate_videoclips
from pathlib import Path
import os
import argparse

def convert_mp4_to_gif(mp4s, gif_path):
    """
    Converts a video file (MP4) to a GIF.

    Args:
        mp4_path (str): The full path to the input MP4 file.
        gif_path (str): The full path where the output GIF will be saved.
    """
    clips = []

    
    try:
        print("Loading video clips...")
        for mp4_path in mp4s:
            print(f"  + Adding {mp4_path}")
            # Check if the input file exists
            if os.path.exists(mp4_path):
                # Load the video clip
                clip = VideoFileClip(mp4_path)
                clips.append(clip)
            else:
                print(f"  - Warning: File not found, skipping: {mp4_path}")

        if not clips:
            print("No valid video files were found. GIF creation aborted.")
            return
        

        final_clip = concatenate_videoclips(clips)

        # Write the GIF file
        final_clip.write_gif(gif_path)
        
        # Close the clip to release resources
        final_clip.close()
        
        print(f"Successfully converted video to '{gif_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- Close all the clips to release resources ---
        # This is important to avoid memory leaks
        print("Cleaning up resources...")
        for clip in clips:
            clip.close()


def get_mp4_files(directory_path):
    """
    Returns a list of all .mp4 files in the given directory.
    
    Args:
        directory_path (str or Path): Path to the directory.
    
    Returns:
        list[Path]: List of Path objects for .mp4 files.
    """
    try:
        dir_path = Path(directory_path).expanduser().resolve()

        # Validate directory
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Invalid directory: {dir_path}")

        # Get all .mp4 files (case-insensitive)
        mp4_files = [str(file) for file in dir_path.glob("*.mp4")] + \
                    [str(file) for file in dir_path.glob("*.MP4")]

        return sorted(mp4_files)

    except Exception as e:
        print(f"Error: {e}")
        return []

# --- USAGE EXAMPLE ---
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Convert mp4 episodes to gif")
    
    # Group for mutually exclusive inputs: either a single file or a directory
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("-f", "--file", dest="input_file",
                             help="Path to a single MP4 file to convert.")
    input_group.add_argument("-d", "--dir", dest="input_dir", default="videos/",
                             help="Path to a directory containing MP4 files.")

    # Optional argument for specifying episode numbers
    parser.add_argument("-e", "--episodes", nargs='+', type=str,
                        help="A list of episode numbers to include from the directory.")

    # Argument for the output file name
    parser.add_argument("-o", "--output", default="afrl_challenge.gif",
                        help="Name of the output GIF file (default: afrl_challenge.gif).")

    args = parser.parse_args()

    filtered_files = []
    if args.input_file:
        filtered_files = [args.input_file]

    else:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Directory not found at '{args.input_dir}'")
            exit()

        files = get_mp4_files(args.input_dir)

        if args.episodes:

            prefixes = [ep for ep in args.episodes if not ep.isdigit()]
            ep_nums = [ep for ep in args.episodes if ep.isdigit()]

            # filters for provides episode numbers
            if ep_nums:
                for ep in ep_nums:
                    for file in files:
                        if file.endswith(f"-{ep}.mp4") and file not in filtered_files:
                            
                            # added filter if prefix is included in args
                            if prefixes:
                                for prefix in prefixes:
                                    if os.path.split(file)[1].startswith(prefix):
                                        filtered_files.append(file)
                            else:
                                filtered_files.append(file)

            # just filters on prefix if no numbers are provided
            elif prefixes:
                for prefix in prefixes:
                    for file in files:
                        if os.path.split(file)[1].startswith(prefix) and file not in filtered_files:
                            filtered_files.append(file)
        
        # used if no episodes are provided (uses whole input dir)
        else:
            filtered_files = files


    convert_mp4_to_gif(filtered_files, args.output)