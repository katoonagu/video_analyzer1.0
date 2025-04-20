import os
import sys
import glob
from video_analyzer import process_media, parse_timestamps

def get_video_files():
    """Get list of video files in current directory"""
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(ext))
    return sorted(video_files)

def select_videos():
    """Let user select videos to process"""
    print("\nAvailable video files:")
    video_files = get_video_files()
    
    if not video_files:
        print("No video files found in current directory.")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        sys.exit(1)
    
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video}")
    
    print("\nEnter video numbers to process (comma-separated, e.g., 1,3,5)")
    print("Or press Enter to process all videos")
    selection = input("Your choice: ").strip()
    
    if not selection:
        return video_files
    
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        return [video_files[i] for i in indices if 0 <= i < len(video_files)]
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        return select_videos()

def get_timestamps_file(video_file):
    """Get timestamps file for a video"""
    base_name = os.path.splitext(video_file)[0]
    default_timestamps = f"{base_name}_times.txt"
    
    if os.path.exists(default_timestamps):
        print(f"\nFound default timestamps file: {default_timestamps}")
        use_default = input("Use this file? (y/n): ").lower().strip()
        if use_default == 'y':
            return default_timestamps
    
    print(f"\nSelect timestamps file for {video_file}")
    print("Available .txt files:")
    txt_files = glob.glob("*.txt")
    for i, txt in enumerate(txt_files, 1):
        print(f"{i}. {txt}")
    
    while True:
        try:
            choice = input("Enter number or full path of timestamps file: ").strip()
            if os.path.exists(choice):
                return choice
            idx = int(choice) - 1
            if 0 <= idx < len(txt_files):
                return txt_files[idx]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number or file path.")

def process_video_queue():
    """Process multiple videos in sequence"""
    print("=== Video Queue Processor ===")
    print("Select videos to process:")
    
    # Get videos to process
    videos_to_process = select_videos()
    if not videos_to_process:
        print("No videos selected. Exiting.")
        return
    
    print(f"\nSelected {len(videos_to_process)} videos for processing")
    
    # Process each video
    total_videos = len(videos_to_process)
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos_to_process, 1):
        print(f"\n{'='*50}")
        print(f"Processing video {i}/{total_videos}: {video}")
        print(f"{'='*50}")
        
        # Get timestamps file for this video
        timestamps_file = get_timestamps_file(video)
        print(f"Using timestamps file: {timestamps_file}")
        
        try:
            # Parse timestamps
            with open(timestamps_file, 'r', encoding='utf-8') as f:
                segments = parse_timestamps(f.read())
            
            # Process the video
            process_media(video, segments)
            successful += 1
            print(f"✓ Successfully processed {video}")
            
        except Exception as e:
            failed += 1
            print(f"✗ Error processing {video}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("Processing Summary:")
    print(f"Total videos processed: {total_videos}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_video_queue() 