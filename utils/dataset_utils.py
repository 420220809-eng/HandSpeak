import os
import pandas as pd
from tqdm import tqdm
from utils.landmark_utils import save_landmarks_from_video
import config


def load_dataset():
    """
    Load dataset and extract landmarks from new videos
    
    Returns:
        List of video names that have been processed
    """
    videos = []
    # Support multiple video formats
    video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
    
    for root, dirs, files in os.walk(config.VIDEOS_PATH):
        for file_name in files:
            # Check if file has a video extension
            if any(file_name.endswith(ext) for ext in video_extensions):
                # Get sign name from parent directory
                sign_name = os.path.basename(root)
                # Remove extension to get video name
                video_name = os.path.splitext(file_name)[0]
                videos.append((sign_name, video_name, file_name))
    
    dataset = []
    for root, dirs, files in os.walk(config.DATASET_PATH):
        for file_name in files:
            if file_name.endswith(".pickle") and file_name.startswith("pose_"):
                video_name = file_name.replace(".pickle", "").replace("pose_", "")
                dataset.append(video_name)

    # Create the dataset from the reference videos
    videos_not_in_dataset = [v for v in videos if v[1] not in dataset]
    n = len(videos_not_in_dataset)
    
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")
        
        # Track quality reports
        quality_reports = []

        for idx in tqdm(range(n)):
            report = save_landmarks_from_video(videos_not_in_dataset[idx])
            quality_reports.append(report)
        
        # Save quality report to CSV
        if quality_reports:
            import pandas as pd
            df_report = pd.DataFrame(quality_reports)
            report_path = os.path.join(config.DATASET_PATH, "video_quality_report.csv")
            
            # Append to existing report if it exists
            if os.path.exists(report_path):
                existing_df = pd.read_csv(report_path)
                df_report = pd.concat([existing_df, df_report], ignore_index=True)
            
            df_report.to_csv(report_path, index=False)
            print(f"\n{'='*60}")
            print(f"Quality Report saved to: {report_path}")
            print(f"{'='*60}")
            
            # Print summary
            print("\nQuality Summary:")
            print(f"Total videos processed: {len(quality_reports)}")
            print(f"OK: {len(df_report[df_report['status'] == 'OK'])}")
            print(f"FAIR: {len(df_report[df_report['status'] == 'FAIR'])}")
            print(f"POOR: {len(df_report[df_report['status'] == 'POOR'])}")
            print(f"BAD: {len(df_report[df_report['status'] == 'BAD'])}")
            print(f"ERROR: {len(df_report[df_report['status'] == 'ERROR'])}")
            
            # Show problematic videos
            problematic = df_report[df_report['status'].isin(['BAD', 'ERROR', 'POOR'])]
            if len(problematic) > 0:
                print(f"\n⚠ {len(problematic)} problematic videos found:")
                print("\nBy sign class:")
                print(problematic.groupby('sign_name')['status'].value_counts())

    return [v[1] for v in videos]
