import argparse
from pet_breathy.pet_breathy_runner import PetBreathyRunner

import random

def _main():
    args = _parse_args()
    run(args.in_video_path)

def _parse_args():
    parser = argparse.ArgumentParser(description='Pet Breathy')
    
    parser.add_argument(
        '--in-video-path',
        help='File path for input video to analyze')
    
    return parser.parse_args()

def run(in_video_path: str):
    # For debug, use same seed
    random.seed(a=0)

    runner = PetBreathyRunner.create_from_video_file(in_video_path)
    runner.run()

if __name__ == '__main__':
    _main()
