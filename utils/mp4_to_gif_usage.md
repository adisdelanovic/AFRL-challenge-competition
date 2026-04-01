# Usage

```
usage: mp4_to_gif.py [-h] [-f INPUT_FILE | -d INPUT_DIR] [-e EPISODES [EPISODES ...]] [-o OUTPUT]

Convert mp4 episodes to gif

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --file INPUT_FILE
                        Path to a single MP4 file to convert.
  -d INPUT_DIR, --dir INPUT_DIR
                        Path to a directory containing MP4 files.
  -e EPISODES [EPISODES ...], --episodes EPISODES [EPISODES ...]
                        A list of episode numbers to include from the directory.
  -o OUTPUT, --output OUTPUT
                        Name of the output GIF file (default: afrl_challenge.gif).
```

# Example usage

## stitch multiple epsidoes together
Creates gif with episodes 4, 7, 2 in that order
```
python utils/mp4_to_gif.py -e 0 2 7
Loading video clips...
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-4.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-7.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-2.mp4
MoviePy - Building file afrl_challenge.gif with imageio.
Successfully converted video to 'afrl_challenge.gif'                                                                                                                                                               
Cleaning up resources...
```


## stitch together all episodes that start with prefix
Gets all episodes that start with test
```
python utils/mp4_to_gif.py -e test
Loading video clips...
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-0.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-3.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-6.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-9.mp4
MoviePy - Building file afrl_challenge.gif with imageio.
Successfully converted video to 'afrl_challenge.gif'                                                                                                                                                               
Cleaning up resources...
```

## convert one mp4 file to gif
```
python utils/mp4_to_gif.py -f videos/eval-episode-2.mp4
Loading video clips...
  + Adding videos/eval-episode-2.mp4
MoviePy - Building file afrl_challenge.gif with imageio.
Successfully converted video to 'afrl_challenge.gif'                                                                                                                                                               
Cleaning up resources...
```

## specify different input video dir
```
python utils/mp4_to_gif.py -d my_videos_dir/
Loading video clips...
  + Adding /home/kitchen-lipskiz/code/cca-environment/my_videos_dir/eval-episode-2.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/my_videos_dir/eval-episode-5.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/my_videos_dir/eval-episode-7.mp4
MoviePy - Building file afrl_challenge.gif with imageio.
Successfully converted video to 'afrl_challenge.gif'                                                                                                                                                               
Cleaning up resources...
```

## specify prefixes and output gif filename
Gets all test and eval episodes with those prefixes and gets episodes 0, 2, 3 if they exist (notice there is no test episode 2)
```
python utils/mp4_to_gif.py -e test eval 0 2 3 -o my_gif.gif
Loading video clips...
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-0.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-0.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-2.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/eval-episode-3.mp4
  + Adding /home/kitchen-lipskiz/code/cca-environment/videos/test-episode-3.mp4
MoviePy - Building file my_gif.gif with imageio.
Cleaning up resources...
```