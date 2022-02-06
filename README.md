# video-compression

Here are the steps to run our code. We assume you start from a directory called $DIR, and your working machine contains an NVIDIA GPU.

## Build ffmpeg from source

Alex digged into the source code of ffmpeg and H.264 codec to support RoI encoding with an encoding quality (QP parameter in the H.264 codec) matrix file as the input. To build ffmpeg from source, please 
```
git clone https://github.com/Alex-q-z/myh264.git
```
under $DIR
and checkout to AccMPEG branch
```
git checkout AccMPEG
```

_(Note: this branch does not support parallel encoding. Please use the master branch if you want to use RoI encoding for other purposes.)_

After that, ```cd``` into the repo and run ```build.sh```. It takes time to compile. If you compiled the code successfully, you should see something like
```
INSTALL libavutil/libavutil.pc
```
at the end.

Then, inside x264/encoder/encoder.c, search for ```/tank```, and you'll see
```C++
    // Qizheng: add qp matrix file here
    h->operation_mode_file = x264_fopen("/tank/kuntai/code/operation_mode_file", "r");
    fscanf(h->operation_mode_file, "%d,", &h->operation_mode_flag);
    h->qp_matrix_file = x264_fopen("/tank/kuntai/code/qp_matrix_file", "r");
```
change the two hard-coded paths (/tank/kuntai/code/...) to $DIR/myh264/... (the path must be absolute path), and rerun build.sh (the compilation will take much less time this time, don't worry.)

## Run AccMPEG on one video

First, git clone our repo under $DIR and cd into our repo.

Then, install the conda environment through the ```conda_env.yml```:
```conda env create -f conda_env.yml```
_(If your CUDA version < 11.1, you may need to uninstall 3 packages (pytorch, torchvision and detectron2), and re-install other versions that are compatible with your CUDA version. Any pytorch version > 1.7 should work. I recommand re-install these three packages completely from pip, as the conda install torchvision will install ancient version of torchvision and ffmpeg that triggers non-intuitive bugs.)_

After that, please install a version of ffmpeg that supports -qp parameter (in our server it is version 4.2.1) (we will only use the modified version of ffmpeg in AccMPEG, not in baselines.)

Then go back to our repo, run
```
conda activate diff
```
to activate the conda environment, ```cd artifact``` and run ```extract.py``` to extract the video ```dashcamcropped_1.mp4``` to pngs. 

Then, ```cd ..``` and open ```settings.toml```:
```vim settings.toml```
and assign $DIR/myh264/ to ```x264_dir```.

Then run
```python generate_mpeg_curve.py```
That generates the accuracy-bandwidth trade-off for AWStream baseline _(we only pick AWStream baseline as it is the closest baseline. We use bandwidth since it is the dominant factor of the delay in AccMPEG and AWStream settings, so better accuracy-bandwidth trade-off ==> better accuracy-delay trade-off)_.

Then run ```python batch_blackgen_roi.py``` to run AccMPEG. After finish running, take a look at the stats file
```vim artifact/stats_QP30_thresh7_segmented_FPN```

at the end of this file, the stats should look like this:
```yaml
- application: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
  bw: 8283676
  conf: 0.7
  f1: 0.948907196521759
  fn: 57
  fp: 42
  ground_truth_name: artifact/dashcamcropped_1_qp_30.mp4
  gt_conf: 0.7
  pr: 0.9632415771484375
  re: 0.959693431854248
  sum_f1: 0.611764705882353
  tp: 78
  video_name: artifact/dashcamcropped_1_roi_bound_0.2_conv_1_hq_30_lq_40_app_FPN.mp4
```
(The exact accuracy number and bandwidth may differ due to different environment setup.)
And if you scroll up, you can see that the performance of AWStream is this:
```yaml
- application: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
  bw: 15416126
  conf: 0.7
  f1: 0.9420496225357056
  fn: 40
  fp: 96
  ground_truth_name: artifact/dashcamcropped_1_qp_30.mp4
  gt_conf: 0.7
  pr: 0.9506137371063232
  re: 0.9616043567657471
  sum_f1: 0.5828220858895705
  tp: 95
  video_name: artifact/dashcamcropped_1_qp_32.mp4
```
which has almost 2x bandwidth comsumption (thus almost 2x delay) and still has lower accuracy (f1 score).


