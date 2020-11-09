# video-compression

To run the code, just need to use three files: ```genereate_mpeg_curve.py``` to generate mpeg curve, ```batch_saliency.py``` to generate ground truth mask, and ```batch_maskgen.py``` to generate the mask through the neural network. 

To use ```generate_mepg_curve.py```, here is an example:
```
python generate_mpeg_curve.py -i /tank/youtube_videos/train_first/trafficcam_1_train
```

To use ```batch_saliency.py```, you need to go into that file and edit this line
```
v_list = ['train_first/trafficcam_1_train', 'train_first/dashcam_1_train']
```
to specify the video you want to run. The path can be absolute path or relative path.

The usage of ```batch_maskgen.py``` is similiar as above.

While running ```batch_saliency.py``` and ```batch_maskgen.py```, the program will automatically generate scenes for visualization purpose. To see the visualization, just go to the ```visualization``` folder under this repo.
