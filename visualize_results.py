import matplotlib.pyplot as plt
import yaml
with open('stats', 'r') as f:
    data_full = yaml.load(f.read())

plt.style.use('bmh')

video_names = ['trafficcam_%d' % (i+1) for i in range(4)] + ['dashcam_%d' % (i+1) for i in range(4)] 
fig, axs = plt.subplots(len(video_names), figsize=(7, 30))
for i, video_name in enumerate(video_names):
    data = [i for i in data_full if video_name in i['video_name']]
    metric = 'f1'
    
    axs[i].scatter([i['bw'] for i in data if 'compressed' not  in i['video_name']], [i[metric] for i in data if 'compressed' not in i['video_name']])
    axs[i].scatter([i['bw'] for i in data if 'downsample'  in i['video_name']], [i[metric] for i in data if 'downsample'  in i['video_name']])
    axs[i].scatter([i['bw'] for i in data if 'full'  in i['video_name']], [i[metric] for i in data if 'full'  in i['video_name']])

    axs[i].set_xlim(left=0)
    axs[i].set_ylabel('Accuracy (%s)' % metric)
    axs[i].set_xlabel('Bandwidth (B)')
fig.tight_layout()
fig.savefig('results_8_videos.jpg', bbox_inches='tight')