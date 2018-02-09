# YOLO_DeepSORT

Detect using [YOLOv2](https://pjreddie.com/darknet/) and track using [DeepSORT](https://github.com/nwojke/deep_sort)

## Reuqirement
	tensorflow >= 1.0
	cv2, numpy

## Arch
```shell
YOLO_DeepSORT/
├── backup
	├── yolo_model_weights_file.weights
	...

├──cfg
	├──yolo_model_config_file.cfg
	├──yolo_model_class_type_file.data

├──resources
	├──networks
		├──tensorflow_model_weights_file.ckpt

├──0130 (it is a directory for frames saving)
  ├──01 (the serial id)
		├──det (for saving det.txt which contains all the detection informations)
		├──img1
			├──your_video_frames.jpg
			...

├──temp

├──your_yolo_lib_file.so
```
## remember to modify *_path in deep_sort_app.py

## How to run

```shell
python deep_sort_app.py  --display true --run_type instant
python deep_sort_app.py  --display true --run_type pre_computed
```
or
```shell
python3 deep_sort_app.py --display true --run_type instant
python3 deep_sort_app.py --display true --run_type pre_computed
```
### see argparser() for more infomations about other arguments.

### Have been tested under python 2.7/3.5/3.6 with tensorflow 1.2.0 (because I work with CUDA 8)

# NOTICE
The tensorflow ckpt file can be download from [here](https://owncloud.uni-koblenz.de/owncloud/s/f9JB0Jr7f3zzqs8).
See [DeepSORT](https://github.com/nwojke/deep_sort).

You can compile your own yolo lib file, but be sure that all the functions have the same usage as those in yolov2.py.

# Citation

## DeepSORT

If you find DeepSORT useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649}
    }

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      doi={10.1109/ICIP.2016.7533003}
    }

 ## YOLOv2

 Umm, he does not like this.
 [YOLOv2](https://pjreddie.com/darknet/)




