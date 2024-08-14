# sam-api
SAM via API (Segment Anything v1)

## Install Segment Anything from git repo

```
cd /your_working_dir
git clone https://github.com/facebookresearch/segment-anything.git
```

Follow the instructions to install dependencies and download models.

Refer to https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md

## Download API code

Download this repo to /your_working_dir/segment-anything/api and install dependencies.
```
pip install -r api/requirements.txt
```
## Note
onnxruntime should be fixed to 1.15.1 according to this issue: 
https://github.com/facebookresearch/segment-anything/issues/585
```shell
pip uninstall onnxruntime
pip install onnxruntime==1.15.1
```
