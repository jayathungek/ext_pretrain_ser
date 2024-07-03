# Extended pretraining for SER

## Install
```bash
git clone https://github.com/jayathungek/ext_pretrain_ser.git
cd ext_pretrain_ser
conda env create -f environment.yml
mkdir saved_models
```

## Running tests
```bash
python -m unittest
```
## Data preparation
1. Start by downloading your dataset from the original source. 
2. Navigate to `extpt/datasets/<your-dataset-name>.py` and change the DATA_DIR variable to point to your downloaded dataset root.
3. Run the following to create a training manifest under your dataset root:
    - `python -m extpt.data make_manifest --dataset-name [your-dataset-name]`

NOTE: if you want to use datasets that are not in the following list, you need to create a python file describing the various constants associated with your dataset. Please see the examples under `extpt/datasets` for reference. You will also need to write a custom manifest function that will handle the creation of the manifest.

### Datasets Used
Some of these may require filling in a release form and waiting for approval:

* ASVP-ESD -- [link](https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances)
* eNTERFACE'05 -- [link](https://enterface.net/enterface05/docs/results/databases/project2_database.zip)
* IEMOCAP -- [link](https://sail.usc.edu/iemocap/iemocap_release.htm)
* MSPodcast -- [link](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
* TESS -- [link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## Usage
### Pretrained models
Pretrained models have been uploaded here:
* [SSAST-Base-Patch-400.pth](https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1)
    * Copy into `extpt/saved_models`
* [ssast.base+mspodcast.tar](https://www.dropbox.com/scl/fi/323zrodunuvjtihsuylci/ssast.base-mspodcast.tar?rlkey=3miakel7ksrl5wv958rz46vns&st=xpi3kv0w&dl=0)
    * Extract using `tar -xvf ssast.base+mspodcast.tar` into `extpt/saved_models`

### Sample pretraining code:

```bash
python -m extpt.train start \ 
    --mode pretrain \ 
    --name pretrain_mspodcast \ 
    --log-wandb \ 
    --dataset mspodcast \ 
    --checkpoint-freq 5
```

###  Sample fine-tuning code
```bash
python -m extpt.train start \ 
    --mode finetune\ 
    --name finetune_tess \ 
    --log-wandb \ 
    --dataset tess\ 
    --checkpoint-freq 5 \ 
    --freeze-pt-weights \ 
    --embed-dim 688 \ 
    --pretrained-name  ssast.base+mspodcast
```
