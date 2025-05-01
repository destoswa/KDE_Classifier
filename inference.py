import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from time import time
# import tkinter as tk
# from tkinter import messagebox


# ===================================================
# ================= HYPERPARAMETERS =================
# ===================================================
# preprocessing
do_preprocess = False

# inference
batch_size = 12
num_workers = 12
num_class = 3
grid_size = 64
kernel_size = 1
num_repeat_kernel = 2
SRC_INF_ROOT = "./inference/"
SRC_INF_DATA = "test"
SRC_INF_RESULTS = os.path.join(SRC_INF_ROOT, 'results/')
SRC_MODEL = "./models/pretrained/model_KDE.tar"
INFERENCE_FILE = "modeltrees_inference.csv"
with open(SRC_INF_ROOT + 'modeltrees_shape_names.txt', 'r') as f:
    SAMPLE_LABELS = f.read().splitlines()

# ===================================================
# ===================================================

# store relation between number and class label
dict_labels = {}
for idx, cls in enumerate(SAMPLE_LABELS):
    dict_labels[idx] = cls


def inference(args):
    # create the folders for results
    if os.path.exists(SRC_INF_RESULTS):
        print('A "results" directory already exists.')
        answer = None
        while answer not in ['y', 'yes', 'n', 'no', '']:
            answer = input("Do you want to overwrite it (y/n)?")
            if answer.lower() in ['y', 'yes', '']:
                shutil.rmtree(SRC_INF_RESULTS)
            elif answer.lower() in ['n', 'no']:
                print("Stoping the process..")
                quit()
            else:
                print("wrong input.")
    os.makedirs(SRC_INF_RESULTS)

    # load the model
    print("Loading model...")
    conf = {
        "num_class": args['num_class'],
        "grid_dim": args['grid_size'],
    }
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    checkpoint = torch.load(SRC_MODEL, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for cls in SAMPLE_LABELS:
        os.makedirs(os.path.join(SRC_INF_RESULTS, cls), exist_ok=True)

    # preprocess the samples
    if args['do_preprocess']:
        lst_files_to_process = [os.path.join(SRC_INF_DATA, cls) for cls in os.listdir(os.path.join(SRC_INF_ROOT, SRC_INF_DATA)) if cls.endswith('.pcd')]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(SRC_INF_ROOT + INFERENCE_FILE, sep=';', index=False)

    # make the predictions
    print("making predictions...")
    kde_transform = ToKDE(args['grid_size'], args['kernel_size'], args['num_repeat_kernel'])
    inferenceSet = ModelTreesDataLoader(INFERENCE_FILE, SRC_INF_ROOT, split='inference', transform=None, do_update_caching=args['do_preprocess'], kde_transform=kde_transform)
    if len(inferenceSet.num_fails) > 0:
        os.makedirs(os.path.join(SRC_INF_RESULTS, 'failures/'), exist_ok=True)
        for _, file_src in inferenceSet.num_fails:
            shutil.copyfile(
                src=file_src, 
                dst=os.path.join(SRC_INF_RESULTS, 'failures/', os.path.basename(file_src)))

    inferenceDataLoader = DataLoader(inferenceSet, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for batch_id, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9):
        # load the samples and labels on cuda
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.cuda(), target.cuda()

        # compute prediction
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]

        # copy samples into right result folder
        for idx, pred in enumerate(pred_choice):
            fn = os.path.basename(filenames[idx].replace('.pickle', ''))
            dest = os.path.join("inference/results/", dict_labels[pred.item()], fn)
            shutil.copyfile(
                os.path.abspath(os.path.join(SRC_INF_ROOT, SRC_INF_DATA, fn)),
                os.path.abspath(dest)
                )
            df_predictions.loc[len(df_predictions)] = [os.path.join(SRC_INF_DATA, fn), pred.item()]

    # save results in csv file
    df_predictions.to_csv(SRC_INF_ROOT + 'results/results.csv', sep=';', index=False)


def main():
    args = {
        'do_preprocess': do_preprocess,
        'grid_size': grid_size,
        'num_class': num_class,
        'kernel_size': kernel_size,
        'num_repeat_kernel': num_repeat_kernel,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    inference(args)


if __name__ == "__main__":
    start = time()
    main()
    duration = time() - start
    hours = int(duration/3600)
    mins = int((duration - 3600 * hours)/60)
    secs = int((duration - 3600 * hours - 60 * mins))
    print(duration)
    print(f"Time to process inference: {hours}:{mins}:{secs}")