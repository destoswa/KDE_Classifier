import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from time import time
from packaging import version
# import tkinter as tk
# from tkinter import messagebox


# ===================================================
# ================= HYPERPARAMETERS =================
# ===================================================
# preprocessing
do_preprocess = True
verbose = False

# inference
chunk_size=100    # number of files processed at the time (0 = all of them)
batch_size = 12
num_workers = 12
num_class = 3
grid_size = 64
kernel_size = 1
num_repeat_kernel = 2

src_inf_root = "./inference/"
src_inf_data = "data_test"
src_inf_results = "results"
src_model = "./models/pretrained/model_KDE.tar"
inference_file = "modeltrees_inference.csv"
with open(src_inf_root + 'modeltrees_shape_names.txt', 'r') as f:
    sample_labels = f.read().splitlines()

# ===================================================
# ===================================================

# store relation between number and class label
dict_labels = {}
for idx, cls in enumerate(sample_labels):
    dict_labels[idx] = cls


def inference_by_chunk(args):
    verbose = args['verbose']
    lst_files = os.listdir(os.path.join(args['src_inf_root'], args['src_inf_data']))

    if args["chunk_size"] > 1 and args['chunk_size'] < len(lst_files):
        # creates chunks of samples to infer on
        lst_chunk_of_tiles = [lst_files[x:min(y,len(lst_files))] for x, y in zip(
            range(0, len(lst_files) - args["chunk_size"], args["chunk_size"]),
            range(args["chunk_size"], len(lst_files), args["chunk_size"]),
            )]
        if lst_chunk_of_tiles[-1][-1] != lst_files[-1]:
            lst_chunk_of_tiles.append(lst_files[(len(lst_chunk_of_tiles)*args["chunk_size"])::])
        
        # creates results architecture
        if os.path.exists(os.path.join(args['src_inf_root'], args['src_inf_results'])):
            print('A "results" directory already exists.')
            answer = None
            while answer not in ['y', 'yes', 'n', 'no', '']:
                answer = input("Do you want to overwrite it (y/n)?")
                if answer.lower() in ['y', 'yes', '']:
                    shutil.rmtree(os.path.join(args['src_inf_root'], args['src_inf_results']))
                elif answer.lower() in ['n', 'no']:
                    print("Stoping the process..")
                    quit()
                else:
                    print("wrong input.")
        os.makedirs(os.path.join(args['src_inf_root'], args['src_inf_results']))

        # modify args for inference
        base_results = args['src_inf_results']
        base_data = args['src_inf_data']
        args['src_inf_data'] = "temp_chunk_data"
        args['src_inf_results'] = 'results_temp'
        df_results = pd.DataFrame(columns=['file_name', 'class'])
        df_failed_samples = pd.DataFrame(columns=['Index', 'data', 'label'])
        for num_chunk, chunk in tqdm(enumerate(lst_chunk_of_tiles), total=len(lst_chunk_of_tiles), desc="Infering on chunks", smoothing=0.9):
            if verbose:
                print(f"=== PROCESSING CHUNK {num_chunk + 1} / {len(lst_chunk_of_tiles)}")
                
            # create temp folder for chunks
            if os.path.exists(os.path.join(args['src_inf_root'], 'temp_chunk_data')):
                shutil.rmtree(os.path.join(args['src_inf_root'], 'temp_chunk_data'))
            os.makedirs(os.path.join(args['src_inf_root'], 'temp_chunk_data'))

            # copy chunk of tiles
            if verbose:
                print("Copying:")
            for _, file in tqdm(enumerate(chunk), total=len(chunk), desc="Copying", disable=not verbose):
                shutil.copyfile(
                    os.path.join(args['src_inf_root'], base_data, file),
                    os.path.join(args['src_inf_root'], args['src_inf_data'], file),
                )

            # call inference
            inference(args, verbose=False)
                        
            # transfert results
            for r,_,f in os.walk(os.path.join(args['src_inf_root'], args['src_inf_results'])):
                for file in f:
                    if file.endswith('.pcd'):
                        source_file_path = os.path.join(r, file)
                        rel_path = os.path.relpath(source_file_path, os.path.join(args['src_inf_root'], args['src_inf_results']))
                        target_file_path = os.path.join(args['src_inf_root'], base_results, rel_path)
                        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, target_file_path)
                    elif file.endswith('.csv'):
                        if file == 'failed_data.csv':
                            df_failed_samples = pd.concat([df_failed_samples, pd.read_csv(os.path.join(r, file), sep=';')], axis=0)
                            df_failed_samples.to_csv(os.path.join(args['src_inf_root'], base_results, 'failed_data.csv'), sep=';', index=False)
                        elif file == "results.csv":
                            df_results = pd.concat([df_results, pd.read_csv(os.path.join(r, file), sep=';')], axis=0)
                            df_results.to_csv(os.path.join(args['src_inf_root'], base_results, 'results.csv'), sep=';', index=False)
                    else:
                        print("WARNNING: Weird file: ", os.path.join(r, file))
            
            # empty temp results
            shutil.rmtree(os.path.join(args['src_inf_root'], args['src_inf_results']))
        # empty temp data
        shutil.rmtree(os.path.join(args['src_inf_root'], 'temp_chunk_data'))
    else:
        inference(args)


def inference(args, verbose=True):
    # create the folders for results
    if os.path.exists(os.path.join(args['src_inf_root'], args['src_inf_results'])):
        print('A "results" directory already exists.')
        answer = None
        while answer not in ['y', 'yes', 'n', 'no', '']:
            answer = input("Do you want to overwrite it (y/n)?")
            if answer.lower() in ['y', 'yes', '']:
                shutil.rmtree(os.path.join(args['src_inf_root'], args['src_inf_results']))
            elif answer.lower() in ['n', 'no']:
                print("Stoping the process..")
                quit()
            else:
                print("wrong input.")
    os.makedirs(os.path.join(args['src_inf_root'], args['src_inf_results']))

    # load the model
    if verbose:
        print("Loading model...")
    conf = {
        "num_class": args['num_class'],
        "grid_dim": args['grid_size'],
    }
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        checkpoint = torch.load(args['src_model'], weights_only=False)
    else:
        checkpoint = torch.load(args['src_model'])
    # checkpoint = torch.load(SRC_MODEL, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for cls in args['sample_labels']:
        os.makedirs(os.path.join(os.path.join(args['src_inf_root'], args['src_inf_results']), cls), exist_ok=True)

    # preprocess the samples
    if args['do_preprocess']:
        lst_files_to_process = [os.path.join(args['src_inf_data'], cls) for cls in os.listdir(os.path.join(args['src_inf_root'], args['src_inf_data'])) if cls.endswith('.pcd')]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(args['src_inf_root'] + args['inference_file'], sep=';', index=False)

    # make the predictions
    if verbose:
        print("making predictions...")
    kde_transform = ToKDE(args['grid_size'], args['kernel_size'], args['num_repeat_kernel'])
    inferenceSet = ModelTreesDataLoader(args['inference_file'], args['src_inf_root'], split='inference', transform=None, do_update_caching=args['do_preprocess'], kde_transform=kde_transform, result_dir=args['src_inf_results'], verbose=verbose)
    if len(inferenceSet.num_fails) > 0:
        os.makedirs(os.path.join(os.path.join(args['src_inf_root'], args['src_inf_results']), 'failures/'), exist_ok=True)
        for _, file_src in inferenceSet.num_fails:
            shutil.copyfile(
                src=file_src, 
                dst=os.path.join(os.path.join(args['src_inf_root'], args['src_inf_results']), 'failures/', os.path.basename(file_src)))

    inferenceDataLoader = DataLoader(inferenceSet, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for _, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9, desc="Classifying", disable=not verbose):
        # load the samples and labels on cuda
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.cuda(), target.cuda()

        # compute prediction
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]

        # copy samples into right result folder
        for idx, pred in enumerate(pred_choice):
            fn = os.path.basename(filenames[idx].replace('.pickle', ''))
            shutil.copyfile(
                os.path.join(args['src_inf_root'], args['src_inf_data'], fn),
                os.path.join(os.path.join(args['src_inf_root'], args['src_inf_results']), dict_labels[pred.item()], fn),
                )
            df_predictions.loc[len(df_predictions)] = [os.path.join(args['src_inf_data'], fn), pred.item()]

    # save results in csv file
    df_predictions.to_csv(os.path.join(os.path.join(args['src_inf_root'], args['src_inf_results']), 'results.csv'), sep=';', index=False)

    # clean temp
    inferenceSet.clean_temp()


def main():
    args = {
        'do_preprocess': do_preprocess,
        'verbose': verbose,
        'chunk_size': chunk_size,
        'grid_size': grid_size,
        'num_class': num_class,
        'kernel_size': kernel_size,
        'num_repeat_kernel': num_repeat_kernel,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'src_inf_root': src_inf_root,
        'src_inf_data': src_inf_data,
        'src_inf_results': src_inf_results,
        'src_model': src_model,
        'inference_file':inference_file,
        'sample_labels': sample_labels,
    }

    inference_by_chunk(args)


if __name__ == "__main__":
    start = time()
    main()
    duration = time() - start
    hours = int(duration/3600)
    mins = int((duration - 3600 * hours)/60)
    secs = int((duration - 3600 * hours - 60 * mins))
    print(duration)
    print(f"Time to process inference: {hours}:{mins}:{secs}")