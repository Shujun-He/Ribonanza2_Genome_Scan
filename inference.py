import pandas as pd
from Dataset import *
from Network import *
#from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json
import yaml
from Scoring import get_scores, get_scores_parallel, get_ok_scores
from Functions import *
from Bio import SeqIO
#import torch_tensorrt


parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--genome_file', type=str, default="../../input/GRCh38_latest_genomic.fna")


args = parser.parse_args()

config = load_config_from_yaml(args.config_path)

accelerator = Accelerator(mixed_precision='bf16')

config.print()


os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
os.system('mkdir predictions')
os.system('mkdir plots')
os.system('mkdir subs')

def inference_chromosome(sequence,chromosome):


    print("doing inference for",chromosome)

    os.system(f'mkdir predictions/')
    os.system(f'mkdir predictions/{chromosome}')

    val_dataset=SlidingWindowTestRNAdataset(sequence,window_size=config.window_size, stride=config.stride)
    val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,num_workers=4)

    val_loader= accelerator.prepare(
         val_loader
    )

    from multiprocessing import Pool
    pool=Pool(4)

    tbar = tqdm(val_loader)
    val_loss=0
    preds=[]
    model.eval()
    for idx, batch in enumerate(tbar):
        src=batch['sequence']#.cuda()
        #bpp=batch['bpps'].cuda().float()
        bs=len(src)


        with torch.no_grad():
            with accelerator.autocast():
                output=model(src).sigmoid()
                #output=output*diag_mask.to(output.device)

                #reactivity=reactivity_model(src,src_mask=torch.ones(*src.shape[:2]).to(src.device))
                reactivity=reactivity_model(src)

        output = output.cpu().numpy()
        reactivity = reactivity.cpu().numpy()
        #all_output = accelerator.gather(output).cpu().numpy()
        src = src.cpu().numpy()
        start_position = batch["start_position"].cpu().numpy()

        # try:
        #     df=get_scores(output,src)
        #     #df=get_scores_parallel(output,src,pool)
        #     df['start_position']=start_position
        #     df['SHAPE']=list(reactivity[:,:,0])
        #     df['DMS']=list(reactivity[:,:,1])
        #     df.to_parquet(f"predictions/{chromosome}/batch{idx}_{accelerator.process_index}.parquet")
        # except:
        #     pass
        try:
            df=get_scores(output,src)
            #df=get_scores_parallel(output,src,pool)
            df['start_position']=start_position
            df['SHAPE']=list(reactivity[:,:,0])
            df['DMS']=list(reactivity[:,:,1])
            df=get_ok_scores(df)
            df.to_parquet(f"predictions/{chromosome}/batch{idx}_{accelerator.process_index}.parquet")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # Optionally, you can print more detailed traceback information:
            import traceback
            print(traceback.format_exc())

# sequence_file=f'chromosomes/{args.chromosome_file}'
# sequence=open(sequence_file).read()
# chromosome=args.chromosome_file.strip('.txt')


model=finetuned_RibonanzaNet(load_config_from_yaml("configs/pairwise.yaml"))
model.load_state_dict(torch.load("../weights/RibonanzaNet-SS.pt",map_location='cpu'))

reactivity_model=RibonanzaNet(load_config_from_yaml("configs/pairwise.yaml"))
reactivity_model.load_state_dict(torch.load("../weights/RibonanzaNet.pt",map_location='cpu'))
#exit()

diag_mask=mask_diagonal(np.ones((config.window_size,config.window_size)))
diag_mask=torch.tensor(diag_mask).unsqueeze(0)

model, reactivity_model= accelerator.prepare(
    model, reactivity_model
)

model=torch.compile(model)
reactivity_model=torch.compile(reactivity_model)


start_time = time.time()
restart=False
for record in SeqIO.parse(args.genome_file, "fasta"):
    print(record.id)
    # if record.id=="NC_000004.12":
    #     restart=True
    #print(record.id)
    #if restart:
    chromosome=sanitize_chromosome_name(record.id)
    sequence=str(record.seq).upper().replace('T','U')
    inference_chromosome(sequence,chromosome)

end_time = time.time()
elapsed_time = end_time - start_time

with open("inference_stats.json", 'w') as file:
        json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
