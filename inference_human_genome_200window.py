import pandas as pd
from Dataset import *
from Network import *
#from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from ranger import Ranger
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json
import yaml
from Scoring import get_scores, get_scores_parallel

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)


start_time = time.time()

parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--chromosome_file', type=str, default="NC_000001.11.txt")

args = parser.parse_args()

config = load_config_from_yaml(args.config_path)

accelerator = Accelerator(mixed_precision='fp16')

os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
os.system('mkdir predictions')
os.system('mkdir plots')
os.system('mkdir subs')

sequence_file=f'chromosomes/{args.chromosome_file}'
sequence=open(sequence_file).read()

chromosome=args.chromosome_file.strip('.txt')

print(chromosome)

os.system(f'mkdir predictions/')
os.system(f'mkdir predictions/{chromosome}')

val_dataset=SlidingWindowTestRNAdataset(sequence,window_size=200)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False)


class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.3
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("../../w_pair_exps/test5/models/model0.pt",map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)
        self.ct_predictor=nn.Linear(64,1)

    def forward(self,src):
        
        #with torch.no_grad():
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0,2,1,3)

        output=self.ct_predictor(self.dropout(pairwise_features))

        #output=output+output.permute()

        return output.squeeze(-1)

model=finetuned_RibonanzaNet(load_config_from_yaml("configs/pairwise.yaml"))
model.load_state_dict(torch.load("finetuned_model_v7.pt"))

#exit()

model, val_loader= accelerator.prepare(
    model, val_loader
)
from multiprocessing import Pool
pool=Pool(4)
#print(val_dataset.max_len)
# print(attention_mask.device)
# exit()
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
    # plt.imshow(output[0].cpu().numpy(),vmin=0,vmax=1)
    # plt.savefig("test.png")
    # #plt.show()
    # exit()
    #output = accelerator.pad_across_processes(output,1)
    output = output.cpu().numpy()
    #all_output = accelerator.gather(output).cpu().numpy()
    src = src.cpu().numpy()
    start_position = batch["start_position"].cpu().numpy()
    #preds.append(all_output)

    # if accelerator.is_local_main_process:
    #     np.save(f"predictions/{chromosome}/batch{idx}_2D",all_output)
    #     np.save(f"predictions/{chromosome}/batch{idx}_seq",src)
    # try:
    #     df=get_scores(output,src)
    #     #df=get_scores_parallel(output,src,pool)
    #     df.to_parquet(f"predictions/{chromosome}/batch{idx}_{accelerator.process_index}.parquet")
    #     df['start_position']=start_position
    # except:
    #     pass
    try:
        df=get_scores(output,src)
        #df=get_scores_parallel(output,src,pool)
        df['start_position']=start_position
        df.to_parquet(f"predictions/{chromosome}/batch{idx}_{accelerator.process_index}.parquet")
    except:
        pass
    #df = accelerator.gather_for_metrics(df)

    #df = pd.concat(df)

    #print(df.shape)

    #exit()


    #break
#exit()



# if accelerator.is_local_main_process:
#     preds=np.concatenate(preds)
#     preds_uint8=np.uint8(preds*255+0.5)
    
#     np.savez("predictions/human_genome",preds_uint8)


    end_time = time.time()
    elapsed_time = end_time - start_time

    with open("inference_stats_human_genome.json", 'w') as file:
            json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
