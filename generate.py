import argparse
import logging
from pathlib import Path
import pandas as pd
from model.minGPT import load_gpt_model
from model.sampler import sample
import math


def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'device:\t{args.device}')

    gpt_path = args.model_path
    out_path = args.out_file

    model_def = Path(gpt_path).with_suffix('.json')
    model = load_gpt_model(model_def, gpt_path, args.device, copy_to_cpu=True)

    logger.info(f'Generate samples...')
    num_to_sample = args.num_to_sample
    sample_seqs = sample(model, num_to_sample=num_to_sample, device=args.device, batch_size=args.batch_size,
                         max_len=args.max_len, temperature=args.temperature)
    uniq_seqs = list(set(sample_seqs))

    logger.info(f"Totally {len(uniq_seqs)} unique sequences!")
    # Save seqs
    df_seqs = pd.DataFrame(uniq_seqs, columns=['seqs'])
    df_seqs.to_csv(out_path, index=False)

    logger.info(f'Generation finished!')


def get_args():
    parser = argparse.ArgumentParser(description='Generate SMILES from a GPT model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='Full path to GPT model')
    parser.add_argument('--out_file', type=str, help='Output file path')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--num_to_sample', default=10000, type=int, help='Molecules to sample, default=10000')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--batch_size', default=64, type=int, help='Batch_size during sampling, default=64')
    optional.add_argument('--max_len', default=13, type=int, help='Maximum seqs length, default=13')
    optional.add_argument('--temperature', default=1.0, type=float, help='Temperature during sampling, default=1.0')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
