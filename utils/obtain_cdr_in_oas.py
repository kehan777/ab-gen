import os
import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def process_unpaired(input_file, output_dir, out_type='cdr', is_gz=False):
    data = pd.read_csv(input_file, skiprows=1) if not is_gz \
        else pd.read_csv(input_file, skiprows=1, compression='gzip', error_bad_lines=False)
    output_file = os.path.join(output_dir, '.'.join((Path(input_file).stem, out_type)))
    if out_type == 'cdr':
        cdr = data[['cdr1_aa', 'cdr2_aa', 'cdr3_aa']]
        cdr.to_csv(output_file)


def process_paired(input_file, output_dir, out_type='cdr', is_gz=False):
    data = pd.read_csv(input_file, skiprows=1) if not is_gz \
        else pd.read_csv(input_file, skiprows=1, compression='gzip', error_bad_lines=False)
    output_file = os.path.join(output_dir, '.'.join((Path(input_file).stem, out_type)))
    if out_type == 'cdr':
        cdr = data[['cdr1_aa_heavy', 'cdr2_aa_heavy', 'cdr3_aa_heavy',
                    'cdr1_aa_light', 'cdr2_aa_light', 'cdr3_aa_light']]
        cdr.to_csv(output_file)


def main(args):
    logger.info(f'Parsing file {args.input}!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.in_type == 'unpaired':
        process_unpaired(args.input, args.output_dir, out_type=args.out_type, is_gz=args.gz)
    elif args.in_type == 'paired':
        process_paired(args.input, args.output_dir, out_type=args.out_type, is_gz=args.gz)
    logger.info(f"Parsing finished! Results saved to folder {args.output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-p', type=str, help='Path to OAS file, csv format')
    parser.add_argument('--output_dir', '-d', type=str, help='Output directory')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--in_type', type=str, default='unpaired', help='Input: paired/unpaired, default=unpaired')
    optional.add_argument('--out_type', type=str, default='cdr', help='Output type, default=cdr')
    optional.add_argument('--gz', action="store_true", help='File gzip compressed or not, default False')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
