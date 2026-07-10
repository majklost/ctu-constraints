'''
Create a dataset with distance functions and masks for training - arficial ellipse dataset.
'''
from argparse import ArgumentParser
from pathlib import Path
from constraints.generators.generators import ArteryGeneratorAffine, ArteryGeneratorDeformed



def create_affine(args):
    pass

def create_deformed(args):
    dataset_trn = ArteryGeneratorDeformed(num_samples=1000, fixed_seed=42, magnitude=7.0, integrations=2, scales=14,fractal_mode="blur")





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "num_samples", type=int,  help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory e.g. data/artificial/affine"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--generator_type",
        type=str,
        choices=["affine", "deformed","both"],
        default="affine",
        help="Type of generator to use",
    )
    
    args = parser.parse_args()

    if args.generator_type == "affine":
        create_affine(args)
    elif args.generator_type == "deformed":
        create_deformed(args)
    else:
        raise ValueError(f"Unknown generator type: {args.generator_type}")

