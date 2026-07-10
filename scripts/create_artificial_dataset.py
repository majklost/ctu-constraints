'''
Create a dataset with distance functions and masks for training - arficial ellipse dataset.
'''
from argparse import ArgumentParser
from constraints.generators.generators import ArteryGeneratorAffine, ArteryGeneratorDeformed








if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "num_samples", type=int,  help="Number of samples to generate"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory"
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
        generator = ArteryGeneratorAffine(num_samples=args.num_samples, fixed_seed=args.seed)
    elif args.generator_type == "deformed":
        generator = ArteryGeneratorDeformed(num_samples=args.num_samples, fixed_seed=args.seed)
    else:
        raise ValueError(f"Unknown generator type: {args.generator_type}")

