from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_CONDITIONS_PATH = Path(__file__).with_name("initial_conditions.json")


def yaw_to_quaternion(yaw: float) -> list[float]:
    """Return a Z-axis yaw quaternion in [qw, qx, qy, qz] order."""
    half_yaw = yaw / 2.0
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def rounded(values: list[float], precision: int | None) -> list[float]:
    if precision is None:
        return values
    return [round(value, precision) for value in values]


def validate_pose_entry(path: Path, pose_index: int, pose_entry: Any) -> dict[str, list[float]]:
    if not isinstance(pose_entry, dict) or not pose_entry:
        raise ValueError(f"{path} poses[{pose_index}] must contain object pose entries")

    for object_name, pose in pose_entry.items():
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(
                f"{path} object '{object_name}' must be [x, y, z, qw, qx, qy, qz]"
            )

    return pose_entry


def load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    poses = data.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{path} must contain a non-empty 'poses' list")

    return data


def sample_positions(
    object_names: list[str],
    rng: random.Random,
    center_x: float,
    center_y: float,
    square_size: float,
    min_distance: float,
    max_attempts: int,
) -> dict[str, tuple[float, float]]:
    half_size = square_size / 2.0
    min_x = center_x - half_size
    max_x = center_x + half_size
    min_y = center_y - half_size
    max_y = center_y + half_size

    for _ in range(max_attempts):
        sampled: dict[str, tuple[float, float]] = {}

        for object_name in object_names:
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if all(
                math.dist((x, y), other_xy) >= min_distance
                for other_xy in sampled.values()
            ):
                sampled[object_name] = (x, y)
            else:
                break

        if len(sampled) == len(object_names):
            return sampled

    raise RuntimeError(
        "Failed to sample valid positions. Increase --square-size, lower "
        "--min-distance, or raise --max-attempts."
    )


def generate_conditions(args: argparse.Namespace) -> dict[str, Any]:
    template = load_template(args.input)
    poses = template["poses"]
    if args.template_pose_index >= len(poses):
        raise ValueError(
            f"--template-pose-index {args.template_pose_index} is out of range "
            f"for {args.input}; valid range is 0..{len(poses) - 1}"
        )

    base_pose = validate_pose_entry(
        path=args.input,
        pose_index=args.template_pose_index,
        pose_entry=poses[args.template_pose_index],
    )
    object_names = list(base_pose)
    rng = random.Random(args.seed)

    generated = copy.deepcopy(template)
    generated_poses: list[dict[str, list[float]]] = []

    for _ in range(args.count):
        xy_by_object = sample_positions(
            object_names=object_names,
            rng=rng,
            center_x=args.square_center[0],
            center_y=args.square_center[1],
            square_size=args.square_size,
            min_distance=args.min_distance,
            max_attempts=args.max_attempts,
        )

        pose_entry: dict[str, list[float]] = {}
        for object_name in object_names:
            original_pose = base_pose[object_name]
            x, y = xy_by_object[object_name]
            yaw = rng.uniform(-math.pi, math.pi)
            pose = [x, y, original_pose[2], *yaw_to_quaternion(yaw)]
            pose_entry[object_name] = rounded(pose, args.precision)

        generated_poses.append(pose_entry)

    generated["poses"] = generated_poses
    return generated


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate initial_conditions.json with randomized object xy "
            "positions and Z-axis yaw rotations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CONDITIONS_PATH,
        help="Template JSON path. Defaults to pbl_umi/initial_conditions.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONDITIONS_PATH,
        help="Output JSON path. Defaults to overwriting the template file.",
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        default=1,
        help="Number of pose entries to generate.",
    )
    parser.add_argument(
        "--square-center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(0.2, 0.0),
        help="Center of the square sampling area in xy coordinates.",
    )
    parser.add_argument(
        "--square-size",
        type=positive_float,
        default=0.5,
        help="Side length of the square sampling area.",
    )
    parser.add_argument(
        "--min-distance",
        type=non_negative_float,
        default=0.1,
        help="Minimum allowed xy distance between any two objects.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=10000,
        help="Maximum rejection-sampling attempts per generated pose entry.",
    )
    parser.add_argument(
        "--precision",
        type=non_negative_int,
        default=7,
        help="Decimal places for generated floats. Use 0 or higher.",
    )
    parser.add_argument(
        "--template-pose-index",
        type=non_negative_int,
        default=0,
        help="Pose entry index in the template used for object names and z values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generated = generate_conditions(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(generated, file, indent=4, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
