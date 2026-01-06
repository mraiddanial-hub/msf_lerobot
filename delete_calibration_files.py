#!/usr/bin/env python3
"""
Delete calibration files that limit range of motion for teleoperation.

These calibration files are generated during recording sessions and clamp
motor positions and tip vectors to recorded ranges. This prevents full
range of motion during teleoperation.

This script safely backs up and deletes them.
"""

import argparse
import logging
from pathlib import Path
import shutil
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def delete_calibrations(backup: bool = True, dry_run: bool = False):
    """Delete calibration files that limit teleoperation range"""
    
    logger.info("=" * 60)
    logger.info("AmazingHand Calibration File Cleanup")
    logger.info("=" * 60)
    
    calib_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    
    # Find calibration files
    patterns = [
        "teleoperators/lerobot_teleoperator_amazinghandtracker/**/*.json",
        "robots/lerobot_robot_amazinghand/**/*.json",
    ]
    
    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(calib_dir.glob(pattern))
    
    if not files_to_delete:
        logger.info("✓ No calibration files found - already clean!")
        return
    
    logger.info(f"\nFound {len(files_to_delete)} calibration file(s):")
    for f in files_to_delete:
        # Read and show what's being limited
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            # Show range limits
            first_key = list(data.keys())[0]
            range_info = data[first_key]
            logger.info(f"\n  {f.relative_to(calib_dir)}")
            logger.info(f"    Limits: range_min={range_info.get('range_min')}, "
                       f"range_max={range_info.get('range_max')}")
            logger.info(f"    ⚠️  This CLAMPS values during teleoperation!")
        except Exception as e:
            logger.warning(f"  Could not read {f}: {e}")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would delete these files (use --no-dry-run to actually delete)")
        return
    
    # Backup if requested
    if backup:
        backup_dir = calib_dir.parent / "calibration_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nBacking up to: {backup_dir}")
        for f in files_to_delete:
            rel_path = f.relative_to(calib_dir)
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, backup_path)
            logger.info(f"  ✓ Backed up: {rel_path}")
    
    # Delete
    logger.info("\nDeleting calibration files...")
    for f in files_to_delete:
        f.unlink()
        logger.info(f"  ✓ Deleted: {f.relative_to(calib_dir)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Calibration files deleted!")
    logger.info("=" * 60)
    logger.info("\nYour robot should now have FULL range of motion.")
    logger.info("The robot will use full servo range (0-4095) automatically.")
    
    if backup:
        logger.info(f"\nBackup saved to: {backup_dir}")
        logger.info("You can restore by copying files back if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete calibration files that limit teleoperation range"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup (default: create backup)"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually delete files (default: dry run)"
    )
    
    args = parser.parse_args()
    
    delete_calibrations(
        backup=not args.no_backup,
        dry_run=not args.no_dry_run
    )
