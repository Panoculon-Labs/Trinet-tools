"""
Trinet-Tools — utilities for working with Trinet camera recordings.

The Trinet camera produces synchronized video + inertial data. Recordings come
in two flavours:

  1. On-board SD recordings: a triple of files sharing a base name —
     <name>.mp4, <name>.imu, <name>.vts.

  2. Host-side UVC recordings: a single .mp4 with the inertial samples embedded
     as SEI NAL units inside the H.264 bitstream. ``extract_sei`` recovers the
     same .imu / .vts sidecars from such an MP4 so the rest of the toolkit can
     consume both flavours uniformly.

Public API:

  reader.read_imu(path)            -> ImuData
  reader.read_vts(path)            -> VtsData
  reader.interpolate_imu_to_frames -> per-frame IMU samples aligned to video
  reader.get_per_frame_fsync_delay_us
  extract_sei.extract(mp4, out)    -> recover sidecars from a UVC MP4
  madgwick.run_madgwick            -> orientation fusion (post-processing)

See ``docs/data_formats.md`` for the on-disk file layout.
"""

from . import reader      # noqa: F401
from . import extract_sei  # noqa: F401
from . import madgwick    # noqa: F401

__version__ = "1.0.0"
