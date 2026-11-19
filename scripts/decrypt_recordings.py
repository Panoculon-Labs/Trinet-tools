#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""
Decrypt Trinet recordings that the camera encrypted on its memory card, and make
the key pair that encryption uses. Needs camera firmware 0.5.16 or newer.

    # once: make a key pair. Keep my_key.pem secret and back it up; install
    # my_key.tvk on the camera from the Trinet app (Camera settings -> SD encryption)
    python scripts/decrypt_recordings.py keygen my_key.pem my_key.tvk --passphrase

    # decrypt a take, a folder or a whole card; the originals are left untouched
    python scripts/decrypt_recordings.py decrypt --key my_key.pem --out decrypted/ /media/SD/Trinet

    # write the camera key (.tvk) again from the private key
    python scripts/decrypt_recordings.py export-pubkey my_key.pem my_key.tvk

Each encrypted take has a .tkey file beside it; copy it along with the take.
The decrypted MP4, .imu, .vts and .tel are ordinary Trinet recordings for the
rest of the toolkit. A private key that is not the take's key is refused with
the take's key ID, and nothing is written. Guide:
https://docs.panoculonlabs.com/data/encryption/
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trinet_tools.decrypt import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
