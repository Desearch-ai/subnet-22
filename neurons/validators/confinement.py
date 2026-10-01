"""Locks the check's child process in before it reads anything a miner wrote."""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys

PR_SET_NO_NEW_PRIVS = 38
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
O_PATH = 0o10000000
O_CLOEXEC = 0o2000000

LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
FS_EXECUTE, FS_READ_FILE, FS_READ_DIR = 1, 4, 8
FS_RIGHTS_BY_ABI = {
    1: (1 << 13) - 1,
    2: (1 << 14) - 1,
    3: (1 << 15) - 1,
    4: (1 << 15) - 1,
}
FS_RIGHTS_NEWEST = (1 << 16) - 1
NET_BIND_TCP, NET_CONNECT_TCP = 1, 2
NET_ABI = 4
SYSTEM_ROOTS = ("/usr", "/lib", "/lib64", "/etc/ssl", "/proc/self")


def confine() -> list[str]:
    """Clears secrets, then shuts off network and files; returns what this system could not shut."""
    os.environ.clear()
    os.chdir("/")
    if sys.platform != "linux":
        return ["network", "files"]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        return ["network", "files"]
    network = libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) == 0
    files, blocks_tcp = restrict_files(libc)
    return [
        name
        for name, done in (("network", network or blocks_tcp), ("files", files))
        if not done
    ]


def restrict_files(libc) -> tuple[bool, bool]:
    """Read-only access to the interpreter and libraries, nothing else; and no TCP where Landlock can."""
    if platform.machine() not in ("x86_64", "aarch64", "arm64"):
        return False, False
    abi = libc.syscall(
        LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION
    )
    if abi < 1:
        return False, False
    fs_rights = FS_RIGHTS_BY_ABI.get(abi, FS_RIGHTS_NEWEST)
    if abi >= NET_ABI:
        attr = struct.pack("QQ", fs_rights, NET_BIND_TCP | NET_CONNECT_TCP)
    else:
        attr = struct.pack("Q", fs_rights)
    ruleset = libc.syscall(LANDLOCK_CREATE_RULESET, attr, len(attr), 0)
    if ruleset < 0:
        return False, False
    try:
        readable = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
        for root in readable_roots():
            try:
                fd = os.open(root, O_PATH | O_CLOEXEC)
            except OSError:
                continue
            try:
                rule = struct.pack("=Qi", readable, fd)
                libc.syscall(
                    LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, rule, 0
                )
            finally:
                os.close(fd)
        if libc.syscall(LANDLOCK_RESTRICT_SELF, ruleset, 0) != 0:
            return False, False
    finally:
        os.close(ruleset)
    return True, abi >= NET_ABI


def readable_roots() -> list[str]:
    """The interpreter and installed packages, never the working tree with its keys and settings."""
    here = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, *SYSTEM_ROOTS}
    roots |= {
        path
        for path in sys.path
        if path and os.path.isdir(path) and os.path.realpath(path) != here
    }
    return sorted(os.path.realpath(root) for root in roots if os.path.exists(root))
