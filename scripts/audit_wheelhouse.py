#!/usr/bin/env python3
"""Report missing Linux/Python 3.10 wheelhouse dependencies without installing.

The resolver in pip exits at its first unavailable distribution.  This tool
walks metadata from every wheel already present, so a single audit can expose
the remaining direct and transitive requirements before another upload.
"""

from __future__ import annotations

import argparse
import email
import sys
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


def read_wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if not metadata_names:
            raise ValueError(f"no METADATA entry in {wheel.name}")
        # A few bundled wheels (notably Ray) contain metadata for vendored
        # distributions too.  The first top-level dist-info record is the
        # package's own metadata and is sufficient for dependency auditing.
        metadata_name = sorted(metadata_names, key=lambda name: (name.count("/"), name))[0]
        message = email.message_from_bytes(archive.read(metadata_name))
    name = canonicalize_name(message["Name"])
    version = Version(message["Version"])
    requirements = [Requirement(item) for item in message.get_all("Requires-Dist", [])]
    return name, version, requirements


def marker_matches(requirement: Requirement, environment: dict, extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    for extra in extras or {""}:
        marker_environment = dict(environment)
        marker_environment["extra"] = extra
        if requirement.marker.evaluate(marker_environment):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheelhouse", type=Path)
    args = parser.parse_args()

    package_data = defaultdict(list)
    for wheel in sorted(args.wheelhouse.glob("*.whl")):
        try:
            name, version, requirements = read_wheel_metadata(wheel)
        except Exception as error:
            print(f"warning: skip {wheel.name}: {error}", file=sys.stderr)
            continue
        package_data[name].append((version, requirements, wheel.name))

    for entries in package_data.values():
        entries.sort(key=lambda item: item[0], reverse=True)

    environment = default_environment()
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": "3.10.21",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "python_full_version": "3.10.21",
            "python_version": "3.10",
            "sys_platform": "linux",
        }
    )

    roots = [
        Requirement("vllm==0.11.2"),
        Requirement("transformers==4.57.6"),
        Requirement("qwen-vl-utils==0.0.14"),
    ]
    queue = deque((root, "requested by root") for root in roots)
    processed = set()
    missing = defaultdict(set)

    while queue:
        requirement, parent = queue.popleft()
        name = canonicalize_name(requirement.name)
        choices = [
            entry for entry in package_data.get(name, []) if entry[0] in requirement.specifier
        ]
        if not choices:
            missing[str(requirement)].add(parent)
            continue
        version, requirements, wheel_name = choices[0]
        extras = frozenset(requirement.extras)
        key = (name, version, extras)
        if key in processed:
            continue
        processed.add(key)
        for child in requirements:
            if marker_matches(child, environment, set(extras)):
                queue.append((child, f"{name}=={version}"))

    print(f"wheel files: {sum(len(items) for items in package_data.values())}")
    print(f"resolved installed-wheel nodes: {len(processed)}")
    if not missing:
        print("MISSING: none")
        return 0
    print("MISSING:")
    for requirement in sorted(missing, key=str.lower):
        parents = ", ".join(sorted(missing[requirement]))
        print(f"{requirement}\t(required by: {parents})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
