# SPDX-License-Identifier: Apache-2.0
"""Local/Buildkite QEMU runner; all DAX configuration stays in the guest."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import uuid

# Third Party
from ci_selection import controls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = json.loads((HERE / "image/manifest.json").read_text())


def _run(
    args: list[str], timeout: int = 60, **kwargs: Any
) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, timeout=timeout, **kwargs)


def _sha(path: Path) -> str:
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def topology() -> list[str]:
    """Return three independent volatile Type-3 devices on one CXL host bridge."""
    args = [
        "-device",
        "pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.1",
        "-M",
        "cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G",
    ]
    for i in range(MANIFEST["cxl_device_count"]):
        args += [
            "-object",
            f"memory-backend-ram,id=vmem{i},share=on,size={MANIFEST['cxl_device_bytes']}",
            "-device",
            f"cxl-rp,port={i},bus=cxl.1,id=rp{i},chassis=0,slot={i + 2}",
            "-device",
            f"cxl-type3,bus=rp{i},volatile-memdev=vmem{i},id=cxl-vmem{i},sn={i + 1}",
        ]
    return args


def package_source(target: Path) -> dict:
    """Archive the tested checkout, including local edits, and record its identity."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    with tarfile.open(target, "w:gz") as archive:
        for name in sorted(set(os.fsdecode(n) for n in names.split(b"\0") if n)):
            path = ROOT / name
            if (path.exists() or path.is_symlink()) and not name.startswith(
                "artifacts/"
            ):
                archive.add(path, arcname=name, recursive=False)
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        tree=subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
        ).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        archive_sha256=_sha(target),
    )


class Guest:
    """Own one QEMU process, unique control endpoints, key and scratch directory."""

    def __init__(
        self,
        scratch: Path,
        output: Path,
        disk: Path,
        accel: str,
        kernel: Path | None = None,
    ) -> None:
        """Start a VM on disk using the requested accelerator; boot() awaits SSH."""
        self.scratch, self.output = scratch, output
        self.port = 0
        self.process: subprocess.Popen
        _run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(scratch / "key")]
        )
        pubkey = (scratch / "key.pub").read_text().strip()
        (scratch / "user-data").write_text(
            "#cloud-config\n"
            + json.dumps(
                {
                    "disable_root": False,
                    "ssh_pwauth": False,
                    "users": [{"name": "root", "ssh_authorized_keys": [pubkey]}],
                    "write_files": [
                        {
                            "path": "/etc/ssh/sshd_config.d/10-lmcache.conf",
                            "content": "PermitRootLogin prohibit-password\n",
                        }
                    ],
                    "runcmd": [["systemctl", "restart", "ssh"]],
                }
            )
        )
        (scratch / "meta-data").write_text(
            json.dumps(
                {"instance-id": str(uuid.uuid4()), "local-hostname": "lmcache-devdax"}
            )
        )
        _run(
            [
                "cloud-localds",
                str(scratch / "seed.img"),
                str(scratch / "user-data"),
                str(scratch / "meta-data"),
            ]
        )
        command = shlex.split("""
            qemu-system-x86_64 -M q35,cxl=on -display none -no-reboot
            -smbios type=1,product=LMCache-DevDAX-QEMU
            -netdev user,id=net0,hostfwd=tcp:127.0.0.1:0-:22
            -device virtio-net-pci,netdev=net0,bus=pcie.0
            -device virtio-blk-pci,drive=os,bus=pcie.0
            -device virtio-blk-pci,drive=seed,bus=pcie.0
        """)
        command += [
            "-accel",
            accel,
            "-cpu",
            "host" if accel == "kvm" else "max",
            "-m",
            f"{MANIFEST['guest_memory_mib']},maxmem=16G,slots=8",
            "-smp",
            str(MANIFEST["vcpus"]),
            "-drive",
            f"file={disk},format=qcow2,if=none,id=os",
            "-drive",
            f"file={scratch}/seed.img,format=raw,if=none,id=seed",
            "-serial",
            f"file:{output}/qemu-console.log",
            "-qmp",
            f"unix:{scratch}/qmp.sock,server=on,wait=off",
            *topology(),
        ]
        if kernel is not None:
            command += [
                "-kernel",
                str(kernel),
                "-append",
                "root=/dev/vda1 console=ttyS0 rw",
            ]
        (output / "qemu-command.json").write_text(json.dumps(command, indent=2))
        with (output / "qemu.log").open("w") as log:
            self.process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            )

    def boot(self, timeout: int) -> None:
        """Wait for QMP, discover QEMU's ephemeral host port, then await SSH."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("QEMU exited before guest boot; see qemu.log")
            try:
                with socket.socket(socket.AF_UNIX) as qmp:
                    qmp.settimeout(5)
                    qmp.connect(str(self.scratch / "qmp.sock"))
                    stream = qmp.makefile("rwb")
                    stream.readline()
                    for command in (
                        {"execute": "qmp_capabilities"},
                        {
                            "execute": "human-monitor-command",
                            "arguments": {"command-line": "info usernet"},
                        },
                    ):
                        stream.write(json.dumps(command).encode() + b"\n")
                        stream.flush()
                        while True:
                            response = json.loads(stream.readline())
                            if "return" in response or "error" in response:
                                break
                    match = re.search(
                        r"TCP\[HOST_FORWARD\].*?127\.0\.0\.1\s+(\d+)",
                        response.get("return", ""),
                    )
                    if not match:
                        raise RuntimeError(f"cannot discover SSH port: {response}")
                    self.port = int(match[1])
                    break
            except (OSError, ValueError):
                time.sleep(0.2)
        if not self.port:
            raise TimeoutError("QMP startup deadline exceeded")
        while time.monotonic() < deadline:
            try:
                self.ssh(
                    "true",
                    timeout=8,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except (subprocess.SubprocessError, OSError):
                if self.process.poll() is not None:
                    raise RuntimeError("QEMU exited during boot") from None
                time.sleep(1)
        raise TimeoutError("guest SSH boot deadline exceeded")

    def ssh(
        self, command: str, timeout: int = 60, **kwargs: Any
    ) -> subprocess.CompletedProcess:
        """Run a bounded guest command over this VM's private SSH endpoint."""
        return _run(
            shlex.split(
                "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                "-o ConnectTimeout=5 -o LogLevel=ERROR"
            )
            + [
                "-i",
                str(self.scratch / "key"),
                "-p",
                str(self.port),
                "-o",
                f"UserKnownHostsFile={self.scratch}/known_hosts",
                "root@127.0.0.1",
                command,
            ],
            timeout=timeout,
            **kwargs,
        )

    def close(self) -> None:
        """Stop only this instance's QEMU process, waiting before deleting scratch."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def main() -> None:
    """Prepare a cached image or run both suites with scoped cleanup and reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-image", action="store_true")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            os.environ.get("LMCACHE_DEVDAX_IMAGE", "/opt/lmcache-devdax/base.qcow2")
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="Checksum-verified downloaded cloud image (image preparation only)",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path(
            os.environ.get(
                "LMCACHE_DEVDAX_KERNEL", "/opt/lmcache-devdax/kernel/bzImage"
            )
        ),
    )
    args = parser.parse_args()
    controls_values = controls(dict(os.environ))
    accel = controls_values["LMCACHE_DEVDAX_QEMU_ACCEL"]
    signal.signal(signal.SIGALRM, _cancel)
    signal.alarm(3600 if args.prepare_image else (1800 if accel == "kvm" else 7200))
    output = ROOT / "artifacts/devdax-qemu" / f"run-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True)
    print(f"DevDAX artifacts: {output}", flush=True)
    summary = {
        name: {
            "state": "infrastructure failure"
            if controls_values["LMCACHE_DEVDAX_SUITE"] in (name, "both")
            else "not selected"
        }
        for name in ("l1", "l2")
    }
    code = 1
    guest = None
    try:
        for tool in (
            "qemu-system-x86_64",
            "qemu-img",
            "cloud-localds",
            "ssh",
            "ssh-keygen",
            "git",
        ):
            if not shutil.which(tool):
                raise RuntimeError(f"missing runner dependency: {tool}")
        version = subprocess.check_output(
            ["qemu-system-x86_64", "--version"], text=True
        )
        if version.split()[3] != MANIFEST["qemu_version"]:
            raise RuntimeError(f"QEMU version differs from manifest: {version}")
        if accel == "kvm":
            fd = os.open("/dev/kvm", os.O_RDWR)
            os.close(fd)
        image = args.image.resolve()
        if args.prepare_image:
            if image.exists():
                raise ValueError("refusing to overwrite an existing prepared image")
            if not args.base or _sha(args.base) != MANIFEST["rootfs_sha256"]:
                raise ValueError("--base must match the manifest rootfs checksum")
            image.parent.mkdir(parents=True, exist_ok=True)
        else:
            kernel = args.kernel.resolve()
            for asset in (image, kernel):
                if not asset.is_file():
                    raise FileNotFoundError(asset)
                expected = Path(str(asset) + ".sha256").read_text().split()[0]
                if _sha(asset) != expected:
                    raise ValueError(f"checksum mismatch: {asset}")
        with tempfile.TemporaryDirectory(prefix="lmcache-dax-") as directory:
            scratch = Path(directory)
            disk = scratch / "overlay.qcow2"
            _run(
                [
                    "qemu-img",
                    "create",
                    "-f",
                    "qcow2",
                    "-F",
                    "qcow2",
                    "-b",
                    str(args.base.resolve() if args.prepare_image else image),
                    str(disk),
                ]
            )
            if args.prepare_image:
                _run(["qemu-img", "resize", str(disk), "24G"])
            source = package_source(scratch / "source.tar.gz")
            source["image_manifest"] = MANIFEST
            source["image_sha256"] = _sha(args.base if args.prepare_image else image)
            if not args.prepare_image:
                source["kernel_sha256"] = _sha(kernel)
            (output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
            guest = Guest(
                scratch, output, disk, accel, None if args.prepare_image else kernel
            )
            try:
                guest.boot(
                    int(
                        os.environ.get(
                            "LMCACHE_DEVDAX_BOOT_TIMEOUT",
                            "300" if accel == "kvm" else "1200",
                        )
                    )
                )
                guest.ssh("cloud-init status --wait", timeout=300)
                with (scratch / "source.tar.gz").open("rb") as archive:
                    guest.ssh("cat > /root/source.tar.gz", stdin=archive, timeout=120)
                digest = source["archive_sha256"]
                guest.ssh(
                    f"echo '{digest}  /root/source.tar.gz' | sha256sum -c - && "
                    "mkdir -p /root/source && "
                    "tar xzf /root/source.tar.gz -C /root/source"
                )
                if args.prepare_image:
                    with (output / "image-build.log").open("w") as log:
                        guest.ssh(
                            "cd /root/source && "
                            "bash .buildkite/k3_tests/devdax/image/provision.sh",
                            timeout=2400,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    guest.ssh("rm -rf /root/source /root/source.tar.gz && sync")
                    guest.ssh("shutdown -h now", timeout=30)
                    guest.process.wait(timeout=60)
                    _run(
                        ["qemu-img", "convert", "-O", "qcow2", str(disk), str(image)],
                        timeout=180,
                    )
                    Path(str(image) + ".sha256").write_text(_sha(image) + "\n")
                    code = 0
                else:
                    with (output / "guest-setup.log").open("w") as log:
                        guest.ssh(
                            "cd /root/source && "
                            "/opt/lmcache-test/bin/python "
                            ".buildkite/k3_tests/devdax/guest-setup.py",
                            timeout=180,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    suite = shlex.quote(controls_values["LMCACHE_DEVDAX_SUITE"])
                    with (output / "guest-test.log").open("w") as log:
                        try:
                            guest.ssh(
                                f"cd /root/source && LMCACHE_DEVDAX_SUITE={suite} "
                                f"LMCACHE_DEVDAX_QEMU_ACCEL={accel} "
                                "/opt/lmcache-test/bin/python "
                                ".buildkite/k3_tests/devdax/guest_test.py",
                                timeout=1300 if accel == "kvm" else 5400,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                            code = 0
                        except subprocess.CalledProcessError as exc:
                            code = exc.returncode
            finally:
                if guest.port and not args.prepare_image:
                    try:
                        guest.ssh(
                            "mkdir -p /root/source/artifacts/devdax-qemu; "
                            "cd /root/source/artifacts/devdax-qemu; "
                            "dmesg > guest-dmesg.log; "
                            "cp -r /opt/lmcache-image image; tar czf - .",
                            timeout=30,
                            stdout=(scratch / "reports.tar.gz").open("wb"),
                            stderr=subprocess.DEVNULL,
                        )
                        with tarfile.open(scratch / "reports.tar.gz") as reports:
                            reports.extractall(output, filter="data")
                    except (
                        OSError,
                        subprocess.SubprocessError,
                        tarfile.TarError,
                    ) as exc:
                        print(f"Guest log collection failed: {exc}", flush=True)
                        code = code or 1
                guest.close()
            if not args.prepare_image:
                if (output / "versions.json").is_file():
                    versions = json.loads((output / "versions.json").read_text())
                    versions["qemu"] = version.splitlines()[0]
                    (output / "versions.json").write_text(
                        json.dumps(versions, indent=2) + "\n"
                    )
                required = [
                    "summary.json",
                    "devices.json",
                    "versions.json",
                    "guest-dmesg.log",
                    "cxl-list.json",
                    "daxctl-list.json",
                    "kernel.config",
                ]
                missing = [name for name in required if not (output / name).is_file()]
                if missing:
                    code = code or 1
                if (output / "summary.json").exists():
                    summary = json.loads((output / "summary.json").read_text())
                    for name in ("l1", "l2"):
                        if controls_values["LMCACHE_DEVDAX_SUITE"] in (name, "both"):
                            suite_reports = [
                                f"junit-{name}.xml",
                                f"pytest-{name}.log",
                                f"collected-{name}.json",
                            ]
                            if missing or any(
                                not (output / p).is_file() for p in suite_reports
                            ):
                                summary[name]["state"] = "infrastructure failure"
                            if summary[name]["state"] != "passed":
                                code = code or 1
    except (Exception, KeyboardInterrupt) as exc:
        print(f"DevDAX infrastructure failure: {exc}", flush=True)
        (output / "infrastructure-error.txt").write_text(str(exc) + "\n")
        code = code or 1
    finally:
        signal.alarm(0)
        if guest is not None:
            guest.close()
        if not args.prepare_image:
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if os.environ.get("BUILDKITE") and shutil.which("buildkite-agent"):
            subprocess.run(
                [
                    "buildkite-agent",
                    "annotate",
                    "--context",
                    "devdax-qemu",
                    "--style",
                    "success" if code == 0 else "error",
                    f"DevDAX {controls_values['LMCACHE_DEVDAX_SUITE']}: "
                    f"{'passed' if code == 0 else 'FAILED'}. "
                    f"Artifacts: {output.relative_to(ROOT)}",
                ],
                timeout=15,
            )
    raise SystemExit(code)


def _cancel(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"cancelled by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _cancel)
    main()
