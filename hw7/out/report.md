# HW7 Package Vulnerability Mini-Advisor Report

- Target: ssh
- Service unit: ssh.service

## Candidate packages

- openssh-server 1:10.0p1-7

## Extracted CVE references (best-effort)

- CVE-2025-32728
- CVE-2025-26465
- CVE-2025-26466
- CVE-2024-39894
- CVE-2006-5051
- CVE-2023-48795
- CVE-2023-51384
- CVE-2023-51385
- CVE-2023-38408
- CVE-2023-28531
- CVE-2021-41617
- CVE-2021-28041

## Security-relevant changelog lines (subset)

```
* Ensure that configure knows the path to passwd; fixes reproducibility of
- CVE-2025-32728: sshd(8): fix the DisableForwarding directive, which
has been traffic on a X11 forwarding channel recently. Should fix X11
- ssh(1), sshd(8): fix regression in openssh-9.8, which would fail to
- sftp(1), ssh(1): fix a number of possible NULL dereference bugs,
- sshd(8): fix PerSourcePenalty incorrectly using "crash" penalty when
- sshd(8): fix "Match invalid-user" from incorrectly being activated in
- sshd(8): fix debug logging of user specific delay.
- ssh(1): fix argument of "Compression" directive in ssh -G config dump,
- sshd(8): fix a corner-case triggered by UpdateHostKeys when sshd
- CVE-2025-26465: ssh(1) in OpenSSH versions 6.8p1 to 9.9p1 (inclusive)
- CVE-2025-26466: sshd(8) in OpenSSH versions 9.5p1 to 9.9p1 (inclusive)
is vulnerable to a memory/CPU denial-of-service related to the
- ssh(1), sshd(8): fix regression in Match directive that caused
- sshd(8): fix the "Match invalid-user" predicate, which was matching
* Fix mlkem768x25519-sha256 key exchange algorithm on big-endian
- sshd(8): fix regression introduced in openssh-9.8 that swapped the
- sshd(8): fix minor memory leak in Subsystem option parsing.
- ssh(1): fix proxy multiplexing (-O proxy) bug. If a mux started with
Fixes an issue where the SSH_CONNECTION_ABANDON event was not
- Fix detection of setres*id on GNU/Hurd.
- CVE-2024-39894: Fix Logic error in ssh(1) ObscureKeystrokeTiming that
- misc: fix a number of inaccuracies in the PROTOCOL.* documentation
- sftp-server(8): fix home-directory extension implementation, it
- regression tests: fix rekey test that was testing the same KEX
- build: fix OpenSSL ED25519 support detection. An incorrect function
a minimal workaround for a regression from CVE-2006-5051.
* Stop reading ~/.pam_environment, which has a history of security
* Fix gssapi-keyex declaration further (thanks, Andreas Hasenack;
needed now that Twisted is fixed).
- ssh(1): fix the multiplexing "channel proxy" mode, broken when
- ssh(1), sshd(8): fix spurious configuration parsing errors when
- ssh-agent(1): fix potential spin in signal handler (bz3670)
- Many fixes to manual pages and other documentation.
* Fix gssapi-keyex declaration, broken when rebasing onto 8.9p1
https://bugzilla.mindrot.org/show_bug.cgi?id=3645; fixes build on
- [CVE-2023-48795] ssh(1), sshd(8): implement protocol extensions to
- [CVE-2023-51384] ssh-agent(1): when adding PKCS#11-hosted private keys
- [CVE-2023-51385] ssh(1): if an invalid user or hostname that contained
fixed intervals (default: every 20ms) when there is only a small
```

## Remediation command plan (printed only, not executed)

```
sudo apt-get update
sudo apt-get install --only-upgrade openssh-server
sudo systemctl restart ssh.service
```

## Known limitations

- Scan mode is deterministic-only (no LLM used).
- Package and service detection is heuristic (may miss non-standard SSH servers).
- Changelog parsing is best-effort and does not prove vulnerability presence or absence.
- No upgrades or restarts are executed (command plan only).
- On some distros (including Kali), apt changelog may be unavailable; local /usr/share/doc fallback is used.
