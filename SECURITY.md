# Security

Please report security problems privately through
[GitHub's private vulnerability reporting](https://github.com/ATHARV-IIT-BOMBAY/meow-decoder/security/advisories/new),
not in a public issue. You'll get a reply within a week.

Useful context: `meow serve` has no logins. Each browser only sees its own cats, using a random ID it keeps, and
the server refuses requests without that ID whenever it listens beyond this machine. Uploaded audio is decoded by
ffmpeg with file and network access switched off and a 20-second limit. If you host it publicly, put it behind
https and see "Deploy it" in the README.
