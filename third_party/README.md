# Third-party source dependencies

`metadrive/` is an editable source checkout of [`metadriverse/metadrive`](https://github.com/metadriverse/metadrive). It is intentionally ignored by this repository. On this Windows host it was downloaded from GitHub's official commit ZIP because Git smart-HTTP cloning stalled repeatedly.

To recreate the documented source tree:

```bash
curl -L -o metadrive.zip https://codeload.github.com/metadriverse/metadrive/zip/85e5dadc6c7436d324348f6e3d8f8e680c06b4db
```

The expected checkout commit for the local editable MetaDrive source tree is `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. This records the repository's expected local revision; it is not a package-manager-enforced dependency pin.
