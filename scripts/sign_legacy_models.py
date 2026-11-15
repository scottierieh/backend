"""sign_legacy_models.py — one-off: sign model artifacts written before signing existed.

    SKARI_MODEL_SIGNING_KEY=... python scripts/sign_legacy_models.py            # dry run
    SKARI_MODEL_SIGNING_KEY=... python scripts/sign_legacy_models.py --apply
    ... --owners owners.json      # optional {"<model_id>": "<firebase uid>"}
    ... --owners owners.json --fill-owners --apply   # also give signed, ownerless artifacts their owner

Since model_store.py started verifying an HMAC signature before joblib.load,
artifacts saved by older revisions answer 403 on /predict. Every object under
models/{model_id}/<ts>/pipeline.joblib was written by this service — the
bucket's storage.rules give clients no write access under models/ (default
deny) — so signing them in place is safe. Objects at any other path are left
alone. Run it with the SAME key the service is deployed with
(SKARI_MODEL_SIGNING_KEY, else SKARI_BACKEND_KEY).

Without an owner mapping the artifact is signed with owner "" — only the
service key can use it: since the 2026-11 audit (H3) /predict, like DELETE,
answers a browser user only for the account recorded as owner.

--fill-owners: artifacts that are ALREADY signed with owner "" (signed by this
script earlier, or trained while authentication was off) and whose model id
is in --owners are re-signed with that owner, after their signature has been
verified. Build owners.json from the registry records the app keeps
(Firestore modelLab2Models/{modelId}.userId); run it before deploying the H3
change so owners keep scoring their old models.
"""

import argparse
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import model_store  # noqa: E402

_PATH_RE = re.compile(r'^models/([A-Za-z0-9_-]{1,128})/[0-9]{1,20}/pipeline\.joblib$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write metadata (default: dry run)')
    ap.add_argument('--owners', help='JSON file mapping model_id -> owner uid')
    ap.add_argument('--fill-owners', action='store_true',
                    help='re-sign already-signed artifacts that have no owner, using --owners')
    args = ap.parse_args()
    if not (os.environ.get('SKARI_MODEL_SIGNING_KEY') or os.environ.get('SKARI_BACKEND_KEY')):
        sys.exit('Set SKARI_MODEL_SIGNING_KEY (or SKARI_BACKEND_KEY) to the deployed value first.')
    owners = json.load(open(args.owners)) if args.owners else {}

    bucket = model_store._bucket()
    signed = skipped = already = filled = 0
    for blob in bucket.list_blobs(prefix='models/'):
        m = _PATH_RE.match(blob.name)
        if not m:
            skipped += 1
            continue
        if (blob.metadata or {}).get(model_store._META_SIG):
            new_owner = owners.get(m.group(1), '') if args.fill_owners else ''
            try:
                current = model_store._verify(blob.name, blob.metadata)
            except ValueError:
                current = None  # not ours to vouch for: leave it
            if new_owner and current == '':
                sha = blob.metadata.get(model_store._META_SHA)
                if hashlib.sha256(blob.download_as_bytes()).hexdigest() != sha:
                    print('skip (content does not match its signature) ' + blob.name)
                    already += 1
                    continue
                meta = dict(blob.metadata)
                meta.update({model_store._META_OWNER: new_owner,
                             model_store._META_SIG: model_store._sign(blob.name, new_owner, sha)})
                print(('fill ' if args.apply else 'would fill ') + blob.name + f' (owner {new_owner})')
                if args.apply:
                    blob.metadata = meta
                    blob.patch()
                filled += 1
                continue
            already += 1
            continue
        owner = owners.get(m.group(1), '')
        sha = hashlib.sha256(blob.download_as_bytes()).hexdigest()
        meta = dict(blob.metadata or {})
        meta.update({model_store._META_OWNER: owner, model_store._META_SHA: sha,
                     model_store._META_SIG: model_store._sign(blob.name, owner, sha)})
        print(('sign ' if args.apply else 'would sign ') + blob.name + (f' (owner {owner})' if owner else ''))
        if args.apply:
            blob.metadata = meta
            blob.patch()
        signed += 1
    print(f'{signed} {"signed" if args.apply else "to sign"}, '
          f'{filled} {"given an owner" if args.apply else "to give an owner"}, {already} already signed, '
          f'{skipped} other objects left alone')


if __name__ == '__main__':
    main()
