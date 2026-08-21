from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from eval.phase07_ann_campaign import execute, validate_request

def _request(stage="screening"):
    p={"schema_version":1,"stage":stage,"request_id":"request-1","environment":{},"model_manifest_sha256":"a"*64,"corpus_manifest_sha256":"b"*64}
    if stage=="confirmation": p.update(prior_screening_sha256="c"*64,nominated_m=[16],run_ordinal=1,run_identity={"run_id":"1","run_attempt":1,"job_id":"2","job_allocation_nonce":"n"*16})
    if stage=="continuation": p.update(mode="stage2_sq",prior_evidence_sha256="c"*64)
    return p

def test_screening_creates_sealed_owned_artifacts_and_fixed_plan(tmp_path: Path):
    result=execute(_request(),tmp_path)
    assert result["authorization"]=="none"
    assert (tmp_path/"screening-request.json").is_file() and (tmp_path/"screening-ledger.json").is_file()
    assert result["result"]["index"]["m"]==[16,20,32]
    assert result["result"]["replicates"]==3

def test_rejects_unknown_secret_and_unbounded_continuation(tmp_path: Path):
    bad=_request(); bad["token"]="nope"
    with pytest.raises(ValueError): validate_request(bad)
    bad=_request("continuation"); bad["mode"]="production-selection"
    with pytest.raises(ValueError): execute(bad,tmp_path)
